"""Cell controller for the packaging / palletizing line.

Runs outside CoppeliaSim through the ZMQ remote API in stepped mode:

    python -m factory.controller                 # run until Ctrl+C
    python -m factory.controller --time 180      # run 180 simulated seconds
    python -m factory.controller --oracle        # skip the CNN, use ground-truth labels

Every robot is a small coroutine (Python generator) that is advanced once per simulation
step.  All motion is kinematic: the coroutines move IK *target* dummies, the in-scene
`FactoryBridge` script solves the inverse kinematics, and parts are attached to grippers
by re-parenting (the usual way vacuum grippers are simulated).
"""
import argparse
import collections
import math
import random
import time

import cv2

from coppeliasim_zmqremoteapi_client import RemoteAPIClient

from . import layout as L
from .vision import Classifier, grab

PI = math.pi


def yaw_quat(yaw):
    return [0.0, 0.0, math.sin(yaw / 2), math.cos(yaw / 2)]


def smooth(t):
    return t * t * t * (t * (6 * t - 15) + 10)      # quintic ease in/out


def wrap(a):
    return (a + PI) % (2 * PI) - PI


class Tool:
    """An IK target.  State is (x, y, z, yaw); `center` switches to cylindrical interpolation
    around a robot base so that big arms swing around their axis instead of cutting through it."""

    def __init__(self, cell, name, center=None):
        self.cell = cell
        self.name = name
        self.target = cell.sim.getObject('/%s_target' % name)
        self.tip = cell.sim.getObject('/%s_tip' % name)
        p = cell.sim.getObjectPosition(self.target, -1)
        self.yaw = cell.sim.getObjectOrientation(self.target, -1)[2]
        self.pos = list(p)
        self.center = center
        if center:
            self.theta = math.atan2(p[1] - center[1], p[0] - center[0])

    def cyl(self, p):
        """(x, y, z) -> (r, theta, z) with theta unwrapped next to the current angle."""
        dx, dy = p[0] - self.center[0], p[1] - self.center[1]
        th = math.atan2(dy, dx)
        th = self.theta + wrap(th - self.theta)
        return [math.hypot(dx, dy), th, p[2]]

    def set(self, p, yaw):
        self.pos, self.yaw = list(p), yaw
        self.cell.frame[self.target] = list(p) + yaw_quat(yaw)


class Cell:
    def __init__(self, oracle=False, seed=1, realtime=False):
        self.client = RemoteAPIClient()
        self.sim = sim = self.client.require('sim')
        self.rnd = random.Random(seed)
        self.realtime = realtime
        self.bridge = sim.getObject('/FactoryBridge/script')
        self.dt = sim.getSimulationTimeStep()
        self.t = 0.0
        self.frame = {}
        self.spawned = []

        g = sim.getObject
        self.tpl = {c: [g('/tpl_%s_%02d' % (c, v)) for v in range(L.N_VARIANTS)] for c in L.CLASSES}
        self.tpl_carton, self.tpl_lid, self.tpl_pallet = g('/tpl_carton'), g('/tpl_lid'), g('/tpl_pallet')
        self.stripes = {n: g('/Belt' + n) for n in ('A', 'B')}
        self.belt_pos = {'A': 0.0, 'B': 0.0}
        self.amr = g('/AMR')
        self.amr_z = sim.getObjectPosition(self.amr, -1)[2]
        self.cams = {'qc': g('/QC_cam'), 'sort': g('/Sort_cam')}
        self.classifier = Classifier.load()
        if self.classifier is None:
            print('[cell] no trained CNN found -> ground-truth labels (oracle mode)')

        # ---- operator setpoints and cycle-stop requests (changed live from the dashboard) ----
        self.set = dict(feed_rate=52.0,                 # bars per minute requested from the feeder
                        belt_speed=L.A_SPEED,           # conveyor A, m/s
                        rework_pct=11.0, scrap_pct=9.0, # share of defective bars in the supply
                        layers=L.PALLET_LAYERS,         # carton layers on the next pallet
                        mode='oracle' if (oracle or self.classifier is None) else 'cnn')
        self.hold = dict(feeder=False, delta=False, palletizer=False, pallet_robot=False, amr=False, sorter=False)
        self.status = dict(feeder='RUN', delta='IDLE', palletizer='IDLE', pallet_robot='IDLE', amr='DOCKED', sorter='IDLE')
        self.events = collections.deque(maxlen=80)      # (id, t, level, message)
        self.event_id = 0
        self.scans = collections.deque(maxlen=48)       # inspection records, newest last
        self.scan_png = {}                              # scan id -> PNG bytes
        self.scan_id = 0
        self.serial = 0
        self.history = []                               # (t, bars in, packed, rejects, cartons) every 2 s
        self.tasks = []

        self.delta = Tool(self, 'Delta')
        self.ur = Tool(self, 'UR10', center=L.UR_BASE)
        self.irb = Tool(self, 'IRB4600', center=L.IRB_BASE)
        self.bax = {'L': Tool(self, 'BaxterL'), 'R': Tool(self, 'BaxterR')}

        # ---- line state ------------------------------------------------
        self.a_offset = 0.0          # distance travelled by belt A
        self.a_blocked = False
        self.products = []           # dicts, in spawn order
        self.end_slot = None         # product waiting in front of Baxter
        self.last_spawn = -1e9
        self.cartons = []            # on conveyor B, downstream first
        self.station_pallet = None   # pallet being loaded
        self.station_busy = False    # IRB is working at the station
        self.ur_clear = True
        self.stack = []
        self.amr_x = L.DOCK[0]
        self.amr_state = 'docked'    # docked | loaded | away
        self.shipping = []           # pallets on the outbound roller conveyor
        self.bins = {'rework': [], 'scrap': []}
        self.last_spawn_offset = -1e9
        self.amr_load = None
        self.stats = self.zero_stats()

    @staticmethod
    def zero_stats():
        return dict(spawned=0, packed=0, cartons=0, pallets=0, shipped=0, rejects=0, qc_total=0, qc_correct=0,
                    missed=0, qc_good=0, qc_rework=0, qc_scrap=0, sorted_rework=0, sorted_scrap=0,
                    sort_total=0, sort_correct=0)

    # ------------------------------------------------------------ sim helpers
    def spawn(self, tpl, pos, yaw=0.0, parent=-1):
        sim = self.sim
        h = sim.copyPasteObjects([tpl], 0)[0]
        sim.setObjectParent(h, -1, True)
        sim.setObjectPose(h, list(pos) + yaw_quat(yaw), -1)
        if parent != -1:
            sim.setObjectParent(h, parent, True)
        self.spawned.append(h)
        return h

    def remove(self, handles):
        for h in handles:
            if h in self.spawned:
                self.spawned.remove(h)
        self.sim.removeObjects(list(handles))

    def attach(self, h, parent, rel_pos, rel_yaw=0.0):
        self.sim.setObjectParent(h, parent, True)
        self.sim.setObjectPose(h, list(rel_pos) + yaw_quat(rel_yaw), parent)

    def place(self, h, pos, yaw):
        self.sim.setObjectParent(h, -1, True)
        self.sim.setObjectPose(h, list(pos) + yaw_quat(yaw), -1)

    def belt(self, name, vel):
        """Animate a belt surface (its stripe pattern wraps around every pitch)."""
        if vel:
            self.belt_pos[name] += vel * self.dt
            self.frame[self.stripes[name]] = [self.belt_pos[name] % L.BELT_STRIPE_PITCH, 0, 0, 0, 0, 0, 1]

    def log(self, msg, level='info'):
        self.event_id += 1
        self.events.append((self.event_id, round(self.t, 1), level, msg))
        print('[%7.2fs] %s' % (self.t, msg), flush=True)

    # --------------------------------------------------------------- motions
    def wait(self, seconds):
        for _ in range(max(1, int(round(seconds / self.dt)))):
            yield

    def move(self, tool, goal, yaw=None, speed=1.0, arc=0.0, T=None, min_t=0.25, theta=None, wrap_yaw=True):
        """Move `tool` to goal (x, y, z) -- or to a moving goal if `goal` is callable."""
        fn = goal if callable(goal) else (lambda: goal)
        g = fn()
        yaw0 = tool.yaw
        yaw1 = yaw0 if yaw is None else yaw0 + wrap(yaw - yaw0) if wrap_yaw else yaw
        if tool.center:
            s0 = tool.cyl(tool.pos)
            g0 = tool.cyl(g)
            if theta is not None:
                g0[1] = theta
            r_mean = (s0[0] + g0[0]) / 2
            dist = math.hypot(math.hypot(g0[0] - s0[0], r_mean * (g0[1] - s0[1])), g0[2] - s0[2])
        else:
            s0 = list(tool.pos)
            dist = math.dist(s0, g)
        dist += 2 * arc + 0.15 * abs(yaw1 - yaw0)
        if T is None:
            T = max(min_t, dist / speed)
        n = max(1, int(round(T / self.dt)))
        for i in range(1, n + 1):
            s = smooth(i / n)
            g = fn()
            if tool.center:
                gc = g0
                r = s0[0] + (gc[0] - s0[0]) * s
                th = s0[1] + (gc[1] - s0[1]) * s
                tool.theta = th
                p = [tool.center[0] + r * math.cos(th), tool.center[1] + r * math.sin(th), s0[2] + (gc[2] - s0[2]) * s]
            else:
                p = [s0[k] + (g[k] - s0[k]) * s for k in range(3)]
            p[2] += arc * 4 * s * (1 - s)
            tool.set(p, yaw0 + (yaw1 - yaw0) * s)
            yield

    def move_cyl(self, tool, r, theta, z, yaw=None, **kw):
        """Cylindrical goal with an explicit (unwrapped) angle -- no shortest-way guessing."""
        c = tool.center
        yield from self.move(tool, [c[0] + r * math.cos(theta), c[1] + r * math.sin(theta), z], yaw=yaw, theta=theta, wrap_yaw=False, **kw)

    # ------------------------------------------------------------ conveyor A
    def prod_x(self, p):
        return p['x0'] + self.a_offset

    def infeed(self):
        """Feeder + belt A + QC camera gate."""
        sim = self.sim
        carrier = sim.getObject('/CarrierA')
        z = L.BELT_Z + L.PROD[2] / 2
        run_time = 0.0
        min_gap = 0.17                 # keeps a single bar in the QC camera's field of view
        while True:
            v = self.set['belt_speed']
            # the belt halts if a reject would run into an occupied end stop
            lead = next((p for p in self.products if p['state'] == 'belt'), None)
            self.a_blocked = bool(lead and self.end_slot and self.prod_x(lead) + v * self.dt >= L.A_END_X - 0.09)
            self.status['feeder'] = 'BLOCKED' if self.a_blocked else 'HOLD' if self.hold['feeder'] else 'RUN'
            if not self.a_blocked:
                self.belt('A', v)
                self.a_offset += v * self.dt
                run_time += self.dt
                self.frame[carrier] = [self.a_offset, 0, 0, 0, 0, 0, 1]
                due = run_time - self.last_spawn >= 60.0 / max(1.0, self.set['feed_rate'])
                if due and not self.hold['feeder'] and self.a_offset - self.last_spawn_offset >= min_gap:
                    self.last_spawn, self.last_spawn_offset = run_time, self.a_offset
                    rw, sc = self.set['rework_pct'] / 100.0, self.set['scrap_pct'] / 100.0
                    cls = self.rnd.choices(L.CLASSES, (max(0.0, 1.0 - rw - sc), rw, sc))[0]
                    y = L.A_Y + self.rnd.uniform(-0.07, 0.07)
                    yaw = self.rnd.uniform(-PI, PI)
                    h = self.spawn(self.rnd.choice(self.tpl[cls]), [L.A_SPAWN_X, y, z], yaw, carrier)
                    self.serial += 1
                    self.products.append(dict(h=h, cls=cls, pred=None, x0=L.A_SPAWN_X - self.a_offset, y=y,
                                              yaw=yaw, state='belt', serial=self.serial))
                    self.stats['spawned'] += 1
            for p in self.products:
                if p['state'] != 'belt':
                    continue
                x = self.prod_x(p)
                if p['pred'] is None and x >= L.QC_X:
                    p['qc_pending'] = True          # image is taken after this frame is rendered
                if x >= L.A_END_X and self.end_slot is None:
                    p['state'] = 'at_end'
                    self.place(p['h'], [L.A_END_X, p['y'], z], p['yaw'])
                    self.end_slot = p
                    if p['pred'] == 'good':
                        self.stats['missed'] += 1
                        self.log('good bar #%d was not picked and reached the reject stop' % p['serial'], 'warn')
            self.products = [p for p in self.products if p['state'] != 'done']
            yield

    def classify(self, cam, p):
        """Take a picture of bar p, grade it and file the scan for the dashboard."""
        img = grab(self.sim, self.cams[cam])
        if self.set['mode'] == 'cnn' and self.classifier is not None:
            label, conf = self.classifier.predict(img)
        else:
            label, conf = p['cls'], 1.0
        self.scan_id += 1
        self.scan_png[self.scan_id] = cv2.imencode('.png', img[:, :, ::-1])[1].tobytes()
        if len(self.scans) == self.scans.maxlen:
            self.scan_png.pop(self.scans[0]['id'], None)
        self.scans.append(dict(id=self.scan_id, serial=p['serial'], t=round(self.t, 1), cam=cam, pred=label,
                               conf=round(conf, 3), truth=p['cls'], source=self.set['mode']))
        return label, conf

    def after_step(self):
        """Work that needs the freshly rendered frame (camera triggers)."""
        for p in self.products:
            if p.pop('qc_pending', False):
                p['pred'], conf = self.classify('qc', p)
                self.stats['qc_total'] += 1
                self.stats['qc_' + p['pred']] += 1
                self.stats['qc_correct'] += p['pred'] == p['cls']
                if p['pred'] != p['cls']:
                    self.log('QC camera graded bar #%d as %s (%.0f%%) but it is %s'
                             % (p['serial'], p['pred'], conf * 100, p['cls']), 'warn')

    # ------------------------------------------------------------ conveyor B
    def carton_line(self):
        """Carton erector, fill-station stopper, case sealer and accumulation at the end stop."""
        v = L.B_SPEED
        gap = L.CARTON[0] + 0.03
        z = L.BELT_Z
        while True:
            self.belt('B', -v)
            cs = self.cartons
            if not cs or (cs[-1]['x'] <= L.B_SPAWN_X - L.B_PITCH and sum(c['x'] > L.B_FILL_X for c in cs) < 2):
                h = self.spawn(self.tpl_carton, [L.B_SPAWN_X, L.B_Y, z])
                cs.append(dict(h=h, x=L.B_SPAWN_X, count=0, full=False, sealed=False, state='belt', reserved=0))
            for i, c in enumerate(cs):
                if c['state'] != 'belt':
                    continue
                limit = L.B_END_X
                if not c['full']:
                    limit = L.B_FILL_X
                if i > 0 and cs[i - 1]['state'] in ('belt', 'ready'):
                    limit = max(limit, cs[i - 1]['x'] + gap)
                nx = max(limit, c['x'] - v * self.dt)
                if nx != c['x']:
                    c['x'] = nx
                    self.frame[c['h']] = [nx, L.B_Y, z, 0, 0, 0, 1]
                if c['full'] and not c['sealed'] and c['x'] <= L.B_SEAL_X:
                    c['sealed'] = True
                    lid = self.spawn(self.tpl_lid, [c['x'], L.B_Y, z + L.CARTON[2]], 0.0, c['h'])
                    self.spawned.remove(lid)      # removed together with its carton
                if c['x'] <= L.B_END_X:
                    c['state'] = 'ready'
            yield

    def fill_carton(self):
        return next((c for c in self.cartons if c['state'] == 'belt' and not c['full']
                     and abs(c['x'] - L.B_FILL_X) < 1e-6), None)

    # ----------------------------------------------------------------- delta
    def delta_task(self):
        tool = self.delta
        home = list(L.DELTA_TCP_HOME)
        zp = L.BELT_Z + L.PROD[2]
        hop = 0.07
        t_hop = 0.40
        idle = True
        while True:
            c = self.fill_carton()
            cand = None
            if self.hold['delta']:
                self.status['delta'] = 'HOLD'
            elif c and c['count'] + c['reserved'] < len(L.SLOTS):
                # most downstream good bar that will still be inside the pick window on arrival
                for p in self.products:
                    if p['state'] == 'belt' and p['pred'] == 'good':
                        x = self.prod_x(p) + self.set['belt_speed'] * t_hop
                        if -0.40 <= x <= 0.27:
                            cand = p
                            break
                if cand is None:
                    self.status['delta'] = 'WAIT PRODUCT'
            else:
                self.status['delta'] = 'WAIT CARTON'
            if cand is None:
                if not idle:
                    yield from self.move(tool, home, yaw=0.0, speed=1.5)
                    idle = True
                yield
                continue
            idle = False
            p = cand
            p['state'] = 'claimed'
            c['reserved'] += 1
            self.status['delta'] = 'PICK'
            a = wrap(p['yaw'])
            a = a - PI if a > PI / 2 else a + PI if a < -PI / 2 else a      # bars are 180deg symmetric
            yield from self.move(tool, lambda: [self.prod_x(p), p['y'], zp], yaw=a, arc=hop, T=t_hop)
            self.attach(p['h'], tool.tip, [0, 0, -L.PROD[2] / 2], p['yaw'] - a)
            p['state'] = 'held'
            self.status['delta'] = 'PLACE'
            slot = L.SLOTS[c['count']]
            goal = [c['x'] + slot[0], L.B_Y + slot[1], L.BELT_Z + L.CARTON_FLOOR + L.PROD[2]]
            yield from self.move(tool, goal, yaw=0.0, arc=hop + 0.03, T=t_hop + 0.1)
            self.attach(p['h'], c['h'], [slot[0], slot[1], L.CARTON_FLOOR + L.PROD[2] / 2], 0.0)
            self.spawned.remove(p['h'])           # now owned by the carton
            p['state'] = 'done'
            c['count'] += 1
            c['reserved'] -= 1
            self.stats['packed'] += 1
            if c['count'] == len(L.SLOTS):
                c['full'] = True
                self.stats['cartons'] += 1

    # ------------------------------------------------------------- UR10 cobot
    def pallet_slot(self, n):
        """World pose of carton #n on the station pallet (bottom centre)."""
        gx, gy = L.PALLET_GRID[n % 4]
        layer = n // 4
        if layer % 2:                       # mirror every other layer
            gx, gy = -gx, -gy
        return [L.STATION[0] + gx, L.STATION[1] + gy, L.STATION[2] + L.PALLET[2] + layer * L.CARTON_H]

    def ur_task(self):
        tool = self.ur
        yaw0 = tool.yaw
        home = list(L.UR_HOME)
        pick = [L.B_END_X, L.B_Y, L.BELT_Z + L.CARTON_H]
        yield from self.move(tool, home, speed=0.8)
        while True:
            pal = self.station_pallet
            c = self.cartons[0] if self.cartons and self.cartons[0]['state'] == 'ready' and self.cartons[0]['sealed'] else None
            ready = pal and not self.station_busy and pal['count'] < pal['cap']
            if self.hold['palletizer'] or not (c and ready):
                self.status['palletizer'] = ('HOLD' if self.hold['palletizer'] else 'WAIT PALLET' if not ready
                                             else 'WAIT CARTON')
                yield
                continue
            self.ur_clear = False
            cap = pal['cap']
            self.status['palletizer'] = 'PICK'
            up = 0.16
            yield from self.move(tool, [pick[0], pick[1], pick[2] + up], yaw=yaw0, speed=1.8)
            yield from self.move(tool, pick, speed=0.9)
            self.attach(c['h'], tool.tip, [0, 0, -L.CARTON_H], 0.0)
            self.cartons.pop(0)
            yield from self.wait(0.05)
            yield from self.move(tool, [pick[0], pick[1], pick[2] + up], speed=1.0)
            self.status['palletizer'] = 'PLACE'
            slot = self.pallet_slot(pal['count'])
            top = [slot[0], slot[1], slot[2] + L.CARTON_H]
            yield from self.move(tool, [top[0], top[1], top[2] + up], yaw=yaw0 + PI / 2, speed=1.8, arc=0.08)
            yield from self.move(tool, top, speed=0.9)
            self.attach(c['h'], pal['h'], [slot[0] - L.STATION[0], slot[1] - L.STATION[1], slot[2] - L.STATION[2]], PI / 2)
            self.spawned.remove(c['h'])           # now owned by the pallet
            pal['count'] += 1
            yield from self.wait(0.05)
            yield from self.move(tool, [top[0], top[1], top[2] + up], speed=1.0)
            if pal['count'] == cap:
                yield from self.move(tool, home, yaw=yaw0, speed=1.8)
                pal['full'] = True
                self.stats['pallets'] += 1
                self.log('pallet #%d complete (%d cartons)' % (self.stats['pallets'], cap), 'good')
            self.ur_clear = True

    # ------------------------------------------------------- IRB 4600 + fork
    def restock(self):
        sx, sy, sz = L.STACK
        for i in range(L.STACK_COUNT):
            h = self.spawn(self.tpl_pallet, [sx, sy, sz + i * L.PALLET[2]], PI / 2)
            self.stack.append(dict(h=h, count=0, full=False))

    def irb_task(self):
        tool = self.irb
        R = L.IRB_R
        T_STACK, T_STATION, T_DOCK = PI / 2, 0.0, -PI / 2
        back = 0.46                       # radial retreat that clears the pallet
        carry_r = 1.20
        zc = 1.10                         # height used for carrying and for swinging around
        fast, slow = 2.0, 0.7

        def fork(theta, r, z, speed=fast):
            return self.move_cyl(tool, r, theta, z, yaw=theta, speed=speed)

        def pick(theta, z_bottom, pallet):
            z = z_bottom + L.PALLET_POCKET_Z
            yield from fork(theta, R - back, zc)                 # swing high, then come down behind the pallet
            yield from fork(theta, R - back, z - 0.012)
            yield from fork(theta, R, z - 0.012, slow)           # slide the tines into the pockets
            self.attach(pallet['h'], tool.tip, [0, 0, -L.PALLET_POCKET_Z], 0.0)
            yield from fork(theta, R, z + 0.14, slow)
            yield from fork(theta, carry_r, zc, slow)

        def put(theta, z_bottom, pallet, parent=-1):
            z = z_bottom + L.PALLET_POCKET_Z
            yield from fork(theta, carry_r, zc, 1.3)
            yield from fork(theta, R, z + 0.14, slow)
            yield from fork(theta, R, z, slow)
            pos = [L.IRB_BASE[0] + R * math.cos(theta), L.IRB_BASE[1] + R * math.sin(theta), z_bottom]
            self.place(pallet['h'], pos, theta)
            if parent != -1:
                self.sim.setObjectParent(pallet['h'], parent, True)
            yield from fork(theta, R, z - 0.012, slow)
            yield from fork(theta, R - back, z - 0.012, slow)
            yield from fork(theta, R - back, zc)

        home = (1.25, PI / 4, 1.25)
        st = self.status
        while True:
            if self.hold['pallet_robot']:
                st['pallet_robot'] = 'HOLD'
                yield
            elif self.station_pallet is None:                     # bring an empty pallet
                if not self.stack:
                    self.restock()
                    self.log('empty-pallet stack replenished')
                p = self.stack.pop()
                st['pallet_robot'] = 'FETCH EMPTY'
                yield from pick(T_STACK, L.STACK[2] + len(self.stack) * L.PALLET[2], p)
                while not self.ur_clear:
                    yield
                self.station_busy = True
                st['pallet_robot'] = 'PLACE EMPTY'
                yield from put(T_STATION, L.STATION[2], p)
                p['cap'] = 4 * int(self.set['layers'])
                self.station_pallet = p
                self.station_busy = False
                yield from fork(home[1], home[0], home[2])
            elif self.station_pallet['full']:                     # take the finished pallet to the AMR
                p = self.station_pallet
                self.station_busy = True
                st['pallet_robot'] = 'REMOVE FULL'
                yield from pick(T_STATION, L.STATION[2], p)
                self.station_pallet = None
                self.station_busy = False
                while self.amr_state != 'docked':
                    st['pallet_robot'] = 'WAIT AMR'
                    yield
                st['pallet_robot'] = 'LOAD AMR'
                yield from put(T_DOCK, L.AMR_TOP, p, parent=self.amr)
                self.amr_load = p
                self.amr_state = 'loaded'
            else:
                st['pallet_robot'] = 'IDLE'
                yield

    # ---------------------------------------------------- AMR + shipping lane
    def amr_task(self):
        y = L.DOCK[1]

        def drive(x1, speed=0.9):
            x0 = self.amr_x
            n = max(1, int(round(abs(x1 - x0) / speed / self.dt)))
            for i in range(1, n + 1):
                self.amr_x = x0 + (x1 - x0) * smooth(i / n)
                self.frame[self.amr] = [self.amr_x, y, self.amr_z] + yaw_quat(PI / 2)
                yield

        while True:
            if self.amr_state != 'loaded' or self.hold['amr']:
                self.status['amr'] = 'HOLD' if self.hold['amr'] else 'DOCKED'
                yield
                continue
            yield from self.wait(1.0)
            self.amr_state = 'away'
            self.status['amr'] = 'TO SHIPPING'
            self.log('AMR leaves the dock with a pallet of %d cartons' % self.amr_load['count'])
            yield from drive(L.AMR_UNLOAD_X)
            p = self.amr_load
            self.amr_load = None
            self.status['amr'] = 'UNLOADING'
            self.place(p['h'], [self.amr_x, y, L.AMR_TOP], -PI / 2)
            p['x'] = self.amr_x
            self.shipping.append(p)
            while p['x'] < L.SHIP_X0 + 0.35:      # wait until the pallet has rolled off the AMR
                yield
            self.status['amr'] = 'RETURNING'
            yield from drive(L.DOCK[0])
            self.amr_state = 'docked'

    def shipping_task(self):
        y = L.DOCK[1]
        keep = 3
        while True:
            for i, p in enumerate(self.shipping):
                limit = L.SHIP_X1 - 0.30 - i * (L.PALLET[1] + 0.12)
                if p['x'] < limit:
                    p['x'] = min(limit, p['x'] + L.SHIP_SPEED * self.dt)
                    self.frame[p['h']] = [p['x'], y, L.AMR_TOP] + yaw_quat(-PI / 2)
                elif not p.get('shipped'):
                    p['shipped'] = True
                    self.stats['shipped'] += 1
                    self.log('pallet delivered to shipping (%d total)' % self.stats['shipped'], 'good')
            if len(self.shipping) > keep and all(p.get('shipped') for p in self.shipping):
                old = self.shipping.pop(0)        # collected by the truck
                self.remove([old['h']])
            yield

    # ------------------------------------------------- Baxter: ML reject sorter
    def baxter_task(self):
        zp = L.BELT_Z + L.PROD[2]
        bins = {'rework': L.BIN_REWORK, 'scrap': L.BIN_SCRAP}
        arm_for = {'rework': 'R', 'scrap': 'L'}
        homes = {s: list(t.pos) for s, t in self.bax.items()}
        yaws = {s: t.yaw for s, t in self.bax.items()}
        for s, t in self.bax.items():           # tuck both arms next to the body
            homes[s] = [L.BAXTER[0] - 0.55, L.BAXTER[1] + (0.45 if s == 'R' else -0.45), 1.02]
            yield from self.move(t, homes[s], speed=0.6)
        while True:
            p = self.end_slot
            if p is None or p.get('sorting') or self.hold['sorter']:
                self.status['sorter'] = 'HOLD' if self.hold['sorter'] else 'IDLE'
                yield
                continue
            p['sorting'] = True
            self.status['sorter'] = 'INSPECT'
            yield                                  # let the bar settle, then look at it
            label, conf = self.classify('sort', p)
            dest = 'rework' if label == 'good' else label
            truth = 'rework' if p['cls'] == 'good' else p['cls']
            self.stats['rejects'] += 1
            self.stats['sorted_' + dest] += 1
            self.stats['sort_total'] += 1
            self.stats['sort_correct'] += dest == truth
            self.status['sorter'] = 'SORT ' + dest.upper()
            side = arm_for[dest]
            tool = self.bax[side]
            b = bins[dest]
            yield from self.move(tool, [L.A_END_X, p['y'], zp + 0.12], speed=1.4, arc=0.03)
            yield from self.move(tool, [L.A_END_X, p['y'], zp], speed=0.6)
            self.attach(p['h'], tool.tip, [0, 0, -L.PROD[2] / 2], p['yaw'] - yaws[side])
            p['state'] = 'done'
            self.end_slot = None
            yield from self.move(tool, [L.A_END_X, p['y'], zp + 0.14], speed=0.8)
            over = [b[0], b[1], b[2] + 0.20]
            yield from self.move(tool, over, speed=1.4, arc=0.05)
            items = self.bins[dest]
            if len(items) >= 12:                   # bin emptied by an operator
                self.remove(items)
                items.clear()
            k = len(items)
            drop = [b[0] + (k % 3 - 1) * 0.08, b[1] + ((k // 3) % 2 - 0.5) * 0.11, b[2] - 0.20 + 0.02 + (k // 6) * 0.032]
            self.place(p['h'], drop, self.rnd.uniform(-0.3, 0.3))
            items.append(p['h'])
            yield from self.move(tool, homes[side], speed=1.4, arc=0.04)

    # -------------------------------------------------------------- main loop
    def report(self):
        s = self.stats
        acc = 100.0 * s['qc_correct'] / s['qc_total'] if s['qc_total'] else float('nan')
        self.log('KPI  bars in=%d packed=%d | cartons=%d | pallets built=%d shipped=%d | rejects sorted=%d '
                 '| QC accuracy=%.1f%% (%d) | good bars missed=%d'
                 % (s['spawned'], s['packed'], s['cartons'], s['pallets'], s['shipped'], s['rejects'],
                    acc, s['qc_total'], s['missed']))

    def snapshot(self):
        """Plain-data picture of the line for the dashboard (no simulator calls)."""
        code = {None: 0, 'good': 1, 'rework': 2, 'scrap': 3}
        bars = [[round(self.prod_x(p) if p['state'] == 'belt' else L.A_END_X, 3), round(p['y'], 3), code[p['pred']]]
                for p in self.products if p['state'] in ('belt', 'at_end')]
        pal = self.station_pallet
        fill = self.fill_carton()
        r3 = lambda v: [round(v[0], 3), round(v[1], 3)]
        return dict(
            t=round(self.t, 2), stats=dict(self.stats), settings=dict(self.set), hold=dict(self.hold),
            status=dict(self.status), cnn_available=self.classifier is not None,
            wip=sum(p['state'] in ('belt', 'claimed', 'held', 'at_end') for p in self.products),
            bars=bars,
            cartons=[[round(c['x'], 3), c['count'], int(c['sealed'])] for c in self.cartons],
            fill_carton=fill['count'] if fill else None,
            pallet=dict(count=pal['count'], cap=pal['cap']) if pal else None,
            stack=len(self.stack),
            amr=dict(x=round(self.amr_x, 3), load=self.amr_load['count'] if self.amr_load else None),
            shipping=[[round(p['x'], 3), p['count']] for p in self.shipping],
            bins={k: len(v) for k, v in self.bins.items()},
            tools=dict(delta=r3(self.delta.pos), ur=r3(self.ur.pos), irb=r3(self.irb.pos),
                       bax_l=r3(self.bax['L'].pos), bax_r=r3(self.bax['R'].pos)),
            scans=list(self.scans), events=list(self.events), history=self.history[-450:])

    def start(self):
        sim = self.sim
        if sim.getSimulationState() != sim.simulation_stopped:
            sim.stopSimulation()
            while sim.getSimulationState() != sim.simulation_stopped:
                time.sleep(0.05)
        self.client.setStepping(True)
        sim.startSimulation()
        self.restock()
        self.tasks = [self.infeed(), self.carton_line(), self.delta_task(), self.ur_task(), self.irb_task(),
                      self.amr_task(), self.shipping_task(), self.baxter_task()]
        self.log('line started (%s inspection)' % ('CNN' if self.set['mode'] == 'cnn' else 'ground-truth'))

    def step(self):
        """Advance every station by one simulation step."""
        for task in self.tasks:
            next(task)
        if self.frame:
            hs = list(self.frame)
            self.sim.callScriptFunction('setPoses', self.bridge, hs, [v for h in hs for v in self.frame[h]])
            self.frame.clear()
        self.client.step()
        self.t += self.dt
        self.after_step()
        if not self.history or self.t - self.history[-1][0] >= 2.0:
            st = self.stats
            self.history.append((round(self.t, 1), st['spawned'], st['packed'], st['rejects'], st['cartons']))

    def stop(self):
        sim = self.sim
        self.report()
        try:
            fails = sim.callScriptFunction('getIkFailures', self.bridge)
            if fails:
                print('IK failures (steps):', fails)
        except Exception:
            pass
        sim.stopSimulation()
        while sim.getSimulationState() != sim.simulation_stopped:
            time.sleep(0.05)
        left = [h for h in self.spawned if sim.isHandle(h)]
        if left:
            sim.removeObjects(left)
        self.spawned.clear()

    def run(self, duration=None, report_every=20.0, on_step=None):
        self.start()
        next_report = report_every
        wall0 = time.time()
        try:
            while duration is None or self.t < duration:
                self.step()
                if on_step:
                    on_step(self)
                if self.t >= next_report:
                    next_report += report_every
                    self.report()
                if self.realtime:
                    lag = self.t - (time.time() - wall0)
                    if lag > 0:
                        time.sleep(lag)
        except KeyboardInterrupt:
            print('interrupted')
        finally:
            print('wall time %.1fs for %.1fs simulated' % (time.time() - wall0, self.t))
            self.stop()


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument('--time', type=float, default=None, help='simulated seconds to run (default: until Ctrl+C)')
    ap.add_argument('--oracle', action='store_true', help='use ground-truth labels instead of the CNN')
    ap.add_argument('--realtime', action='store_true', help='do not run faster than real time')
    ap.add_argument('--seed', type=int, default=1)
    a = ap.parse_args()
    Cell(oracle=a.oracle, seed=a.seed, realtime=a.realtime).run(a.time)


if __name__ == '__main__':
    main()
