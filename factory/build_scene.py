"""Builds the complete factory scene in the running CoppeliaSim instance and saves it.

    python -m factory.build_scene
"""
import math
import os
import random

from coppeliasim_zmqremoteapi_client import RemoteAPIClient

from . import layout as L

HERE = os.path.dirname(os.path.abspath(__file__))
SCENE = os.path.join(os.path.dirname(HERE), 'factory.ttt').replace('\\', '/')
KEEP = {'DefaultCamera', 'XYZCameraProxy', 'DefaultLights'}

STEEL = [0.55, 0.57, 0.60]
DARK = [0.18, 0.18, 0.20]
YELLOW = [0.95, 0.75, 0.10]
BLUE = [0.15, 0.35, 0.75]
WOOD = [0.72, 0.55, 0.33]
CARD = [0.70, 0.52, 0.32]
SOAP = [0.93, 0.88, 0.72]
D2R = math.pi / 180

# Seed joint configurations (tool pointing straight down, elbow up) used as IK starting points.
UR_SEED = [90, 30, 90, -30, -90, 0]
IRB_SEED = [0, 0, 0, 0, 90, 0]
BAXTER_SEED = {'L': [-45, -40, 0, 80, 0, 50, 0], 'R': [45, -40, 0, 80, 0, 50, 0]}


class Builder:
    def __init__(self):
        self.client = RemoteAPIClient()
        self.sim = self.client.require('sim')
        self.models = self.sim.getStringParam(self.sim.stringparam_application_path) + '/models/'

    # ---------------------------------------------------------------- helpers
    def box(self, name, size, pos, color, parent=-1, yaw=0.0, kind='cuboid', rot=None):
        sim = self.sim
        prim = {'cuboid': sim.primitiveshape_cuboid, 'cyl': sim.primitiveshape_cylinder,
                'sphere': sim.primitiveshape_spheroid}[kind]
        h = sim.createPrimitiveShape(prim, list(size), 0)
        sim.setObjectAlias(h, name)
        sim.setShapeColor(h, None, sim.colorcomponent_ambient_diffuse, color)
        sim.setObjectInt32Param(h, sim.shapeintparam_static, 1)
        sim.setObjectInt32Param(h, sim.shapeintparam_respondable, 0)
        sim.setObjectPosition(h, list(pos), -1)
        if yaw or rot:
            sim.setObjectOrientation(h, rot or [0, 0, yaw], -1)
        if parent != -1:
            sim.setObjectParent(h, parent, True)
        return h

    def dummy(self, name, pos, parent=-1, size=0.02):
        sim = self.sim
        h = sim.createDummy(size)
        sim.setObjectAlias(h, name)
        sim.setObjectPosition(h, list(pos), -1)
        sim.setObjectInt32Param(h, sim.objintparam_visibility_layer, 1024)
        if parent != -1:
            sim.setObjectParent(h, parent, True)
        return h

    def group(self, name, parts, frame_pos):
        sim = self.sim
        h = sim.groupShapes(parts, False)
        sim.relocateShapeFrame(h, list(frame_pos) + [0, 0, 0, 1])
        sim.setObjectAlias(h, name)
        sim.setObjectInt32Param(h, sim.shapeintparam_static, 1)
        sim.setObjectInt32Param(h, sim.shapeintparam_respondable, 0)
        return h

    def load(self, rel, name, pos=None, yaw=None):
        sim = self.sim
        h = sim.loadModel(self.models + rel)
        sim.setObjectAlias(h, name)
        if pos is not None:
            sim.setObjectPosition(h, list(pos), -1)
        if yaw is not None:
            sim.setObjectOrientation(h, [0, 0, yaw], -1)
        return h

    def make_kinematic(self, model):
        """Strip the demo scripts and turn a stock robot model into a purely kinematic one."""
        sim = self.sim
        scripts = sim.getObjectsInTree(model, sim.sceneobject_script)
        if scripts:
            sim.removeObjects(scripts)
        sim.setModelProperty(model, sim.getModelProperty(model) | sim.modelproperty_not_dynamic
                             | sim.modelproperty_not_respondable)
        joints = sim.getObjectsInTree(model, sim.sceneobject_joint)
        for j in joints:
            if sim.getJointMode(j)[0] == sim.jointmode_dynamic:
                sim.setJointMode(j, sim.jointmode_kinematic, 0)
        return joints

    def look_at(self, h, eye, target):
        ex, ey, ez = eye
        z = [target[0] - ex, target[1] - ey, target[2] - ez]
        n = math.sqrt(sum(c * c for c in z)); z = [c / n for c in z]
        x = [-z[1], z[0], 0.0]                      # sensor x points to image-left, y is image-up
        n = math.sqrt(sum(c * c for c in x)); x = [c / n for c in x]
        y = [z[1] * x[2] - z[2] * x[1], z[2] * x[0] - z[0] * x[2], z[0] * x[1] - z[1] * x[0]]  # z cross x
        self.sim.setObjectMatrix(h, [x[0], y[0], z[0], ex, x[1], y[1], z[1], ey, x[2], y[2], z[2], ez], -1)

    def vision(self, name, pos, res, fov, far=2.0, bg=(0.1, 0.1, 0.1)):
        sim = self.sim
        h = sim.createVisionSensor(1 | 2, [res[0], res[1], 0, 0],
                                   [0.02, far, fov, 0.03, 0, 0, bg[0], bg[1], bg[2], 0, 0])
        sim.setObjectAlias(h, name)
        sim.setObjectPosition(h, list(pos), -1)
        return h

    def tool_frames(self, name, flange, tcp_pos, yaw, ik_base):
        """World-aligned (plus yaw) tip + target dummies; IK keeps the tool pointing down."""
        sim = self.sim
        tip = self.dummy(name + '_tip', tcp_pos)
        sim.setObjectOrientation(tip, [0, 0, yaw], -1)
        sim.setObjectParent(tip, flange, True)
        tgt = self.dummy(name + '_target', tcp_pos)
        sim.setObjectOrientation(tgt, [0, 0, yaw], -1)
        self.dummy(name + '_ikBase', sim.getObjectPosition(ik_base, -1), parent=ik_base)
        return tip, tgt

    # ------------------------------------------------------------------ scene
    def clear(self):
        sim = self.sim
        if sim.getSimulationState() != sim.simulation_stopped:
            sim.stopSimulation()
            while sim.getSimulationState() != sim.simulation_stopped:
                pass
        for h in sim.getObjectsInTree(sim.handle_scene, sim.handle_all, 2):
            if sim.getObjectAlias(h) not in KEEP:
                sim.removeObjects(sim.getObjectsInTree(h))
        for i, x in enumerate((-6.0, -1.0, 4.0)):
            for k, y in enumerate((-3.5, 1.5)):
                f = self.load('infrastructure/floors/5mX5m concrete floor.ttm', 'Floor%d%d' % (i, k))
                sim.setObjectPosition(f, [x, y, sim.getObjectPosition(f, -1)[2]], -1)
        hall = self.dummy('Hall', [0, 0, 0])
        wall = [0.80, 0.82, 0.84]
        self.box('wall', [15.0, 0.1, 6.0], [-1.0, 4.0, 3.0], wall, hall)
        self.box('wall', [0.1, 10.0, 4.0], [-8.5, -1.0, 2.0], wall, hall)
        self.box('wall', [0.1, 10.0, 4.0], [6.5, -1.0, 2.0], wall, hall)
        self.box('band', [15.0, 0.02, 0.25], [-1.0, 3.94, 1.2], BLUE, hall)
        sim.setBoolParam(sim.boolparam_dynamics_handling_enabled, False)
        sim.setFloatParam(sim.floatparam_simulation_time_step, 0.05)

    def conveyor(self, name, x0, x1, y, z, width, kind=1):
        """Light-weight conveyor made of primitives.  kind 1: belt whose stripes (children of the
        'Belt<name>' dummy) are shifted by the controller to show motion; kind 2: roller track."""
        x0, x1 = min(x0, x1), max(x0, x1)
        length, xc = x1 - x0, (x0 + x1) / 2
        h = self.dummy('Conveyor' + name, [xc, y, z])
        half = math.pi / 2
        if kind == 1:
            r = 0.05
            self.box('belt', [length, width, 0.012], [xc, y, z - 0.006], [0.13, 0.13, 0.14], h)
            self.box('bed', [length - 0.1, width - 0.02, 2 * r - 0.03], [xc, y, z - r], [0.25, 0.25, 0.27], h)
            for x in (x0, x1):
                self.box('drum', [2 * r, 2 * r, width], [x, y, z - r], [0.13, 0.13, 0.14], h, kind='cyl', rot=[half, 0, 0])
            stripes = self.dummy('Belt' + name, [0, 0, 0], h)
            n = int((length - 0.02) / L.BELT_STRIPE_PITCH)
            marks = [self.box('s', [0.012, width - 0.02, 0.001], [x0 + 0.01 + k * L.BELT_STRIPE_PITCH, y, z + 0.0005], [0.30, 0.30, 0.32])
                     for k in range(n)]
            self.sim.setObjectParent(self.group('stripes', marks, [0, 0, 0]), stripes, True)
        else:
            r = 0.03
            n = int(length / 0.09)
            rollers = [self.box('r', [2 * r, 2 * r, width], [x0 + r + k * (length - 2 * r) / (n - 1), y, z - r], [0.70, 0.71, 0.74],
                                kind='cyl', rot=[half, 0, 0]) for k in range(n)]
            self.sim.setObjectParent(self.group('rollers', rollers, [xc, y, z]), h, True)
        xs = (x0 + 0.25, xc, x1 - 0.25) if length > 2 else (x0 + 0.2, x1 - 0.2)
        for x in xs:
            for sd in (-1, 1):
                self.box('leg', [0.04, 0.04, z - 0.10], [x, y + sd * (width / 2 + 0.03), (z - 0.10) / 2], STEEL, h)
        for sd in (-1, 1):
            self.box('rail', [length + 0.06, 0.02, 0.09], [xc, y + sd * (width / 2 + 0.03), z - 0.03], STEEL, h)
        return h

    def pallet(self, o):
        return self.group('tpl_pallet', [
            self.box('p', [L.PALLET[0], L.PALLET[1], 0.015], [o[0], o[1], o[2] + 0.0375], WOOD)] + [
            self.box('p', [L.PALLET[0], 0.04, 0.03], [o[0], o[1] + s * 0.18, o[2] + 0.015], [0.62, 0.46, 0.27])
            for s in (-1, 0, 1)], o)

    def templates(self):
        """Hidden masters (below the floor) that the controller copies at run time."""
        sim = self.sim
        root = self.dummy('Templates', [0, 0, -3])
        rnd = random.Random(7)
        px, py, pz = L.PROD
        top = pz / 2 + 0.0005
        for ci, cls in enumerate(L.CLASSES):
            for v in range(L.N_VARIANTS):
                o = [0.2 * v - 6, 0.3 * ci, -3]
                parts = [self.box('b', [px, py, pz], o, SOAP),
                         self.box('logo', [0.030, 0.016, 0.001], [o[0], o[1], o[2] + top], [0.99, 0.96, 0.86])]
                if cls == 'rework':            # small dark blemishes
                    for _ in range(rnd.randint(1, 3)):
                        d = rnd.uniform(0.005, 0.009)
                        parts.append(self.box('spot', [d, d, 0.0012], [o[0] + rnd.uniform(-0.028, 0.028),
                                     o[1] + rnd.uniform(-0.018, 0.018), o[2] + top], [0.25, 0.14, 0.08], kind='cyl'))
                elif cls == 'scrap':           # crack across the bar and/or a burnt patch
                    if rnd.random() < 0.75:
                        parts.append(self.box('crack', [rnd.uniform(0.045, 0.075), 0.004, 0.0012],
                                     [o[0] + rnd.uniform(-0.006, 0.006), o[1] + rnd.uniform(-0.012, 0.012), o[2] + top],
                                     [0.12, 0.08, 0.06], yaw=rnd.uniform(-1.2, 1.2)))
                    if len(parts) == 2 or rnd.random() < 0.4:
                        parts.append(self.box('burn', [rnd.uniform(0.02, 0.03), rnd.uniform(0.02, 0.03), 0.0012],
                                     [o[0] + rnd.uniform(-0.02, 0.02), o[1] + rnd.uniform(-0.01, 0.01), o[2] + top],
                                     [0.30, 0.18, 0.10], yaw=rnd.uniform(0, 3)))
                sim.setObjectParent(self.group('tpl_%s_%02d' % (cls, v), parts, o), root, True)
        # carton (open box), frame at outer bottom centre
        cx, cy, cz = L.CARTON
        o = [0, 1.5, -3]
        t = 0.005
        parts = [self.box('p', [cx, cy, L.CARTON_FLOOR], [o[0], o[1], o[2] + L.CARTON_FLOOR / 2], CARD)]
        for s in (-1, 1):
            parts.append(self.box('p', [cx, t, cz], [o[0], o[1] + s * (cy - t) / 2, o[2] + cz / 2], CARD))
            parts.append(self.box('p', [t, cy - 2 * t, cz], [o[0] + s * (cx - t) / 2, o[1], o[2] + cz / 2], CARD))
        sim.setObjectParent(self.group('tpl_carton', parts, o), root, True)
        o = [0.5, 1.5, -3]
        lid = [self.box('p', [cx, cy, L.LID_T], [o[0], o[1], o[2] + L.LID_T / 2], [0.76, 0.58, 0.36]),
               self.box('p', [cx, 0.03, 0.001], [o[0], o[1], o[2] + L.LID_T], [0.85, 0.80, 0.65]),
               self.box('p', [0.05, 0.03, 0.001], [o[0] + 0.04, o[1] + 0.04, o[2] + L.LID_T], [0.95, 0.95, 0.95])]
        sim.setObjectParent(self.group('tpl_lid', lid, o), root, True)
        sim.setObjectParent(self.pallet([1.0, 1.5, -3]), root, True)

    def delta_cell(self):
        sim = self.sim
        tcp = L.DELTA_TCP_HOME
        tool_len = 0.07
        m = self.load('robots/non-mobile/ABB IRB 360.ttm', 'irb360')
        sim.removeObjects(sim.getObjectsInTree(m, sim.sceneobject_script))
        ik_tip = sim.getObject('/irb360/ikTip')
        dz = sim.getObjectPosition(m, -1)[2] - sim.getObjectPosition(ik_tip, -1)[2]
        base_z = tcp[2] + tool_len + dz
        sim.setObjectPosition(m, [L.DELTA_XY[0], L.DELTA_XY[1], base_z], -1)
        conn = sim.getObject('/irb360/connection')
        p = sim.getObjectPosition(ik_tip, -1)
        cup = self.box('DeltaGripper', [0.025, 0.025, tool_len - 0.012], [p[0], p[1], p[2] - (tool_len - 0.012) / 2], DARK, conn, kind='cyl')
        self.box('DeltaCup', [0.04, 0.04, 0.012], [p[0], p[1], p[2] - tool_len + 0.006], BLUE, cup, kind='cyl')
        self.dummy('Delta_tip', [p[0], p[1], p[2] - tool_len], cup)
        self.dummy('Delta_target', [p[0], p[1], p[2] - tool_len])
        g = self.dummy('DeltaFrame', [0, 0, 0])
        top = base_z + 0.25
        for sx in (-1, 1):
            for sy in (-1, 1):
                self.box('post', [0.06, 0.06, top], [sx * 0.72, sy * 0.72, top / 2], YELLOW, g)
            self.box('beam', [0.06, 1.50, 0.06], [sx * 0.72, 0, top], YELLOW, g)
        self.box('beam', [1.50, 0.06, 0.06], [0, 0.72, top], YELLOW, g)
        self.box('beam', [1.50, 0.06, 0.06], [0, -0.72, top], YELLOW, g)
        self.box('mount', [1.50, 0.40, 0.04], [0, 0, top - 0.02], STEEL, g)
        return m

    def qc_camera(self, name, x, y):
        z = L.BELT_Z + L.PROD[2] + L.CAM_H
        cam = self.vision(name, [x, y, z], [L.CAM_RES, L.CAM_RES], L.CAM_FOV, far=1.0)
        self.sim.setObjectOrientation(cam, [math.pi, 0, 0], -1)
        st = self.dummy(name + '_stand', [x, y, 0])
        yo = y + L.BELT_W / 2 + 0.12
        self.box('post', [0.04, 0.04, z + 0.10], [x, yo, (z + 0.10) / 2], STEEL, st)
        self.box('arm', [0.04, abs(yo - y) + 0.04, 0.04], [x, (yo + y) / 2, z + 0.10], STEEL, st)
        self.box('body', [0.07, 0.07, 0.07], [x, y, z + 0.045], DARK, st)
        self.sim.setObjectParent(cam, st, True)
        return cam

    def infeed_and_packing(self):
        self.conveyor('A', L.A_X0, L.A_X1, L.A_Y, L.BELT_Z, L.BELT_W)
        self.conveyor('B', L.B_X1, L.B_X0, L.B_Y, L.BELT_Z, L.BELT_W)
        self.dummy('CarrierA', [0, 0, 0])
        f = self.dummy('Feeder', [L.A_X0, L.A_Y, 0])
        self.box('hopper', [0.45, 0.40, 0.45], [L.A_SPAWN_X - 0.12, L.A_Y, L.BELT_Z + 0.33], BLUE, f)
        self.box('chute', [0.20, 0.34, 0.10], [L.A_SPAWN_X + 0.12, L.A_Y, L.BELT_Z + 0.16], DARK, f)
        self.box('leg', [0.35, 0.50, L.BELT_Z + 0.10], [L.A_X0 - 0.20, L.A_Y, (L.BELT_Z + 0.10) / 2], STEEL, f)
        f = self.dummy('CartonErector', [L.B_X0, L.B_Y, 0])
        self.box('body', [0.45, 0.42, 0.50], [L.B_SPAWN_X + 0.20, L.B_Y, L.BELT_Z + 0.36], [0.20, 0.55, 0.35], f)
        self.box('leg', [0.35, 0.50, L.BELT_Z + 0.11], [L.B_X0 + 0.25, L.B_Y, (L.BELT_Z + 0.11) / 2], STEEL, f)
        f = self.dummy('CaseSealer', [L.B_SEAL_X, L.B_Y, 0])
        for s in (-1, 1):
            self.box('side', [0.40, 0.03, 0.26], [L.B_SEAL_X, L.B_Y + s * (L.BELT_W / 2 + 0.075), L.BELT_Z + 0.10], [0.75, 0.2, 0.15], f)
        self.box('top', [0.40, L.BELT_W + 0.18, 0.10], [L.B_SEAL_X, L.B_Y, L.BELT_Z + 0.24], [0.75, 0.2, 0.15], f)
        self.box('StopB', [0.02, L.BELT_W, 0.05], [L.B_END_X - L.CARTON[0] / 2 - 0.012, L.B_Y, L.BELT_Z + 0.03], YELLOW)
        self.box('StopA', [0.02, L.BELT_W, 0.04], [L.A_END_X + L.PROD[0] / 2 + 0.02, L.A_Y, L.BELT_Z + 0.025], YELLOW)
        self.qc_camera('QC_cam', L.QC_X, L.A_Y)
        self.qc_camera('Sort_cam', L.A_END_X, L.A_Y)
        self.delta_cell()

    def suction_tool(self, name, flange, length=0.10, r=0.03):
        """Vacuum tool hanging straight down from `flange`; returns the TCP position."""
        p = self.sim.getObjectPosition(flange, -1)
        body = self.box(name + '_tool', [r, r, length - 0.015], [p[0], p[1], p[2] - (length - 0.015) / 2], DARK, flange, kind='cyl')
        self.box(name + '_cup', [r * 2, r * 2, 0.015], [p[0], p[1], p[2] - length + 0.0075], BLUE, body, kind='cyl')
        return [p[0], p[1], p[2] - length]

    def ur10_cell(self):
        sim = self.sim
        bx, by, bz = L.UR_BASE
        self.box('UR10_pedestal', [0.25, 0.25, bz], [bx, by, bz / 2], STEEL)
        m = self.load('robots/non-mobile/UR10.ttm', 'UR10')
        p = sim.getObjectPosition(m, -1)
        sim.setObjectPosition(m, [bx, by, bz + p[2]], -1)
        self.make_kinematic(m)
        for i, a in enumerate(UR_SEED):
            sim.setJointPosition(sim.getObject('/UR10/joint', {'index': i}), a * D2R)
        flange = sim.getObject('/UR10/connection')
        tcp = self.suction_tool('UR10', flange)
        self.tool_frames('UR10', flange, tcp, 0.0, m)
        return m

    def irb_cell(self):
        sim = self.sim
        m = self.load('robots/non-mobile/ABB IRB 4600-40-255.ttm', 'IRB4600')
        p = sim.getObjectPosition(m, -1)
        sim.setObjectPosition(m, [L.IRB_BASE[0], L.IRB_BASE[1], p[2]], -1)
        self.make_kinematic(m)
        sim.removeObjects([sim.getObject('/IRB4600/IkTarget'), sim.getObject('/IRB4600/IkTip')])
        for i, a in enumerate(IRB_SEED):
            sim.setJointPosition(sim.getObject('/IRB4600/joint', {'index': i}), a * D2R)
        flange = sim.getObject('/IRB4600/connection')
        f = sim.getObjectPosition(flange, -1)
        # C-shaped fork: top plate under the flange, mast behind the load, two tines under the pallet
        H = 0.34
        tcp = [f[0], f[1], f[2] - H]
        top = self.box('Fork', [0.27, 0.12, 0.025], [f[0] - 0.095, f[1], f[2] - 0.0125], DARK, flange)
        self.box('ForkMast', [0.025, 0.26, H + 0.012], [f[0] - 0.2175, f[1], f[2] - (H + 0.012) / 2], YELLOW, top)
        for s in (-1, 1):
            self.box('ForkTine', [0.38, 0.05, 0.012], [f[0] - 0.04, f[1] + s * 0.09, tcp[2] - 0.006], YELLOW, top)
        self.tool_frames('IRB4600', flange, tcp, 0.0, m)
        sx, sy, sz = L.STATION
        st = self.dummy('PalletStation', [sx, sy, 0])
        self.box('top', [0.34, 0.44, 0.03], [sx, sy, sz - 0.015], BLUE, st)
        self.box('col', [0.12, 0.12, sz - 0.03], [sx, sy, (sz - 0.03) / 2], STEEL, st)
        self.box('foot', [0.40, 0.40, 0.02], [sx, sy, 0.01], STEEL, st)
        sx, sy, sz = L.STACK
        st = self.dummy('PalletStack', [sx, sy, 0])
        self.box('top', [0.44, 0.34, 0.03], [sx, sy, sz - 0.015], BLUE, st)
        self.box('col', [0.20, 0.20, sz - 0.03], [sx, sy, (sz - 0.03) / 2], STEEL, st)
        return m

    def amr_and_shipping(self):
        sim = self.sim
        m = self.load('robots/mobile/KUKA Omnirob.ttm', 'AMR')
        self.make_kinematic(m)
        z = sim.getObjectPosition(m, -1)[2]
        sim.setObjectPosition(m, [L.DOCK[0], L.DOCK[1], z], -1)
        sim.setObjectOrientation(m, [0, 0, math.pi / 2], -1)
        self.conveyor('Ship', L.SHIP_X0, L.SHIP_X1, L.DOCK[1], L.AMR_TOP, 0.50, kind=2)
        self.box('Lane', [abs(L.SHIP_X0 - L.DOCK[0]) + 1.6, 0.9, 0.002], [(L.SHIP_X0 + L.DOCK[0]) / 2 - 0.3, L.DOCK[1], 0.004], [0.85, 0.80, 0.25])

    def baxter_cell(self):
        sim = self.sim
        m = self.load('robots/non-mobile/Baxter.ttm', 'Baxter')
        z = sim.getObjectPosition(m, -1)[2]
        sim.setObjectPosition(m, [L.BAXTER[0], L.BAXTER[1], z], -1)
        sim.setObjectOrientation(m, [0, 0, math.pi], -1)
        self.make_kinematic(m)
        upper = sim.getObject('/Baxter/upperBody')
        for side in 'LR':
            arm = 'leftArm' if side == 'L' else 'rightArm'
            conn = sim.getObject('/Baxter/%s_connector' % arm)
            joints, o = [], conn
            while o != upper:
                if sim.getObjectType(o) == sim.sceneobject_joint:
                    joints.append(o)
                o = sim.getObjectParent(o)
            for j, a in zip(reversed(joints), BAXTER_SEED[side]):
                sim.setJointPosition(j, a * D2R)
            tcp = self.suction_tool('Baxter' + side, conn, length=0.08, r=0.022)
            self.tool_frames('Baxter' + side, conn, tcp, 0.0, upper)
        for name, (x, y, zt), col in (('BinRework', L.BIN_REWORK, [0.95, 0.65, 0.10]), ('BinScrap', L.BIN_SCRAP, [0.80, 0.15, 0.15])):
            b = self.dummy(name, [x, y, 0])
            w, hgt, t = 0.30, 0.22, 0.01
            self.box('floor', [w, w, t], [x, y, zt - hgt], col, b)
            for s in (-1, 1):
                self.box('w', [w, t, hgt], [x, y + s * w / 2, zt - hgt / 2], col, b)
                self.box('w', [t, w, hgt], [x + s * w / 2, y, zt - hgt / 2], col, b)
            self.box('stand', [0.2, 0.2, zt - hgt], [x, y, (zt - hgt) / 2], STEEL, b)
        return m

    def cameras(self):
        sim = self.sim
        self.look_at(sim.getObject('/DefaultCamera'), [1.5, -7.5, 5.0], [-1.8, -0.6, 0.5])
        for name, (eye, tgt) in L.VIEWS.items():
            v = self.vision('View_' + name, eye, [960, 600], math.radians(50), far=40.0, bg=(0.72, 0.80, 0.88))
            self.look_at(v, eye, tgt)
            sim.setObjectInt32Param(v, sim.objintparam_visibility_layer, 0)

    def bridge(self):
        sim = self.sim
        d = self.dummy('FactoryBridge', [0, 0, 0])
        code = open(os.path.join(HERE, 'bridge.lua'), encoding='utf-8').read()
        s = sim.createScript(sim.scripttype_simulation, code, 0, 'lua')
        sim.setObjectParent(s, d, True)
        sim.setObjectAlias(s, 'script')

    def build(self, stages=None):
        steps = ['templates', 'infeed_and_packing', 'ur10_cell', 'irb_cell', 'amr_and_shipping',
                 'baxter_cell', 'cameras', 'bridge']
        self.clear()
        for s in steps:
            if stages is None or s in stages:
                getattr(self, s)()
                print('built', s, flush=True)


if __name__ == '__main__':
    import sys
    b = Builder()
    b.build(sys.argv[1:] or None)
    if len(sys.argv) == 1:
        b.sim.saveScene(SCENE)
        print('saved', SCENE)
    print('objects:', len(b.sim.getObjectsInTree(b.sim.handle_scene)))
