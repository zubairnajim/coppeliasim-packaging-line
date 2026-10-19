"""Operator dashboard (HMI) for the packaging line.

    python -m factory.dashboard                # serves http://127.0.0.1:8050 and opens it

The dashboard process owns the line: START creates a fresh cell controller and starts the
simulation, HOLD / RESUME freeze and continue it, STOP ends it and tidies the scene, E-STOP
freezes everything until it is reset.  Setpoints and per-station cycle stops are applied live.

Only the main thread talks to CoppeliaSim.  The HTTP threads read an already-serialised
snapshot and push operator commands into a queue.
"""
import argparse
import json
import os
import queue
import threading
import time
import traceback
import webbrowser
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer

import cv2
from coppeliasim_zmqremoteapi_client import RemoteAPIClient

from . import layout as L
from .controller import Cell
from .vision import MODEL_PATH, grab

HERE = os.path.dirname(os.path.abspath(__file__))
PAGE = os.path.join(HERE, 'dashboard.html')

LIMITS = dict(feed_rate=(10, 70), belt_speed=(0.08, 0.20), rework_pct=(0, 30), scrap_pct=(0, 30), layers=(1, 3))
STATIONS = ('feeder', 'delta', 'palletizer', 'pallet_robot', 'amr', 'sorter')
SPEEDS = (0.5, 1.0, 2.0, 0.0)          # 0 = as fast as possible


def layout_json():
    keys = ('BELT_W A_Y A_X0 A_X1 A_SPAWN_X QC_X A_END_X B_Y B_X0 B_X1 B_SPLIT_X B_FILL_X B_SEAL_X B_END_X CARTON PALLET '
            'DELTA_XY UR_BASE STATION IRB_BASE IRB_R STACK DOCK AMR_UNLOAD_X SHIP_X0 SHIP_X1 BAXTER BIN_REWORK '
            'BIN_SCRAP SLOTS').split()
    d = {k: getattr(L, k) for k in keys}
    d['limits'] = LIMITS
    d['views'] = list(L.VIEWS)
    return json.dumps(d).encode()


class Supervisor:
    def __init__(self, seed=1, parts='bars'):
        self.client = RemoteAPIClient()
        self.sim = self.client.require('sim')
        self.views = {n: self.sim.getObject('/View_' + n) for n in L.VIEWS}
        self.seed = seed
        self.parts = parts
        self.state = 'STOPPED'               # STOPPED | RUNNING | HELD | ESTOP | FAULT
        self.fault = ''
        self.cell = None
        self.cmds = queue.Queue()
        self.speed = 1.0
        if parts == 'tiles':
            from .real_parts import MODEL
            cnn = os.path.exists(MODEL)
        else:
            cnn = os.path.exists(MODEL_PATH)
        self.settings = dict(feed_rate=52.0, belt_speed=L.A_SPEED, rework_pct=11.0, scrap_pct=9.0,
                             layers=L.PALLET_LAYERS, mode='cnn' if cnn else 'oracle')
        self.holds = {s: False for s in STATIONS}
        self.cnn_available = cnn
        self.snapshot = b'{}'
        self.view_jpg = {}
        self.view_wanted = {}
        self.view_turn = 0
        self.pace = (0.0, time.time())       # (sim time, wall time) reference for speed control
        self.publish()

    # ----------------------------------------------------------- operator commands
    def repace(self):
        self.pace = (self.cell.t if self.cell else 0.0, time.time())

    def command(self, c):
        cmd = c.get('cmd')
        if cmd == 'start' and self.state in ('STOPPED', 'FAULT'):
            self.cell = Cell(seed=self.seed, parts=self.parts)
            self.cnn_available = self.cell.classifier is not None
            if not self.cnn_available:
                self.settings['mode'] = 'oracle'
            self.cell.set.update(self.settings)
            self.cell.hold.update(self.holds)
            self.cell.start()
            self.state, self.fault = 'RUNNING', ''
            self.repace()
        elif cmd == 'hold' and self.state == 'RUNNING':
            self.state = 'HELD'
            self.cell.log('line held by operator', 'warn')
        elif cmd == 'resume' and self.state == 'HELD':
            self.state = 'RUNNING'
            self.cell.log('line resumed')
            self.repace()
        elif cmd == 'estop' and self.state in ('RUNNING', 'HELD'):
            self.state = 'ESTOP'
            self.cell.log('EMERGENCY STOP pressed - all motion frozen', 'alarm')
        elif cmd == 'reset' and self.state == 'ESTOP':
            self.state = 'HELD'
            self.cell.log('emergency stop reset - line held, press RESUME', 'warn')
        elif cmd == 'stop' and self.state in ('RUNNING', 'HELD', 'ESTOP'):
            self.cell.log('line stopped by operator')
            self.cell.stop()
            self.state = 'STOPPED'
        elif cmd == 'set' and c.get('key') in self.settings:
            k, v = c['key'], c.get('value')
            if k == 'mode':
                if v not in ('cnn', 'oracle') or (v == 'cnn' and not self.cnn_available):
                    return
            else:
                lo, hi = LIMITS[k]
                v = min(hi, max(lo, float(v)))
                if k == 'layers':
                    v = int(round(v))
            if self.settings[k] != v:
                self.settings[k] = v
                if self.cell and self.state != 'STOPPED':
                    self.cell.set[k] = v
                    self.cell.log('setpoint %s -> %s' % (k.replace('_', ' '), v))
        elif cmd == 'hold_station' and c.get('station') in self.holds:
            s, v = c['station'], bool(c.get('value'))
            self.holds[s] = v
            if self.cell and self.state != 'STOPPED':
                self.cell.hold[s] = v
                self.cell.log('%s: cycle stop %s' % (s.replace('_', ' '), 'requested' if v else 'released'),
                              'warn' if v else 'info')
        elif cmd == 'speed' and c.get('value') in SPEEDS:
            self.speed = c['value']
            self.repace()
        elif cmd == 'reset_counters' and self.cell:
            self.cell.stats = self.cell.zero_stats()
            self.cell.history.clear()
            self.cell.log('counters reset')

    # ------------------------------------------------------------------ publishing
    def publish(self):
        d = dict(line=self.state, fault=self.fault, speed=self.speed, settings=self.settings, holds=self.holds,
                 cnn_available=self.cnn_available, cell=self.cell.snapshot() if self.cell else None)
        self.snapshot = json.dumps(d, separators=(',', ':')).encode()

    def render_view(self):
        """Render one of the scene cameras that a browser asked for recently."""
        now = time.time()
        wanted = [n for n in self.views if now - self.view_wanted.get(n, 0) < 3.0]
        if not wanted:
            return
        self.view_turn += 1
        n = wanted[self.view_turn % len(wanted)]
        img = grab(self.sim, self.views[n])
        self.view_jpg[n] = cv2.imencode('.jpg', img[:, :, ::-1], [cv2.IMWRITE_JPEG_QUALITY, 82])[1].tobytes()

    # ------------------------------------------------------------------- main loop
    def loop(self):
        last_pub = 0.0
        while True:
            while True:
                try:
                    c = self.cmds.get_nowait()
                except queue.Empty:
                    break
                try:
                    self.command(c)
                except Exception:
                    self.fail()
            if self.state == 'RUNNING':
                try:
                    self.cell.step()
                except Exception:
                    self.fail()
                if self.speed:
                    ahead = (self.cell.t - self.pace[0]) / self.speed - (time.time() - self.pace[1])
                    if ahead > 0:
                        time.sleep(min(ahead, 0.2))
                    elif ahead < -1.0:
                        self.repace()                # cannot keep up: do not try to catch up later
            else:
                time.sleep(0.03)
            now = time.time()
            if now - last_pub >= 0.2:
                last_pub = now
                try:
                    self.render_view()
                except Exception:
                    pass
                self.publish()

    def fail(self):
        self.fault = traceback.format_exc(limit=3)
        print(self.fault, flush=True)
        try:
            if self.cell:
                self.cell.stop()
        except Exception:
            pass
        self.state = 'FAULT'

    def shutdown(self):
        if self.cell and self.state in ('RUNNING', 'HELD', 'ESTOP'):
            self.cell.stop()


def make_handler(sup, port):
    layout = layout_json()
    allowed = {'127.0.0.1:%d' % port, 'localhost:%d' % port}

    class Handler(BaseHTTPRequestHandler):
        protocol_version = 'HTTP/1.1'

        def log_message(self, *a):
            pass

        def send(self, code, body=b'', ctype='application/json', cache='no-store'):
            self.send_response(code)
            self.send_header('Content-Type', ctype)
            self.send_header('Content-Length', str(len(body)))
            self.send_header('Cache-Control', cache)
            self.end_headers()
            self.wfile.write(body)

        def do_GET(self):
            path = self.path.split('?')[0]
            if path in ('/', '/index.html'):
                with open(PAGE, 'rb') as f:
                    self.send(200, f.read(), 'text/html; charset=utf-8')
            elif path == '/api/state':
                self.send(200, sup.snapshot)
            elif path == '/api/layout':
                self.send(200, layout)
            elif path.startswith('/scan/') and path.endswith('.png'):
                png = sup.cell.scan_png.get(int(path[6:-4])) if sup.cell and path[6:-4].isdigit() else None
                if png:
                    self.send(200, png, 'image/png', 'max-age=3600')
                else:
                    self.send(404)
            elif path.startswith('/view/') and path.endswith('.jpg'):
                name = path[6:-4]
                if name in sup.views:
                    sup.view_wanted[name] = time.time()
                    jpg = sup.view_jpg.get(name)
                    if jpg:
                        self.send(200, jpg, 'image/jpeg')
                    else:
                        self.send(204)
                else:
                    self.send(404)
            else:
                self.send(404)

        def do_POST(self):
            # commands are only accepted from this page itself (blocks cross-site requests to localhost)
            origin = self.headers.get('Origin', '')
            host_ok = self.headers.get('Host', '') in allowed
            origin_ok = not origin or origin.split('://')[-1] in allowed
            n = int(self.headers.get('Content-Length') or 0)
            if self.path != '/api/cmd' or not host_ok or not origin_ok or n > 4096:
                self.send(403)
                return
            try:
                c = json.loads(self.rfile.read(n) or b'{}')
                assert isinstance(c, dict)
            except Exception:
                self.send(400)
                return
            sup.cmds.put(c)
            self.send(200, b'{"ok":true}')

    return Handler


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument('--port', type=int, default=8050)
    ap.add_argument('--seed', type=int, default=1)
    ap.add_argument('--no-browser', action='store_true')
    ap.add_argument('--autostart', action='store_true', help='start the line immediately')
    ap.add_argument('--parts', choices=['bars', 'tiles'], default='bars', help="'tiles' feeds real magnetic-tile photographs")
    a = ap.parse_args()

    sup = Supervisor(seed=a.seed, parts=a.parts)
    server = ThreadingHTTPServer(('127.0.0.1', a.port), make_handler(sup, a.port))
    server.daemon_threads = True
    threading.Thread(target=server.serve_forever, daemon=True).start()
    url = 'http://127.0.0.1:%d' % a.port
    print('dashboard: %s   (Ctrl+C to quit)' % url, flush=True)
    if a.autostart:
        sup.cmds.put({'cmd': 'start'})
    if not a.no_browser:
        webbrowser.open(url)
    try:
        sup.loop()
    except KeyboardInterrupt:
        print('shutting down')
    finally:
        sup.shutdown()
        server.shutdown()


if __name__ == '__main__':
    main()
