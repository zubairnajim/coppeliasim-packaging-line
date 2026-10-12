"""Run the line and record what the hidden View_* cameras see.

    python -m factory.record --time 120 --view overview --video out/overview.mp4
    python -m factory.record --time 90 --snap 30 60 90 --out out/      # PNG stills of every view
    python -m factory.record --time 60 --check                         # print IK tracking errors
"""
import argparse
import math
import os

import cv2
import numpy as np

from . import layout as L
from .controller import Cell
from .vision import grab


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--time', type=float, default=120)
    ap.add_argument('--view', default='overview', choices=list(L.VIEWS) + ['all'], help="'all' = 2x2 mosaic")
    ap.add_argument('--video', default=None)
    ap.add_argument('--fps', type=int, default=20)
    ap.add_argument('--snap', type=float, nargs='*', default=[])
    ap.add_argument('--out', default='out')
    ap.add_argument('--check', action='store_true')
    ap.add_argument('--oracle', action='store_true')
    ap.add_argument('--seed', type=int, default=1)
    a = ap.parse_args()

    cell = Cell(oracle=a.oracle, seed=a.seed)
    sim = cell.sim
    views = {n: sim.getObject('/View_' + n) for n in L.VIEWS}
    os.makedirs(a.out, exist_ok=True)
    writer = None
    if a.video:
        os.makedirs(os.path.dirname(os.path.abspath(a.video)), exist_ok=True)
        size = (1280, 800) if a.view == 'all' else (960, 600)
        writer = cv2.VideoWriter(a.video, cv2.VideoWriter_fourcc(*'mp4v'), a.fps, size)
    every = max(1, int(round(1.0 / a.fps / cell.dt)))
    snaps = sorted(a.snap)
    tools = [cell.delta, cell.ur, cell.irb, cell.bax['L'], cell.bax['R']]
    worst = {t.name: 0.0 for t in tools}
    state = {'n': 0}

    def on_step(c):
        state['n'] += 1
        if writer and state['n'] % every == 0:
            if a.view == 'all':
                im = [grab(sim, views[n])[:, :, ::-1] for n in ('overview', 'pack', 'pallet', 'baxter')]
                frame = np.vstack([np.hstack(im[:2]), np.hstack(im[2:])])
                s = c.stats
                cv2.putText(frame, 't=%5.1fs  packed %d  cartons %d  pallets %d  shipped %d  rejects %d' % (
                    c.t, s['packed'], s['cartons'], s['pallets'], s['shipped'], s['rejects']),
                    (20, 40), cv2.FONT_HERSHEY_SIMPLEX, 1.0, (255, 255, 255), 2, cv2.LINE_AA)
                writer.write(cv2.resize(frame, size, interpolation=cv2.INTER_AREA))
            else:
                writer.write(grab(sim, views[a.view])[:, :, ::-1])
        if snaps and c.t >= snaps[0]:
            t = snaps.pop(0)
            for n, h in views.items():
                cv2.imwrite(os.path.join(a.out, '%s_%03d.png' % (n, int(t))), grab(sim, h)[:, :, ::-1])
        if a.check and state['n'] % 2 == 0:
            for t in tools:
                e = math.dist(sim.getObjectPosition(t.tip, -1), t.pos)
                if e > worst[t.name]:
                    worst[t.name] = e
                    if e > 0.03:
                        print('  [%.2fs] %s tracking error %.3f m at %s' % (c.t, t.name, e, [round(v, 2) for v in t.pos]))

    cell.run(a.time, on_step=on_step)
    if writer:
        writer.release()
        print('video:', a.video)
    if a.check:
        print('worst tip-vs-target error [m]:', {k: round(v, 4) for k, v in worst.items()})


if __name__ == '__main__':
    main()
