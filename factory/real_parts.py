"""Real parts on the conveyor: magnetic-tile photographs become textured parts that the line
inspects, packs and sorts like the simulated soap bars.

    python -I factory/real_data.py          # once: fetch the photographs
    python -m factory.real_parts            # capture training pictures through the scene cameras + train
    python -m factory.controller --parts tiles      # run the line on the held-out tiles
    python -m factory.dashboard --parts tiles       # ... or watch it on the dashboard

The tiles are split once (stratified 60 / 20 / 20 by defect type): the network learns from camera
pictures of the training tiles, picks its epoch on the validation tiles, and the line is then fed
the test tiles, which neither the network nor the epoch choice has seen.

Line grades for the six tile classes (my mapping, not part of the dataset):
    good   <- free (defect-free)
    rework <- blowhole, uneven       surface flaws that could be reground
    scrap  <- break, crack, fray     structural damage
"""
import json
import math
import os
import shutil
import tempfile

import cv2
import numpy as np

from . import layout as L
from .vision import grab

HERE = os.path.dirname(os.path.abspath(__file__))
TILES = os.path.join(HERE, 'data', 'real', 'magnetic_tile.npz')
CAPTURES = os.path.join(HERE, 'data', 'real', 'tiles_camera.npz')
MODEL = os.path.join(HERE, 'models', 'qc_tiles.pt')
METRICS = os.path.join(HERE, 'models', 'qc_tiles_metrics.json')

GRADE = {'free': 'good', 'blowhole': 'rework', 'uneven': 'rework', 'break': 'scrap', 'crack': 'scrap', 'fray': 'scrap'}
SIZE = 0.058                 # tile edge [m]; four fit in a carton like the bars
CAM_FOV = math.radians(10)   # inspection cameras zoom in on the tiles (guide rails centre them on the belt)
Y_JITTER, YAW_JITTER = 0.004, 0.04
SPLIT_SEED = 0


def load_tiles():
    if not os.path.exists(TILES):
        raise SystemExit('missing %s - run:  python -I factory/real_data.py' % TILES)
    d = np.load(TILES)
    classes = [str(c) for c in d['classes']]
    grades = np.array([L.CLASSES.index(GRADE[classes[k]]) for k in d['y']])
    return d['X'], d['y'], grades, classes


def split():
    """Index arrays (train, validation, test), stratified by the detailed tile class."""
    from .real_world import split as stratified
    _, y, _, _ = load_tiles()
    return stratified(y, SPLIT_SEED)


class Cameras:
    """Narrows the two inspection cameras for tiles and puts them back afterwards."""

    def __init__(self, sim):
        self.sim = sim
        self.cams = [sim.getObject('/QC_cam'), sim.getObject('/Sort_cam')]
        self.old = [sim.getObjectFloatParam(c, sim.visionfloatparam_perspective_angle) for c in self.cams]

    def zoom(self):
        for c in self.cams:
            self.sim.setObjectFloatParam(c, self.sim.visionfloatparam_perspective_angle, CAM_FOV)

    def restore(self):
        for c, a in zip(self.cams, self.old):
            self.sim.setObjectFloatParam(c, self.sim.visionfloatparam_perspective_angle, a)


class PartMaker:
    """Creates textured tile parts from greyscale photographs (one texture per part)."""

    def __init__(self, sim):
        self.sim = sim
        self.tmp = tempfile.mkdtemp(prefix='tile_tex_')
        self.n = 0

    def make(self, img, parent=-1, pos=(0, 0, -3)):
        sim = self.sim
        self.n += 1
        path = os.path.join(self.tmp, 't%d.png' % self.n).replace(os.sep, '/')
        cv2.imwrite(path, img)
        plane, tex, _ = sim.createTexture(path, 1, [SIZE, SIZE])
        h = sim.createPrimitiveShape(sim.primitiveshape_cuboid, [SIZE, SIZE, L.PROD[2]], 0)
        sim.setShapeColor(h, None, sim.colorcomponent_ambient_diffuse, [1, 1, 1])
        sim.setShapeTexture(h, tex, sim.texturemap_plane, 1, [SIZE, SIZE])
        sim.removeObjects([plane])
        os.remove(path)
        sim.setObjectInt32Param(h, sim.shapeintparam_static, 1)
        sim.setObjectInt32Param(h, sim.shapeintparam_respondable, 0)
        sim.setObjectPosition(h, list(pos), -1)
        if parent != -1:
            sim.setObjectParent(h, parent, True)
        return h

    def close(self):
        shutil.rmtree(self.tmp, ignore_errors=True)


def collect(poses=(3, 2, 2), seed=0):
    """Photograph every tile part under both inspection cameras, as the running line would."""
    from coppeliasim_zmqremoteapi_client import RemoteAPIClient
    sim = RemoteAPIClient().require('sim')
    assert sim.getSimulationState() == sim.simulation_stopped, 'stop the simulation first'
    rnd = np.random.default_rng(seed)
    X, _, grades, _ = load_tiles()
    parts = split()
    cams = Cameras(sim)
    qc, sort = cams.cams
    stripes = sim.getObject('/BeltA')
    maker = PartMaker(sim)
    z = L.BELT_Z + L.PROD[2] / 2
    out = dict(X=[], y=[], tile=[], part=[])
    cams.zoom()
    try:
        for k, (idx, n) in enumerate(zip(parts, poses)):
            for i in idx:
                h = maker.make(X[i])
                for j in range(n):
                    on_qc = j % 2 == 0
                    cam, cx = (qc, L.QC_X) if on_qc else (sort, L.A_END_X)
                    dx = rnd.uniform(0, L.A_SPEED * 0.05) if on_qc else rnd.uniform(-0.001, 0.001)
                    yaw = rnd.uniform(-YAW_JITTER, YAW_JITTER) + (math.pi if rnd.random() < 0.5 else 0.0)
                    sim.setObjectPose(h, [cx + dx, L.A_Y + rnd.uniform(-Y_JITTER, Y_JITTER), z,
                                          0, 0, math.sin(yaw / 2), math.cos(yaw / 2)], -1)
                    sim.setObjectPosition(stripes, [rnd.uniform(0, L.BELT_STRIPE_PITCH), 0, 0], -1)
                    out['X'].append(grab(sim, cam))
                    out['y'].append(grades[i]); out['tile'].append(i); out['part'].append(k)
                sim.removeObjects([h])
            print('captured %s tiles: %d' % (('train', 'validation', 'test')[k], len(idx)), flush=True)
    finally:
        cams.restore()
        sim.setObjectPosition(stripes, [0, 0, 0], -1)
        maker.close()
    np.savez_compressed(CAPTURES, **{k: np.array(v) for k, v in out.items()})
    print('saved %d camera pictures -> %s' % (len(out['y']), CAPTURES))


def train(epochs=200):
    import torch
    from .train_qc import balanced_accuracy, fit, predict
    d = np.load(CAPTURES)
    X, y, part = d['X'], d['y'], d['part']
    tr, va, te = part == 0, part == 1, part == 2
    net, best_val, best_epoch = fit(X[tr], y[tr], X[va], y[va], len(L.CLASSES), epochs, 0)
    pe = predict(net, X[te])
    n = len(L.CLASSES)
    cm = np.zeros((n, n), int)
    for t, p in zip(y[te], pe):
        cm[t, p] += 1
    good = L.CLASSES.index('good')
    m = dict(train_pictures=int(tr.sum()), val_pictures=int(va.sum()), test_pictures=int(te.sum()),
             best_epoch=best_epoch, val_accuracy=best_val, test_accuracy=float((pe == y[te]).mean()),
             test_balanced_accuracy=balanced_accuracy(y[te], pe, n),
             test_defect_catch_rate=float((pe[y[te] != good] != good).mean()),
             test_false_reject_rate=float((pe[y[te] == good] != good).mean()),
             classes=L.CLASSES, test_confusion_rows_true_cols_pred=cm.tolist(), grade_mapping=GRADE,
             camera_fov_deg=math.degrees(CAM_FOV), epochs=epochs)
    torch.save(net.cpu().state_dict(), MODEL)
    with open(METRICS, 'w') as f:
        json.dump(m, f, indent=1)
    print('camera pictures of unseen tiles: accuracy %.3f  catch %.3f  false reject %.3f'
          % (m['test_accuracy'], m['test_defect_catch_rate'], m['test_false_reject_rate']))
    print(cm)
    print('model ->', MODEL)


if __name__ == '__main__':
    import argparse
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument('step', nargs='?', default='all', choices=['collect', 'train', 'all'])
    ap.add_argument('--epochs', type=int, default=200)
    a = ap.parse_args()
    if a.step in ('collect', 'all'):
        collect()
    if a.step in ('train', 'all'):
        train(a.epochs)
