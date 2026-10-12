"""Collect a labelled image set from the two inspection cameras and train the QC network.

    python -m factory.train_qc            # collect + train
    python -m factory.train_qc --reuse    # retrain on the saved dataset

The simulation must be stopped.  Product look-alikes ("variants") are split by identity:
0..39 train, 40..47 validation (picks the best epoch), 48..59 test.  The reported test accuracy
is therefore measured on defects the network has never seen in any form.
"""
import argparse
import json
import math
import os
import random

import numpy as np

from . import layout as L
from .vision import MODEL_PATH, build_net, grab

HERE = os.path.dirname(os.path.abspath(__file__))
DATA = os.path.join(HERE, 'data', 'qc_dataset.npz')
TRAIN_VARIANTS, VAL_VARIANTS = 40, 48      # train: 0..39, validation: 40..47, test: 48..59


def collect(per_variant=16, seed=0):
    from coppeliasim_zmqremoteapi_client import RemoteAPIClient
    sim = RemoteAPIClient().require('sim')
    assert sim.getSimulationState() == sim.simulation_stopped, 'stop the simulation first'
    rnd = random.Random(seed)
    cams = [(sim.getObject('/QC_cam'), L.QC_X), (sim.getObject('/Sort_cam'), L.A_END_X)]
    stripes = sim.getObject('/BeltA')
    z = L.BELT_Z + L.PROD[2] / 2
    X, y, var, cam_id = [], [], [], []
    for ci, cls in enumerate(L.CLASSES):
        for v in range(L.N_VARIANTS):
            h = sim.getObject('/tpl_%s_%02d' % (cls, v))
            home = sim.getObjectPose(h, -1)
            for k in range(per_variant):
                cam, cx = cams[k % 2]
                yaw = rnd.uniform(-math.pi, math.pi)
                # camera 1 is triggered on the fly (position jitter), camera 2 sees the bar at the stop
                dx = rnd.uniform(-0.012, 0.012) if k % 2 == 0 else rnd.uniform(-0.002, 0.002)
                pos = [cx + dx, L.A_Y + rnd.uniform(-0.07, 0.07), z]
                sim.setObjectPose(h, pos + [0, 0, math.sin(yaw / 2), math.cos(yaw / 2)], -1)
                sim.setObjectPosition(stripes, [rnd.uniform(0, L.BELT_STRIPE_PITCH), 0, 0], -1)
                X.append(grab(sim, cam)); y.append(ci); var.append(v); cam_id.append(k % 2)
            sim.setObjectPose(h, home, -1)
        print('collected', cls, flush=True)
    sim.setObjectPosition(stripes, [0, 0, 0], -1)
    os.makedirs(os.path.dirname(DATA), exist_ok=True)
    np.savez_compressed(DATA, X=np.array(X), y=np.array(y), var=np.array(var), cam=np.array(cam_id))
    print('saved %d images -> %s' % (len(X), DATA))


def contact_sheet(X, y, path, per_class=10):
    import cv2
    rows = []
    for ci in range(len(L.CLASSES)):
        idx = np.where(y == ci)[0]
        idx = idx[np.linspace(0, len(idx) - 1, per_class).astype(int)]
        row = np.concatenate([np.pad(X[i], ((2, 2), (2, 2), (0, 0)), constant_values=255) for i in idx], 1)
        rows.append(row)
    sheet = cv2.resize(np.concatenate(rows, 0), None, fx=2, fy=2, interpolation=cv2.INTER_NEAREST)
    cv2.imwrite(path, sheet[:, :, ::-1])


def train(epochs=50, seed=0):
    import copy
    import torch
    import torch.nn.functional as F
    torch.manual_seed(seed)
    d = np.load(DATA)
    X, y, var = d['X'], d['y'], d['var']
    tr, va, te = var < TRAIN_VARIANTS, (var >= TRAIN_VARIANTS) & (var < VAL_VARIANTS), var >= VAL_VARIANTS
    dev = 'cuda' if torch.cuda.is_available() else 'cpu'

    def tens(m):
        return (torch.from_numpy(X[m]).permute(0, 3, 1, 2).float().div(255).to(dev), torch.from_numpy(y[m]).long().to(dev))

    def accuracy(net, xs, ys):
        with torch.no_grad():
            pred = torch.cat([net(xs[i:i + 256]).argmax(1) for i in range(0, len(xs), 256)])
        return (pred == ys).float().mean().item(), pred

    Xt, yt = tens(tr)
    Xv, yv = tens(va)
    Xe, ye = tens(te)
    net = build_net().to(dev)
    opt = torch.optim.AdamW(net.parameters(), 1e-3, weight_decay=1e-3)
    sched = torch.optim.lr_scheduler.OneCycleLR(opt, 2e-3, total_steps=epochs * math.ceil(len(Xt) / 64))
    best = (-1.0, None, 0)
    for ep in range(epochs):
        net.train()
        perm = torch.randperm(len(Xt), device=dev)
        for i in range(0, len(Xt), 64):
            b = perm[i:i + 64]
            xb = Xt[b]
            fh = torch.rand(len(b), 1, 1, 1, device=dev) < 0.5          # per-sample flips (bars have no "up")
            xb = torch.where(fh, xb.flip(3), xb)
            fv = torch.rand(len(b), 1, 1, 1, device=dev) < 0.5
            xb = torch.where(fv, xb.flip(2), xb)
            xb = (xb * (0.85 + 0.3 * torch.rand(len(b), 1, 1, 1, device=dev)) + 0.02 * torch.randn_like(xb)).clamp(0, 1)
            loss = F.cross_entropy(net(xb), yt[b])
            opt.zero_grad(); loss.backward(); opt.step(); sched.step()
        net.eval()
        acc_v, _ = accuracy(net, Xv, yv)
        if acc_v >= best[0]:
            best = (acc_v, copy.deepcopy(net.state_dict()), ep + 1)
        if ep % 5 == 4 or ep == epochs - 1:
            print('epoch %2d  loss %.3f  val acc %.3f' % (ep + 1, loss.item(), acc_v), flush=True)
    net.load_state_dict(best[1])
    net.eval()
    acc_t, _ = accuracy(net, Xt, yt)
    acc_e, pe = accuracy(net, Xe, ye)
    n = len(L.CLASSES)
    cm = np.zeros((n, n), int)
    for t, p in zip(ye.cpu().numpy(), pe.cpu().numpy()):
        cm[t, p] += 1
    os.makedirs(os.path.dirname(MODEL_PATH), exist_ok=True)
    torch.save(net.cpu().state_dict(), MODEL_PATH)
    metrics = dict(train_images=int(tr.sum()), val_images=int(va.sum()), test_images=int(te.sum()),
                   train_accuracy=acc_t, val_accuracy=best[0], test_accuracy=acc_e, best_epoch=best[2],
                   classes=L.CLASSES, test_confusion_rows_true_cols_pred=cm.tolist(), device=dev, epochs=epochs)
    with open(os.path.join(os.path.dirname(MODEL_PATH), 'qc_metrics.json'), 'w') as f:
        json.dump(metrics, f, indent=2)
    contact_sheet(X, y, os.path.join(os.path.dirname(MODEL_PATH), 'qc_samples.png'))
    print('best epoch %d: train %.3f  val %.3f  TEST %.3f' % (best[2], acc_t, best[0], acc_e))
    print('test confusion matrix (rows = truth, cols = prediction, order %s):' % L.CLASSES)
    print(cm)
    print('model ->', MODEL_PATH)


if __name__ == '__main__':
    ap = argparse.ArgumentParser()
    ap.add_argument('--reuse', action='store_true')
    ap.add_argument('--per-variant', type=int, default=16)
    ap.add_argument('--epochs', type=int, default=50)
    a = ap.parse_args()
    if not (a.reuse and os.path.exists(DATA)):
        collect(a.per_variant)
    train(a.epochs)
