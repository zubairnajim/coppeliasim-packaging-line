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


def balanced_accuracy(y, pred, n_classes):
    """Mean of the per-class recalls (not fooled by one dominant class)."""
    rec = [float((pred[y == c] == c).mean()) for c in range(n_classes) if (y == c).any()]
    return sum(rec) / len(rec)


def predict(net, X, batch=256):
    """Class predictions for a uint8 image array (N, H, W, 3)."""
    import torch
    dev = 'cuda' if torch.cuda.is_available() else 'cpu'
    net = net.to(dev).eval()                    # stays on that device; callers move it back if they need to
    out = []
    with torch.no_grad():
        for i in range(0, len(X), batch):
            xb = torch.from_numpy(X[i:i + batch]).permute(0, 3, 1, 2).float().div(255).to(dev)
            out.append(net(xb).argmax(1).cpu())
    return torch.cat(out).numpy() if out else np.zeros(0, np.int64)


def fit(Xt, yt, Xv, yv, n_classes=None, epochs=50, seed=0, class_weights=None, select='accuracy', log=True):
    """The training recipe of this project, shared by the simulated QC set and the real-defect
    benchmark: AdamW + one-cycle schedule, random flips, brightness jitter and pixel noise; the
    epoch with the best validation score is kept.  X: uint8 (N, H, W, 3), y: integer labels.
    Returns (network on the CPU in eval mode, best validation score, best epoch)."""
    import copy
    import torch
    import torch.nn.functional as F
    torch.manual_seed(seed)
    n_classes = n_classes or len(L.CLASSES)
    dev = 'cuda' if torch.cuda.is_available() else 'cpu'
    X = torch.from_numpy(Xt).permute(0, 3, 1, 2).float().div(255).to(dev)
    Y = torch.from_numpy(yt).long().to(dev)
    w = None if class_weights is None else torch.tensor(class_weights, dtype=torch.float32, device=dev)
    net = build_net(n_classes).to(dev)
    opt = torch.optim.AdamW(net.parameters(), 1e-3, weight_decay=1e-3)
    sched = torch.optim.lr_scheduler.OneCycleLR(opt, 2e-3, total_steps=epochs * math.ceil(len(X) / 64))
    best = (-1.0, None, 0)
    for ep in range(epochs):
        net.train()
        perm = torch.randperm(len(X), device=dev)
        for i in range(0, len(X), 64):
            b = perm[i:i + 64]
            xb = X[b]
            fh = torch.rand(len(b), 1, 1, 1, device=dev) < 0.5          # per-sample flips (parts have no "up")
            xb = torch.where(fh, xb.flip(3), xb)
            fv = torch.rand(len(b), 1, 1, 1, device=dev) < 0.5
            xb = torch.where(fv, xb.flip(2), xb)
            xb = (xb * (0.85 + 0.3 * torch.rand(len(b), 1, 1, 1, device=dev)) + 0.02 * torch.randn_like(xb)).clamp(0, 1)
            loss = F.cross_entropy(net(xb), Y[b], weight=w)
            opt.zero_grad(); loss.backward(); opt.step(); sched.step()
        pv = predict(net, Xv)
        score = balanced_accuracy(yv, pv, n_classes) if select == 'balanced' else float((pv == yv).mean())
        if score >= best[0]:
            best = (score, copy.deepcopy(net.state_dict()), ep + 1)
        if log and (ep % 5 == 4 or ep == epochs - 1):
            print('epoch %2d  loss %.3f  val %s %.3f' % (ep + 1, loss.item(), select, score), flush=True)
    net.load_state_dict(best[1])
    return net.cpu().eval(), best[0], best[2]


def train(epochs=50, seed=0, out=MODEL_PATH):
    import torch
    d = np.load(DATA)
    X, y, var = d['X'], d['y'], d['var']
    tr, va, te = var < TRAIN_VARIANTS, (var >= TRAIN_VARIANTS) & (var < VAL_VARIANTS), var >= VAL_VARIANTS
    net, best_val, best_epoch = fit(X[tr], y[tr], X[va], y[va], len(L.CLASSES), epochs, seed)
    acc_t = float((predict(net, X[tr]) == y[tr]).mean())
    pe = predict(net, X[te])
    acc_e = float((pe == y[te]).mean())
    n = len(L.CLASSES)
    cm = np.zeros((n, n), int)
    for t, p in zip(y[te], pe):
        cm[t, p] += 1
    os.makedirs(os.path.dirname(out), exist_ok=True)
    torch.save(net.cpu().state_dict(), out)
    metrics = dict(train_images=int(tr.sum()), val_images=int(va.sum()), test_images=int(te.sum()),
                   train_accuracy=acc_t, val_accuracy=best_val, test_accuracy=acc_e, best_epoch=best_epoch,
                   classes=L.CLASSES, test_confusion_rows_true_cols_pred=cm.tolist(),
                   device='cuda' if torch.cuda.is_available() else 'cpu', epochs=epochs)
    if out == MODEL_PATH:
        with open(os.path.join(os.path.dirname(MODEL_PATH), 'qc_metrics.json'), 'w') as f:
            json.dump(metrics, f, indent=2)
        contact_sheet(X, y, os.path.join(os.path.dirname(MODEL_PATH), 'qc_samples.png'))
    print('best epoch %d: train %.3f  val %.3f  TEST %.3f' % (best_epoch, acc_t, best_val, acc_e))
    print('test confusion matrix (rows = truth, cols = prediction, order %s):' % L.CLASSES)
    print(cm)
    print('model ->', out)


if __name__ == '__main__':
    ap = argparse.ArgumentParser()
    ap.add_argument('--reuse', action='store_true')
    ap.add_argument('--per-variant', type=int, default=16)
    ap.add_argument('--epochs', type=int, default=50)
    ap.add_argument('--out', default=MODEL_PATH, help='where to save the weights (default: the model the line uses)')
    a = ap.parse_args()
    if not (a.reuse and os.path.exists(DATA)):
        collect(a.per_variant)
    train(a.epochs, out=a.out)
