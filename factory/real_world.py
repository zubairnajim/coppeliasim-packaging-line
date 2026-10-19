"""Does the inspection pipeline hold up on REAL defects?

    python -I factory/real_data.py        # once: fetch + convert two public datasets (about 77 MB)
    python -m factory.real_world          # run the benchmark, write results and a sample sheet

Four questions, each answered on pictures the network never saw during training:

  1. Transfer      - does the network trained on simulated soap bars recognise real defects as it is?
  2. Retrain       - does the same network with the same training recipe learn real defects when
                     it is given real pictures?
  3. Train longer  - the real sets are smaller than the simulated one, so 50 epochs are far fewer
                     weight updates; what happens with 200 epochs and nothing else changed?
  4. Accept/reject - the decision a line actually takes: good part or defective part
                     (magnetic tiles only; class-weighted because 71 % of the tiles are good).

Every retraining run uses a stratified 60 / 20 / 20 split (train / validation / test), repeated
with several random seeds; the validation part picks the epoch, the test part gives the score.
"""
import argparse
import json
import os

import cv2
import numpy as np

from . import layout as L
from .train_qc import balanced_accuracy, fit, predict
from .vision import MODEL_PATH, build_net

HERE = os.path.dirname(os.path.abspath(__file__))
DATA = os.path.join(HERE, 'data', 'real')
RESULTS = os.path.join(HERE, 'models', 'real_world_results.json')
SHEET = os.path.join(os.path.dirname(HERE), 'docs', 'img', 'real-defects.png')

DATASETS = {
    'magnetic_tile': dict(title='Magnetic tile surfaces', good='free',
                          source='Huang et al. 2020, github.com/abin24/Magnetic-tile-defect-datasets.'),
    'neu_cls': dict(title='Hot-rolled steel strip (NEU-CLS)', good=None,
                    source='Song & Yan, Northeastern University; Figshare copy 28903550 (CC BY 4.0)'),
}


def load(name):
    path = os.path.join(DATA, name + '.npz')
    if not os.path.exists(path):
        raise SystemExit('missing %s - run:  python -I factory/real_data.py' % path)
    d = np.load(path)
    X = np.repeat(d['X'][:, :, :, None], 3, axis=3)          # greyscale -> the 3 channels the network expects
    return X, d['y'], [str(c) for c in d['classes']]


def split(y, seed, parts=(0.6, 0.2, 0.2)):
    """Stratified train / validation / test indices."""
    rng = np.random.default_rng(seed)
    out = [[], [], []]
    for c in np.unique(y):
        idx = rng.permutation(np.where(y == c)[0])
        a, b = int(round(parts[0] * len(idx))), int(round((parts[0] + parts[1]) * len(idx)))
        for k, part in enumerate((idx[:a], idx[a:b], idx[b:])):
            out[k].extend(part.tolist())
    return [np.array(sorted(o)) for o in out]


def score(y, pred, classes, good=None):
    n = len(classes)
    cm = np.zeros((n, n), int)
    for t, p in zip(y, pred):
        cm[t, p] += 1
    m = dict(accuracy=float((pred == y).mean()), balanced_accuracy=balanced_accuracy(y, pred, n),
             recall={c: float(cm[i, i] / max(1, cm[i].sum())) for i, c in enumerate(classes)}, confusion=cm.tolist())
    if good is not None:                      # the two numbers a production line cares about
        g = classes.index(good)
        m['defect_catch_rate'] = float((pred[y != g] != g).mean())      # defective parts that are stopped
        m['false_reject_rate'] = float((pred[y == g] != g).mean())      # good parts that are thrown away
    return m


def summarise(runs):
    keys = [k for k in ('accuracy', 'balanced_accuracy', 'defect_catch_rate', 'false_reject_rate') if k in runs[0]]
    s = {k: dict(mean=float(np.mean([r[k] for r in runs])), min=float(np.min([r[k] for r in runs])),
                 max=float(np.max([r[k] for r in runs]))) for k in keys}
    s['recall_mean'] = {c: float(np.mean([r['recall'][c] for r in runs])) for c in runs[0]['recall']}
    s['confusion_sum'] = np.sum([r['confusion'] for r in runs], axis=0).tolist()
    return s


def retrain(name, seeds, epochs=50, binary=False, keep=None):
    """Same network, same recipe, real pictures.  `binary` merges all defect types into one
    'defect' class and weights the loss by inverse class frequency.  `keep` receives the test
    predictions of the first seed."""
    X, y, classes = load(name)
    good = DATASETS[name]['good']
    y_full = y
    if binary:
        y = (y != classes.index(good)).astype(np.int64)
        classes = [good, 'defect']
    runs = []
    for seed in seeds:
        tr, va, te = split(y_full, seed)        # always stratified by the detailed class
        w = None
        if binary:
            counts = np.bincount(y[tr], minlength=2)
            w = (len(tr) / (2 * counts)).tolist()
        net, _, best_epoch = fit(X[tr], y[tr], X[va], y[va], len(classes), epochs, seed, class_weights=w,
                                 select='balanced' if binary else 'accuracy', log=False)
        pred = predict(net, X[te])
        r = score(y[te], pred, classes, good)
        r['train_accuracy'] = float((predict(net, X[tr]) == y[tr]).mean())
        r['best_epoch'] = best_epoch
        runs.append(r)
        if keep is not None and seed == seeds[0]:
            keep.update(X=X[te], y=y[te], pred=pred, classes=classes)
        print('  %-14s %3d epochs%s seed %d: train %.3f  test %.3f  balanced %.3f' % (
            name, epochs, ' binary' if binary else '', seed, r['train_accuracy'], r['accuracy'], r['balanced_accuracy']), flush=True)
    counts = np.bincount(y, minlength=len(classes))
    out = dict(images=int(len(y)), train_images=int(len(split(y_full, 0)[0])), test_images=int(len(split(y_full, 0)[2])),
               epochs=epochs, classes=classes, class_counts={c: int(n) for c, n in zip(classes, counts)},
               majority_baseline=float(counts.max() / counts.sum()), seeds=list(seeds), runs=runs, **summarise(runs))
    out['train_accuracy_mean'] = float(np.mean([r['train_accuracy'] for r in runs]))
    return out


def transfer():
    """The network that runs the simulated line, applied unchanged to real pictures."""
    import torch
    net = build_net(len(L.CLASSES))
    net.load_state_dict(torch.load(MODEL_PATH, map_location='cpu'))
    out = {}
    for name, info in DATASETS.items():
        X, y, classes = load(name)
        pred = predict(net, X)                 # 0 = good, 1 = rework, 2 = scrap
        flagged = pred != 0
        r = dict(images=int(len(y)), verdicts={c: int((pred == i).sum()) for i, c in enumerate(L.CLASSES)})
        if info['good']:
            g = classes.index(info['good'])
            r['defect_catch_rate'] = float(flagged[y != g].mean())
            r['false_reject_rate'] = float(flagged[y == g].mean())
            r['accuracy_good_vs_defect'] = float((flagged == (y != g)).mean())
            r['balanced_accuracy'] = (r['defect_catch_rate'] + 1 - r['false_reject_rate']) / 2
        else:
            r['defect_catch_rate'] = float(flagged.mean())          # every picture shows a defect
        out[name] = r
    return out


def sample_sheet(rows, path, per_row=10):
    """One row of unseen test pictures per dataset: truth above, prediction below, misses framed."""
    T, PAD, CAP = 128, 6, 38
    bg, ink, dim, red = (25, 26, 26), (255, 255, 255), (183, 194, 195), (59, 59, 208)
    sheet = np.full((len(rows) * (T + CAP + 34) + 10, per_row * (T + PAD) + PAD, 3), bg, np.uint8)
    font = cv2.FONT_HERSHEY_SIMPLEX
    for r, (title, k) in enumerate(rows):
        y0 = r * (T + CAP + 34) + 10
        cv2.putText(sheet, title, (PAD, y0 + 14), font, 0.5, ink, 1, cv2.LINE_AA)
        wrong = np.where(k['pred'] != k['y'])[0][:3].tolist()
        right = [i for i in np.random.default_rng(0).permutation(len(k['y'])) if k['pred'][i] == k['y'][i]]
        picks = sorted(right[:per_row - len(wrong)] + wrong, key=lambda i: (k['y'][i], i))
        for c, i in enumerate(picks):
            x0, ty = PAD + c * (T + PAD), y0 + 24
            sheet[ty:ty + T, x0:x0 + T] = k['X'][i]
            ok = k['pred'][i] == k['y'][i]
            if not ok:
                cv2.rectangle(sheet, (x0, ty), (x0 + T - 1, ty + T - 1), red, 2)
            cv2.putText(sheet, k['classes'][k['y'][i]], (x0, ty + T + 14), font, 0.4, ink, 1, cv2.LINE_AA)
            cv2.putText(sheet, ('= ' if ok else 'X  ') + k['classes'][k['pred'][i]], (x0, ty + T + 30), font, 0.4,
                        dim if ok else red, 1, cv2.LINE_AA)
    os.makedirs(os.path.dirname(path), exist_ok=True)
    cv2.imwrite(path, sheet)


def pct(d):
    return '%5.1f%% (%.1f-%.1f)' % (100 * d['mean'], 100 * d['min'], 100 * d['max'])


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument('--seeds', type=int, default=5)
    a = ap.parse_args()
    seeds = list(range(a.seeds))
    kept = {n: {} for n in DATASETS}

    print('1. transfer: simulated-bar network on real pictures')
    res = dict(transfer=transfer(), as_is={}, longer={}, accept_reject={})
    for name, r in res['transfer'].items():
        print('  %-14s verdicts %s  catch %.1f%%%s' % (name, r['verdicts'], 100 * r['defect_catch_rate'],
              '  false reject %.1f%%' % (100 * r['false_reject_rate']) if 'false_reject_rate' in r else ''))
    print('2. retrain: same network and recipe (50 epochs) on real pictures')
    for name in DATASETS:
        res['as_is'][name] = retrain(name, seeds, epochs=50)
    print('3. train longer: 200 epochs, nothing else changed')
    for name in DATASETS:
        res['longer'][name] = retrain(name, seeds, epochs=200, keep=kept[name])
    print('4. accept / reject decision on magnetic tiles (200 epochs, class-weighted)')
    res['accept_reject']['magnetic_tile'] = retrain('magnetic_tile', seeds, epochs=200, binary=True)

    res['protocol'] = dict(split='stratified 60/20/20 train/validation/test', seeds=seeds,
                           input='128x128 greyscale copied to 3 channels', sources={n: d['source'] for n, d in DATASETS.items()})
    with open(RESULTS, 'w') as f:
        json.dump(res, f, indent=1)
    sample_sheet([(DATASETS[n]['title'] + ' - unseen test pictures: true class, then the network\'s answer', kept[n])
                  for n in DATASETS], SHEET)

    print('\nSUMMARY  mean over %d seeds (min-max), test pictures only' % len(seeds))
    for label in ('as_is', 'longer', 'accept_reject'):
        for name, r in res[label].items():
            line = '  %-13s %-14s train %5.1f%%  test accuracy %s  balanced %s  [majority guess %.1f%%]' % (
                label, name, 100 * r['train_accuracy_mean'], pct(r['accuracy']), pct(r['balanced_accuracy']), 100 * r['majority_baseline'])
            if 'defect_catch_rate' in r:
                line += '\n%31scatch rate %s  false rejects %s' % ('', pct(r['defect_catch_rate']), pct(r['false_reject_rate']))
            print(line)
            print('%31srecall per class: %s' % ('', {c: round(v, 2) for c, v in r['recall_mean'].items()}))
    print('results ->', RESULTS)
    print('samples ->', SHEET)


if __name__ == '__main__':
    main()
