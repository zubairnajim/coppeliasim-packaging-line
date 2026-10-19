"""Fetch two public datasets of REAL industrial defects and convert them to compact arrays.

    python -I factory/real_data.py

Stand-alone on purpose (standard library + NumPy + OpenCV, no project imports) so that it can run
in Python's isolated mode: the archives are third-party data.  They are downloaded to a temporary
folder, read in memory without being extracted, and deleted afterwards.  Only the converted
128x128 greyscale arrays are kept, in factory/data/real/ (not committed to the repository).

Sources
  magnetic_tile  Huang, Qiu, Yuan: "Surface defect saliency of magnetic tile", The Visual Computer 2020.
                 github.com/abin24/Magnetic-tile-defect-datasets.  (authors' repository, research use)
  neu_cls        Song, Yan: NEU surface defect database, Northeastern University (China).
                 Copy on Figshare, article 28903550, CC BY 4.0.
"""
import os
import re
import shutil
import sys
import tempfile
import urllib.request
import zipfile

import cv2
import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
OUT = os.path.join(HERE, 'data', 'real')
SIZE = 128
MAX_DOWNLOAD = 300 * 2 ** 20
MAX_MEMBER = 20 * 2 ** 20

SOURCES = {
    'magnetic_tile': 'https://codeload.github.com/abin24/Magnetic-tile-defect-datasets./zip/refs/heads/master',
    'neu_cls': 'https://ndownloader.figshare.com/files/54094775',
}


def download(url, path):
    req = urllib.request.Request(url, headers={'User-Agent': 'factory-line-benchmark'})
    n = 0
    with urllib.request.urlopen(req, timeout=60) as r, open(path, 'wb') as f:
        while True:
            chunk = r.read(1 << 20)
            if not chunk:
                break
            n += len(chunk)
            if n > MAX_DOWNLOAD:
                raise RuntimeError('download larger than expected, aborted')
            f.write(chunk)
    return n


def images(zip_path):
    """Yield (member name, greyscale image) for every JPEG in the archive, read in memory."""
    with zipfile.ZipFile(zip_path) as z:
        for info in z.infolist():
            if info.is_dir() or not info.filename.lower().endswith('.jpg') or info.file_size > MAX_MEMBER:
                continue
            img = cv2.imdecode(np.frombuffer(z.read(info), np.uint8), cv2.IMREAD_GRAYSCALE)
            if img is not None:
                yield info.filename, img


def convert_magnetic_tile(zip_path):
    """Folders MT_<class>/Imgs/expN_num_<id>.jpg.  Every picture has its own <id>; nothing in the
    archive says which pictures show the same physical tile."""
    X, labels, groups = [], [], []
    for name, img in images(zip_path):
        m = re.search(r'/MT_([A-Za-z]+)/Imgs/exp\d+_num_(\d+)\.jpg$', name)
        if not m:
            continue
        X.append(cv2.resize(img, (SIZE, SIZE), interpolation=cv2.INTER_AREA))
        labels.append(m.group(1).lower())
        groups.append(int(m.group(2)))
    order = ['free', 'blowhole', 'break', 'crack', 'fray', 'uneven']       # defect-free first
    return X, labels, groups, order


def convert_neu(zip_path):
    """Files <split>/<split>/images/<class>_<n>.jpg, six defect types, 300 pictures each."""
    X, labels, groups = [], [], []
    for name, img in images(zip_path):
        m = re.search(r'/images/([A-Za-z_\-]+)_(\d+)\.jpg$', name)
        if not m:
            continue
        X.append(cv2.resize(img, (SIZE, SIZE), interpolation=cv2.INTER_AREA))
        labels.append(m.group(1).lower())
        groups.append(len(groups))                                          # no grouping information
    return X, labels, groups, sorted(set(labels))


def main():
    os.makedirs(OUT, exist_ok=True)
    tmp = tempfile.mkdtemp(prefix='real_defects_')
    try:
        for name, convert in (('magnetic_tile', convert_magnetic_tile), ('neu_cls', convert_neu)):
            path = os.path.join(tmp, name + '.zip')
            size = download(SOURCES[name], path)
            X, labels, groups, classes = convert(path)
            os.remove(path)
            y = np.array([classes.index(c) for c in labels], np.int64)
            np.savez_compressed(os.path.join(OUT, name + '.npz'), X=np.array(X, np.uint8), y=y,
                                group=np.array(groups, np.int64), classes=np.array(classes))
            counts = {c: int((y == i).sum()) for i, c in enumerate(classes)}
            print('%s: downloaded %.1f MB, kept %d images, %d groups, classes %s'
                  % (name, size / 2 ** 20, len(X), len(set(groups)), counts), flush=True)
    finally:
        shutil.rmtree(tmp, ignore_errors=True)


if __name__ == '__main__':
    sys.exit(main())
