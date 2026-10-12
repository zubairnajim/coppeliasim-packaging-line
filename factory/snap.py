"""Render the hidden View_* cameras to PNG files:  python -m factory.snap out_dir [names...]"""
import os, sys
import numpy as np, cv2
from coppeliasim_zmqremoteapi_client import RemoteAPIClient
from . import layout as L

def snap(sim, name, path):
    h = sim.getObject('/View_' + name)
    sim.handleVisionSensor(h)
    img, res = sim.getVisionSensorImg(h)
    a = np.frombuffer(img, np.uint8).reshape(res[1], res[0], 3)[::-1, :, ::-1]
    cv2.imwrite(path, a)

if __name__ == '__main__':
    sim = RemoteAPIClient().require('sim')
    out = sys.argv[1]; os.makedirs(out, exist_ok=True)
    for n in (sys.argv[2:] or L.VIEWS):
        snap(sim, n, os.path.join(out, n + '.png'))
