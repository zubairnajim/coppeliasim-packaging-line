"""Machine-vision quality control: a small CNN that grades soap bars as good / rework / scrap
from a 128x128 top-down camera image."""
import os

import numpy as np

from . import layout as L

MODEL_PATH = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'models', 'qc_cnn.pt')


def grab(sim, cam):
    """Trigger a vision sensor and return its image as an HxWx3 uint8 RGB array."""
    sim.handleVisionSensor(cam)
    img, res = sim.getVisionSensorImg(cam)
    return np.frombuffer(img, np.uint8).reshape(res[1], res[0], 3)[::-1].copy()


def build_net():
    import torch
    import torch.nn as nn

    def block(i, o):
        # GroupNorm instead of BatchNorm: identical behaviour in training and inference
        return nn.Sequential(nn.Conv2d(i, o, 3, padding=1), nn.GroupNorm(4, o), nn.ReLU(), nn.MaxPool2d(2))

    class Head(nn.Module):
        """Average + max pooling: the max branch keeps the response of a single tiny blemish."""

        def __init__(self, ch, n):
            super().__init__()
            self.drop = nn.Dropout(0.3)
            self.fc = nn.Linear(2 * ch, n)

        def forward(self, x):
            return self.fc(self.drop(torch.cat([x.mean((2, 3)), x.amax((2, 3))], 1)))

    return nn.Sequential(block(3, 16), block(16, 32), block(32, 64), block(64, 64), Head(64, len(L.CLASSES)))


class Classifier:
    def __init__(self, net):
        import torch
        self.torch = torch
        self.net = net.eval()

    @classmethod
    def load(cls, path=MODEL_PATH):
        if not os.path.exists(path):
            return None
        import torch
        net = build_net()
        net.load_state_dict(torch.load(path, map_location='cpu'))
        return cls(net)

    def predict(self, img):
        torch = self.torch
        x = torch.from_numpy(img).permute(2, 0, 1).float().div(255).unsqueeze(0)
        with torch.no_grad():
            p = torch.softmax(self.net(x), 1)[0]
        k = int(p.argmax())
        return L.CLASSES[k], float(p[k])
