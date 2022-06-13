from PIL.Image import Image
from numpy import ndarray
import numpy as np


def img2ndarray(img: Image, size: (int, int)) -> ndarray:
    img = img.convert("RGB").resize(size)
    out = np.asarray(img, dtype=float)
    return out.transpose((2, 0, 1))
