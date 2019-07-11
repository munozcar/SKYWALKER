import numpy as np


def normalize_pld(pixel_int):
    pixel_int /= np.sum(pixel_int, axis=0)
    return pixel_int
