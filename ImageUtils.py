import os

import numpy as np
from joblib.numpy_pickle_utils import xrange
from scipy.cluster.vq import vq


def make_img_list(path):
    """
    The function make_img_list returns all the names of the files in
    the directory path supplied as argument to the function.
    """
    return [os.path.join(path, f) for f in os.listdir(path)]


def vector_quantization(image_paths, k, des_list, voc):
    features = np.zeros((len(image_paths), k), "float32")
    for i in xrange(len(image_paths)):
        words, distance = vq(des_list[i][1], voc)
        for w in words:
            features[i][w] += 1
    return features
