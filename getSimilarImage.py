import os
import pickle

import cv2
import joblib
import numpy as np
from joblib.numpy_pickle_utils import xrange
from scipy.cluster.vq import vq

from ImageUtils import vector_quantization

clf, classes_names, stdSlr, k, voc = joblib.load("bof.pkl")

# set test output path
out_path = "Output"

# set test image root
test_image_root = "Input"
# set test image path
test_image_paths = os.listdir(test_image_root)

# des_list - List where all the descriptors are stored
des_list = []

# Get the testing image path store them in a list
# get descriptors for all testing images
for test_image_path in test_image_paths:
    file_path = test_image_root + "/" + test_image_path
    img = cv2.imread(file_path)
    if img is None:
        print("No such file {}\nCheck if the file exists".format(test_image_path))
        exit()
    sift = cv2.xfeatures2d.SIFT_create()
    key_points, des = sift.detectAndCompute(img, None)
    des_list.append((test_image_paths, des))

# Stack all the descriptors vertically in a numpy array
descriptors = des_list[0][1]

for test_image_path, descriptor in des_list[0:]:
    descriptors = np.vstack((descriptors, descriptor))

# Apply Dimension reduction to local features
# test_features = np.zeros((len(test_image_paths), k), "float32")
# for i in xrange(len(test_image_paths)):
#     words, distance = vq(des_list[i][1], voc)
#     for w in words:
#         test_features[i][w] += 1
test_img_features = vector_quantization(test_image_paths, k, des_list, voc)

# Perform Tf-Idf vectorization
nbr_occurrences = np.sum((test_img_features > 0) * 1, axis=0)
idf = np.array(np.log((1.0 * len(test_image_paths) + 1) / (1.0 * nbr_occurrences + 1)), 'float32')

# Scale the features
test_img_features = stdSlr.transform(test_img_features)

# Perform the predictions
predictions = [classes_names[i] for i in clf.predict(test_img_features)]

with open('vocabulary.pkl', 'rb') as f:
    vocabulary = pickle.load(f)

distance = []

for word in vocabulary:
    distance.append([word[0], np.linalg.norm(word[1] - test_img_features[0])])

# sort by distance
distance.sort(key=lambda t: t[1])

# num_of_similar_img - number of outputs
num_of_similar_img = 10
for i in range(len(distance)):
    if i > num_of_similar_img:
        break
    path = distance[i][0]
    image = cv2.imread(path)
    cv2.imwrite(os.path.join(out_path, 'rank{0}.jpg'.format(i)), image)

for test_image_path, prediction in zip(test_image_paths, predictions):
    image = cv2.imread(test_image_path)
    print(prediction)


