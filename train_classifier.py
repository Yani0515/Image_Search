import argparse as ap
import os
import pickle
import sys

import cv2
import joblib
import numpy as np
from joblib.numpy_pickle_utils import xrange
from scipy.cluster.vq import vq, kmeans
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler
from sklearn.svm import LinearSVC

import ImageUtils

# Train the classifier through train set
# Extract features (SIFT) of each image (in all directories of the test set)
# Get a list of  Descriptors for each image
# Stack all the descriptors vertically in a numpy array
# 降维： Clustering descriptors using k-means into k clusters () 每个类的中心为 codebook 的内容
# Calculate histograms of features
# vector quantization
# scale （？）
# SVM - final classifier


# Step 1: load train image set ---------------------------

# Option A: Get the path of the training set from arguments
# parser = ap.ArgumentParser()
# parser.add_argument("-t", "--trainingSet", help="Path to Training Set", required="True")
# args = vars(parser.parse_args())
# train_path = args["trainingSet"]

# Option B: Get the path of the training set by line
train_path = 'caltech-101/sub'

# Get the training classes names and store them in a list
training_names = os.listdir(train_path)

# image_paths and the corresponding label in image_paths
train_image_paths = []
image_classes = []
class_id = 0
for training_name in training_names:
    train_dir = os.path.join(train_path, training_name)
    class_path = ImageUtils.make_img_list(train_dir)
    train_image_paths += class_path
    image_classes += [class_id] * len(class_path)
    class_id += 1

# Step 2: Feature Extraction ------------------------------------
# Create feature extraction and keypoint detector objects

# des_list - List where all the descriptors are stored
des_list = []

counter = 0
for image_path in train_image_paths:
    img_train = cv2.imread(image_path)
    sift = cv2.xfeatures2d.SIFT_create()
    key_points, des = sift.detectAndCompute(img_train, None)
    des_list.append((image_path, des))
    # Display descriptors storing progress
    # print("\r Store Descriptors Progress : {:.2f}%".format(counter * 100 / len(image_paths)), end="", flush=True)
    counter = counter + 1

print("Descriptors of Train Dataset Stored!")

# Stack all the descriptors vertically in a numpy array
descriptors = des_list[0][1]
for image_path, descriptor in des_list[1:]:
    descriptors = np.vstack((descriptors, descriptor))

# Step 3: Codebook construction-----------------------------------
# Apply Dimension reduction to local features
# Perform k-means clustering
k = 10
model = KMeans(n_clusters=k)
model.fit(descriptors)

# voc - clustering centers
voc = model.cluster_centers_

print("Finished Local Feature Clustering!")

# Step 4: vector quantization--------------------------------------
# Calculate the histogram of features
# im_features - List of image feature
# im_features = np.zeros((len(image_paths), k), "float32")
# for i in xrange(len(image_paths)):
#     words, distance = vq(des_list[i][1], voc)
#     for w in words:
#         im_features[i][w] += 1
train_img_features = ImageUtils.vector_quantization(train_image_paths, k, des_list, voc)

print("Finished histogram!")


# Step 4: Perform Tf-Idf vectorization -------------------------------
# TF （Term Frequency）
# IDF（Inverse Document Frequency）
# Feature Importance increases proportionally with its number of occurrences in the document,
# but at the same time decreases inversely with its frequency in the corpus
nbr_occurrence = np.sum((train_img_features > 0) * 1, axis=0)
idf = np.array(np.log((1.0 * len(train_image_paths) + 1) / (1.0 * nbr_occurrence + 1)), 'float32')

print("Finished vectorization!")

# Scaling the words
stdSlr = StandardScaler().fit(train_img_features)
train_img_features = stdSlr.transform(train_img_features)

comp = []
for i in range(len(train_image_paths)):
    comp.append([train_image_paths[i], train_img_features[i]])

# load vocabulary to db
with open('vocabulary.pkl', 'wb') as vocabulary:
    pickle.dump(comp, vocabulary)
# print('vocabulary is:', voc.name, voc.nbr_words)

print("Finished scaling!")


# For classifier: Train the Linear SVM -----------
clf = LinearSVC()
clf.fit(train_img_features, np.array(image_classes))

# Save the SVM
joblib.dump((clf, training_names, stdSlr, k, voc), "bof.pkl", compress=3)
