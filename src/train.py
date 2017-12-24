#!/usr/bin/python3
'''
Created on 6 Jan 2017

@author: Morris Franken
'''

import numpy as np
from os import listdir, path
from os.path import isdir, isfile, join, exists, dirname
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn import linear_model 
from sklearn.externals import joblib
from urllib.request import urlopen
from zipfile import ZipFile
from io import BytesIO

from featureExtractor import FeatureExtractor
from imageLoader import batchGenerator

# setup all the paths adn variables
file_dir         = dirname(__file__)
dataPath         = join(file_dir, "../Dataset/")
stelePath        = join(dataPath, "Manual/Preprocessed")
intermediatePath = join(file_dir, "../intermediates")
featurePath      = join(intermediatePath, "features.npy")
labelsPath       = join(intermediatePath, "labels.npy")
svmPath          = join(intermediatePath, "svm.pkl")
image_paths      = []
labels           = []
batch_size       = 200
if not exists(dataPath):
    print("downloading dataset (57.5MB)")
    url = urlopen("http://iamai.nl/downloads/GlyphDataset.zip")    
    with ZipFile(BytesIO(url.read())) as z:
        z.extractall(join(dataPath, ".."))

# check if the feature file is present, if so; there is no need to recompute the features
# The pre-computed features can also be downloaded from http://iamai.nl/downloads/features.npy
if not isfile(featurePath):
    print("indexing images...")
    Steles = [ join(stelePath,f) for f in listdir(stelePath) if isdir(join(stelePath,f)) ]
    for stele in Steles:    
        imagePaths = [ join(stele,f) for f in listdir(stele) if isfile(join(stele,f)) ]
        for path in imagePaths:
            image_paths.append(path)
            labels.append(path[(path.rfind("_") + 1): path.rfind(".")])
    
    featureExtractor = FeatureExtractor()
    features = []
    print("computing features...")
    for idx, (batch_images, _) in enumerate(batchGenerator(image_paths, labels, batch_size)):
        print("{}/{}".format((idx+1) * batch_size, len(labels)))
        features_ = featureExtractor.get_features(batch_images)
        features.append(features_)
    features = np.vstack(features)
    
    labels = np.asarray(labels)
    print("saving features...")
    np.save(featurePath, features)
    np.save(labelsPath, labels)
else:
    print("loading precomputed features and labels from {} and {}".format(featurePath, labelsPath))
    features = np.load(featurePath)
    labels = np.load(labelsPath)

# on to the SVM trainign phase
tobeDeleted = np.nonzero(labels == "UNKNOWN") # Remove the Unknown class from the database
features = np.delete(features,tobeDeleted, 0)
labels = np.delete(labels,tobeDeleted, 0)
numImages = len(labels)
trainSet, testSet, trainLabels, testLabels = train_test_split(features, labels, test_size=0.20, random_state=42) 

# Training SVM, feel free to use linear SVM (or another classifier for that matter) for faster training, however that will not give the confidence scores that can be used to rank hieroglyphs
print("training SVM...")
if 0: # optinal; either train 1 classifier fast, or search trough the parameter space by training multiple classifiers to sqeeze out that extra 2%
    clf = linear_model.LogisticRegression(C=10000)
else:
    svr = linear_model.LogisticRegression()
    parameters = {'C':[0.001, 0.01, 0.1, 1, 10, 100, 1000, 10000]}
    clf = GridSearchCV(svr, parameters, n_jobs=8)
clf.fit(trainSet, trainLabels)
    
print(clf)
print("finished training! saving...")
joblib.dump(clf, svmPath, compress=1) 

prediction = clf.predict(testSet)
accuracy = np.sum(testLabels == prediction) / float(len(prediction))

# for idx, pred in enumerate(prediction):
#     print("%-5s --> %s" % (testLabels[idx], pred))
print("accuracy = {}%".format(accuracy*100))
