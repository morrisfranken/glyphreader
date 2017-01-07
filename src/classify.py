#!/usr/bin/python3
'''
Created on 7 Jan 2017

@author: Morris Franken

Classifies an image or a set of images inside a directory
'''

import numpy as np
import os
import sys
from sklearn.externals import joblib
from os.path import join, isdir, isfile
from os import listdir

from featureExtractor import FeatureExtractor
from imageLoader import loadBatch

file_dir = os.path.dirname(__file__)
intermediatePath = join(file_dir, "../intermediates")
examplePath = join(file_dir, "../examples")
svmPath = join(intermediatePath, "svm.pkl")

# Input parameters, could be a path to an image or directory. In the case of a directory, all files in that directory will be evaluated
# If no parameters are specified, the default example folder will be used
inputPath = sys.argv[1] if len(sys.argv) > 1 else "examples"
if isdir(inputPath):
    imagePaths = [join(inputPath, f) for f in listdir(inputPath) if isfile(join(inputPath, f))]
else:
    imagePaths = [inputPath,]
    
print("loading images...")
Images = loadBatch(imagePaths)
print("loading SVM model...")
clf = joblib.load(svmPath);
    
print("Extracting features, this may take a while for large collections of images...")
extractor = FeatureExtractor()
features  = extractor.get_features(Images)

classes = clf.best_estimator_.classes_ if hasattr(clf, "best_estimator_") else clf.classes_
print("Predicting the Hieroglyph type...")
prob = np.array(clf.predict_proba(features))
top5_i = np.argsort(-prob)[:,0:5]
top5_s = np.array([prob[row,top5_i[row]] for row, top5_i_row in enumerate(top5_i)])  
top5_n = classes[top5_i]

print("{:<25} ::: {}".format("image name", "top 5 best matching hieroglyphs"))
for idx, path in enumerate(imagePaths):
    print("{:<25} --> {}".format(os.path.basename(path), top5_n[idx]))