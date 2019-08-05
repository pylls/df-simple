#!/usr/bin/env python
# -*- coding: utf-8 -*-

from keras.models import load_model
import cPickle as pickle
import numpy as np
import os
import argparse

# prevents spammy output from TF
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'

ap = argparse.ArgumentParser()
ap.add_argument("-lm", "--labelsm", required=True, help="The corresponding monitored labels (.pkl)")
ap.add_argument("-lu", "--labelsu", required=True, help="The corresponding unmonitored labels (.pkl)")
ap.add_argument("-dm", "--datasetm", required=True, help="The testing monitored dataset (.pkl)")
ap.add_argument("-du", "--datasetu", required=True, help="The testing unmonitored dataset (.pkl)")
ap.add_argument("-m", "--model", required=True, help="The trained model (.h5)")

args = vars(ap.parse_args())

def main():
    print("Loading labels %s and %s..." % (args["labelsm"], args["labelsu"]))
    labels_mon, labels_unmon = load_labels()
    print("Loading dataset %s and %s..." % (args["datasetm"], args["datasetu"]))
    testing_mon, testing_unmon = load_dataset()
    print("Loading model (%s)..." % (args["model"]))
    trained_model = load_model(args["model"])

    print("Predicting labels...")
    predictions_mon = trained_model.predict(testing_mon[:, :, np.newaxis], verbose=2)
    predictions_unmon = trained_model.predict(testing_unmon[:, :, np.newaxis], verbose=2)

    print("Saving predictions to %s..." % (args["savepred"]))
    pickle.dump([predictions_mon, predictions_unmon], open(args["savepred"], "wb")

def load_dataset():
    with open(args["datasetm"], 'rb') as handle: # a sequence of traffic directions
        testing_mon = np.array(pickle.load(handle)).astype('float32')
    with open(args["datasetu"], 'rb') as handle:
        testing_unmon = np.array(pickle.load(handle)).astype('float32')
    return testing_mon, testing_unmon

def load_labels():
    with open(args["labelsm"], 'rb') as handle: #a sequence of corresponding labels
        labels_mon = list(np.array(pickle.load(handle)))
    with open(args["labelsu"], 'rb') as handle:
        labels_unmon = list(np.array(pickle.load(handle)))
    return labels_mon, labels_unmon

if __name__ == "__main__":
    main()
