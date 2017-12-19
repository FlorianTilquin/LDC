#! /usr/bin/env python3
# -*- encoding:utf-8 -*-

import numpy as np
from keras.utils import np_utils
from keras.wrappers.scikit_learn import KerasClassifier
from sklearn.cross_validation import train_test_split
from sklearn.model_selection import KFold, cross_val_score
from sklearn.preprocessing import LabelEncoder

from aec import create_model as CM

X, y = np.load('wdwf.npy'), np.load('labels0.npy')
X = X - np.mean(X, 0)[None, :]
X = X / np.std(X, 0)[None, :]
encoder = LabelEncoder()
encoder.fit(y)
encoded_Y = encoder.transform(y)
# convert integers to dummy variables (i.e. one hot encoded)
dummy_y = np_utils.to_categorical(encoded_Y)

estimator = KerasClassifier(build_fn=CM, epochs=250, batch_size=250, verbose=1)
kfold = KFold(n_splits=10, shuffle=True)
dummy_y.shape
# results = cross_val_score(estimator, X, dummy_y, cv=kfold)
X_train, X_test, Y_train, Y_test = train_test_split(X, y, test_size=0.33)
estimator.fit(X_train, Y_train)
predictions = estimator.predict(X_test)
