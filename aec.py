#! /usr/bin/env python3
# -*- encoding:utf-8 -*-

import numpy as np
import numpy.random as rd
from keras.layers import Dense
from keras.models import Sequential


def create_model():
    model = Sequential()
    model.add(Dense(256, input_dim=614, activation='relu'))
    model.add(Dense(8, activation='relu'))
    model.add(Dense(3, activation='softmax'))
    model.compile(loss='categorical_crossentropy',  optimizer='adam', metrics=['accuracy'])
    return model
