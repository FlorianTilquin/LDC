#! /usr/bin/env python3
# -*- encoding:utf-8 -*-

from keras import regularizers
from keras.layers import Dense, Dropout
from keras.models import Sequential


def create_model():
    model = Sequential()
    model.add(Dense(256, input_dim=614, activation='relu',
                    kernel_regularizer=regularizers.l2(0.5)))
    # model.add(Dropout(0.3))
    model.add(Dense(64, activation='relu'))
    model.add(Dense(16, activation='relu'))
    # model.add(Dropout(0.3))
    model.add(Dense(3, activation='softmax'))
    model.compile(loss='categorical_crossentropy',  optimizer='adam', metrics=['accuracy'])
    return model
