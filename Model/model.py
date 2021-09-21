from numpy import mean
from numpy import std
from numpy import dstack
from pandas import read_csv
from keras.models import Sequential
from keras.layers import Dense
from keras.layers import Flatten
from keras.layers import Dropout
from keras.layers import LSTM
from keras.layers import TimeDistributed
from keras.layers.convolutional import Conv1D
from keras.layers.convolutional import MaxPooling1D
from matplotlib import pyplot
from tensorflow.keras.utils import to_categorical
# fit and evaluate a model

def evaluate_model(
    trainX,
    trainy,
    testX,
    testy,
    ):

    # define model

    (verbose, epochs, batch_size) = (0, 25, 64)
    (n_timesteps, n_features, n_outputs) = (trainX.shape[1],
            trainX.shape[2], trainy.shape[1])

    # reshape data into time steps of sub-sequences

    (n_steps, n_length) = (4, 32)
    trainX = trainX.reshape((trainX.shape[0], n_steps, n_length,
                            n_features))
    testX = testX.reshape((testX.shape[0], n_steps, n_length,
                          n_features))

    # define model

    model = Sequential()
    model.add(TimeDistributed(Conv1D(filters=64, kernel_size=3,
              activation='relu'), input_shape=(None, n_length,
              n_features)))
    model.add(TimeDistributed(Conv1D(filters=64, kernel_size=3,
              activation='relu')))
    model.add(TimeDistributed(Dropout(0.5)))
    model.add(TimeDistributed(MaxPooling1D(pool_size=2)))
    model.add(TimeDistributed(Flatten()))
    model.add(LSTM(100))
    model.add(Dropout(0.5))
    model.add(Dense(100, activation='relu'))
    model.add(Dense(n_outputs, activation='softmax'))
    model.compile(loss='categorical_crossentropy', optimizer='adam',
                  metrics=['accuracy'])

    # fit network

    model.fit(trainX, trainy, epochs=epochs, batch_size=batch_size,
              verbose=verbose)
    model.save('activity_recog')
