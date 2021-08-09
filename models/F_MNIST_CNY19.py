from tensorflow.python.keras.models import Sequential
from tensorflow.python.keras.layers import Convolution2D, Dense, MaxPooling2D, Flatten
from tensorflow.keras.optimizers import Adam


# Note: 'weights' is ignored and just present for compatibility with other networks
def F_MNIST_CNY19(classes, input_shape, weights=None):
    model = Sequential()

    model.add(Convolution2D(40, (5, 5), strides=(1, 1), input_shape=input_shape, activation="relu"))
    model.add(MaxPooling2D(pool_size=(2, 2)))

    model.add(Convolution2D(20, (5, 5), strides=(1, 1), activation="relu"))
    model.add(MaxPooling2D(pool_size=(2, 2)))

    model.add(Flatten())
    model.add(Dense(320, activation='relu'))
    model.add(Dense(160, activation='relu'))
    model.add(Dense(80, activation='relu'))
    model.add(Dense(40, activation='relu'))
    model.add(Dense(classes, activation='softmax'))

    opt = Adam(lr=0.001)
    model.compile(optimizer=opt,
                  loss='categorical_crossentropy',
                  metrics=['accuracy'])

    return model
