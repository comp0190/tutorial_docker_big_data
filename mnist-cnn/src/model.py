from tensorflow.keras.layers import Conv2D, BatchNormalization, AveragePooling2D, Dense, Activation, Flatten, Dropout
from tensorflow.keras import Sequential

def initiate_model():
    model = Sequential()

    model.add(Conv2D(8, kernel_size=(3,3), padding='same', use_bias=False, input_shape=(1,28,28)))
    model.add(BatchNormalization())
    model.add(Activation('relu'))

    model.add(AveragePooling2D((2,2), strides=(2,2), padding='same'))

    model.add(Conv2D(16, kernel_size=(3,3), padding='same', use_bias=False))
    model.add(BatchNormalization())
    model.add(Activation('relu'))

    model.add(AveragePooling2D((2,2), strides=(2,2), padding='same'))

    model.add(Conv2D(16, kernel_size=(3,3), padding='same', use_bias=False))
    model.add(BatchNormalization())
    model.add(Activation('relu'))

    model.add(Dropout(0.2))

    model.add(AveragePooling2D((2,2), strides=(2,2), padding='same'))

    model.add(Conv2D(32, kernel_size=(3,3), padding='same', use_bias=False))
    model.add(BatchNormalization())
    model.add(Activation('relu'))

    model.add(Dropout(0.4))

    model.add(Conv2D(10, kernel_size=(3,3), padding='same'))
    model.add(AveragePooling2D((4,4), strides=(4,4), padding='same'))
    model.add(Flatten())

    model.add(Activation('softmax'))
    return model
