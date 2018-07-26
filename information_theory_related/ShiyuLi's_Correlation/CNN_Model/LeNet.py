import keras

class LeNet:
    @staticmethod
    def build_model(width, height, channels, cfg, num_classes, weights_Path=None):
        model = keras.Sequential()

        model.add(keras.layers.Convolution2D(filters=6, kernel_size=5, padding='same',
                                             input_shape=(width, height, channels)))
        model.add(keras.layers.Activation(cfg['ACTIVATION']))
        model.add(keras.layers.MaxPooling2D(pool_size=(2,2), strides=(2,2)))

        model.add(keras.layers.Convolution2D(filters=16,  kernel_size=5, padding='same'))
        model.add(keras.layers.Activation(cfg['ACTIVATION']))
        model.add(keras.layers.MaxPooling2D(pool_size=(2,2), strides=(2,2)))

        model.add(keras.layers.Flatten())
        model.add(keras.layers.Dense(120))
        model.add(keras.layers.Activation(cfg['ACTIVATION']))

        model.add(keras.layers.Dense(84))
        model.add(keras.layers.Activation(cfg['ACTIVATION']))

        model.add(keras.layers.Dense(num_classes))
        model.add(keras.layers.Activation('softmax'))

        if weights_Path is not None:
            model.load_weights(weights_Path)

        return model
