import keras
class VGG16:
    @staticmethod
    def build_model(width, height, channels, cfg, num_classes,weight_decay, weights_Path=None):
        model = keras.Sequential()
        model.add(keras.layers.Conv2D(filters=64, kernel_size=3, padding='same',
                                      input_shape=(width, height, channels), kernel_regularizer=keras.regularizers.l2(weight_decay)))
        model.add(keras.layers.Activation(cfg['ACTIVATION']))
        model.add(keras.layers.BatchNormalization())
        model.add(keras.layers.Dropout(0.3))

        model.add(keras.layers.Conv2D(filters=64, kernel_size=3, padding='same',
                                      kernel_regularizer=keras.regularizers.l2(weight_decay)))
        model.add(keras.layers.Activation(cfg['ACTIVATION']))
        model.add(keras.layers.BatchNormalization())

        model.add(keras.layers.MaxPooling2D(pool_size=2))

        model.add(keras.layers.Conv2D(filters=128, kernel_size=3, padding='same',
                                      kernel_regularizer=keras.regularizers.l2(weight_decay)))
        model.add(keras.layers.Activation(cfg['ACTIVATION']))
        model.add(keras.layers.BatchNormalization())
        model.add(keras.layers.Dropout(0.4))

        model.add(keras.layers.Conv2D(filters=128, kernel_size=3, padding='same',
                                      kernel_regularizer=keras.regularizers.l2(weight_decay)))
        model.add(keras.layers.Activation(cfg['ACTIVATION']))
        model.add(keras.layers.BatchNormalization())

        model.add(keras.layers.MaxPooling2D(pool_size=2))

        model.add(keras.layers.Conv2D(filters=256, kernel_size=3, padding='same',
                                      kernel_regularizer=keras.regularizers.l2(weight_decay)))
        model.add(keras.layers.Activation(cfg['ACTIVATION']))
        model.add(keras.layers.BatchNormalization())
        model.add(keras.layers.Dropout(0.4))

        model.add(keras.layers.Conv2D(filters=256, kernel_size=3, padding='same',
                                      kernel_regularizer=keras.regularizers.l2(weight_decay)))
        model.add(keras.layers.Activation(cfg['ACTIVATION']))
        model.add(keras.layers.BatchNormalization())
        model.add(keras.layers.Dropout(0.4))

        model.add(keras.layers.Conv2D(filters=256, kernel_size=3, padding='same',
                                      kernel_regularizer=keras.regularizers.l2(weight_decay)))
        model.add(keras.layers.Activation(cfg['ACTIVATION']))
        model.add(keras.layers.BatchNormalization())

        model.add(keras.layers.MaxPooling2D(pool_size=2))

        model.add(keras.layers.Conv2D(filters=512, kernel_size=3, padding='same',
                                      kernel_regularizer=keras.regularizers.l2(weight_decay)))
        model.add(keras.layers.Activation(cfg['ACTIVATION']))
        model.add(keras.layers.BatchNormalization())
        model.add(keras.layers.Dropout(0.4))

        model.add(keras.layers.Conv2D(filters=512, kernel_size=3, padding='same',
                                      kernel_regularizer=keras.regularizers.l2(weight_decay)))
        model.add(keras.layers.Activation(cfg['ACTIVATION']))
        model.add(keras.layers.BatchNormalization())
        model.add(keras.layers.Dropout(0.4))

        model.add(keras.layers.Conv2D(filters=512, kernel_size=3, padding='same',
                                      kernel_regularizer=keras.regularizers.l2(weight_decay)))
        model.add(keras.layers.Activation(cfg['ACTIVATION']))
        model.add(keras.layers.BatchNormalization())

        model.add(keras.layers.MaxPooling2D(pool_size=2))

        model.add(keras.layers.Conv2D(filters=512, kernel_size=3, padding='same',
                                      kernel_regularizer=keras.regularizers.l2(weight_decay)))
        model.add(keras.layers.Activation(cfg['ACTIVATION']))
        model.add(keras.layers.BatchNormalization())
        model.add(keras.layers.Dropout(0.4))

        model.add(keras.layers.Conv2D(filters=512, kernel_size=3, padding='same',
                                      kernel_regularizer=keras.regularizers.l2(weight_decay)))
        model.add(keras.layers.Activation(cfg['ACTIVATION']))
        model.add(keras.layers.BatchNormalization())
        model.add(keras.layers.Dropout(0.4))

        model.add(keras.layers.Conv2D(filters=512, kernel_size=3, padding='same',
                                      kernel_regularizer=keras.regularizers.l2(weight_decay)))
        model.add(keras.layers.Activation(cfg['ACTIVATION']))
        model.add(keras.layers.BatchNormalization())

        model.add(keras.layers.MaxPooling2D(pool_size=2))
        model.add(keras.layers.Dropout(0.5))

        model.add(keras.layers.Flatten())
        model.add(keras.layers.Dense(512, kernel_regularizer=keras.regularizers.l2(weight_decay)))
        model.add(keras.layers.Activation(cfg['ACTIVATION']))
        model.add(keras.layers.BatchNormalization())

        model.add(keras.layers.Dropout(0.5))
        model.add(keras.layers.Dense(num_classes))
        model.add(keras.layers.Activation('softmax'))

        if weights_Path is not None:
            model.load_weights(weights_Path)

        return model

class CIFAR10_4_Conv:
    @staticmethod
    def build_model(width, height, channels, cfg, num_classes, weights_Path=None):
        model = keras.Sequential()
        model.add(keras.layers.Conv2D(filters=32, kernel_size=3, activation=cfg['ACTIVATION'],padding='same', input_shape=(width, height, channels)))
        model.add(keras.layers.Conv2D(filters=32, kernel_size=3, padding='same', activation=cfg['ACTIVATION']))
        model.add(keras.layers.MaxPooling2D(pool_size=2))
        model.add(keras.layers.Dropout(0.25))

        model.add(keras.layers.Conv2D(filters=64, kernel_size=3, padding='same', activation=cfg['ACTIVATION']))
        model.add(keras.layers.Conv2D(filters=64, kernel_size=3, padding='same', activation=cfg['ACTIVATION']))
        model.add(keras.layers.Dropout(0.25))

        model.add(keras.layers.Flatten())
        model.add(keras.layers.Dense(512, activation=cfg['ACTIVATION']))
        model.add(keras.layers.Dropout(0.25))
        model.add(keras.layers.Dense(num_classes))
        model.add(keras.layers.Activation('softmax'))

        if weights_Path is not None:
            model.load_weights(weights_Path)

        return model
