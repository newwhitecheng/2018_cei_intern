import tensorflow as tf
import keras
from CNN_Model import LeNet

import numpy as np

import utils
import loggingreporter

import keras.backend.tensorflow_backend as KTF

config = tf.ConfigProto()
config.gpu_options.allow_growth=True   #不全部占满显存, 按需分配
session = tf.Session(config=config)

# 设置session
KTF.set_session(session)

cfg = {}
cfg['SGD_BATCHSIZE']    = 128
cfg['SGD_LEARNINGRATE'] = 0.0004
cfg['NUM_EPOCHS']       = 400
cfg['FULL_MI']          = False

cfg['ACTIVATION']       = 'tanh'

cfg['LAYER_DIMS']       = [6, 16, 120]
ARCH_NAME = '-'.join(map(str, cfg['LAYER_DIMS']))
cfg['SAVE_DIR'] = 'rawdata/LeNet5_MNIST' + cfg['ACTIVATION'] + '_' + ARCH_NAME

NUM_CLASSES = 10

trn, tst = utils.load_MNIST_data()

model = LeNet.LeNet.build_model(28, 28, 1, cfg, 10)

optimizer = keras.optimizers.TFOptimizer(tf.train.AdamOptimizer(learning_rate=cfg['SGD_LEARNINGRATE']))
model.compile(loss='categorical_crossentropy', optimizer=optimizer, metrics=['accuracy'])

model.summary()

def do_report(epoch):
    if epoch < 20:
        return True
    elif epoch < 100:
        return (epoch % 5 == 0)
    elif epoch < 2000:
        return (epoch % 20 == 0)
    else:
        return (epoch % 100 == 0)

reporter = loggingreporter.LoggingReporter(cfg=cfg, trn=trn, tst=tst, do_save_func=do_report)

model.fit(x=trn.X, y=trn.Y, epochs=cfg['NUM_EPOCHS'],batch_size=cfg['SGD_BATCHSIZE'], validation_data=(tst.X, tst.Y),
          verbose=2, callbacks=[reporter, ])

model.save_weights("weights.h5df")
