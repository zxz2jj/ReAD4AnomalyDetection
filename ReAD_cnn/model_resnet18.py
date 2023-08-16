from __future__ import print_function
import os
import sys
import tensorflow as tf
from keras.models import Sequential, Model
from keras.layers import Conv2D, MaxPooling2D
from keras.layers import Dense, Activation, Dropout, Flatten
from keras.layers import BatchNormalization, add, Input, AveragePooling2D
from keras.optimizers import Adam, SGD

from keras import regularizers
sys.path.append(os.getcwd())


class ResNet18Model(object):
    def __init__(self, train_class_number, train_data, train_label, test_data, test_label, model_save_path,
                 validation_data=None, validation_label=None,):
        self.train_class_number = train_class_number
        self.train_data = train_data
        self.train_label = train_label
        self.test_data = test_data
        self.test_label = test_label
        self.model_save_path = model_save_path
        self.validation_data = validation_data
        self.validation_label = validation_label

    def conv2d_bn(self, x, filters, kernel_size, weight_decay=.0, strides=(1, 1)):
        layer = Conv2D(filters=filters,
                       kernel_size=kernel_size,
                       strides=strides,
                       padding='same',
                       use_bias=False,
                       kernel_regularizer=regularizers.l2(weight_decay)
                       )(x)
        layer = BatchNormalization()(layer)
        return layer

    def conv2d_bn_relu(self, x, filters, kernel_size, weight_decay=.0, strides=(1, 1)):
        layer = self.conv2d_bn(x, filters, kernel_size, weight_decay, strides)
        layer = Activation('relu')(layer)
        return layer

    def ResidualBlock(self, x, filters, kernel_size, weight_decay, downsample=True):
        if downsample:
            # residual_x = conv2d_bn_relu(x, filters, kernel_size=1, strides=2)
            residual_x = self.conv2d_bn(x, filters, kernel_size=1, strides=2)
            stride = 2
        else:
            residual_x = x
            stride = 1
        residual = self.conv2d_bn_relu(x,
                                       filters=filters,
                                       kernel_size=kernel_size,
                                       weight_decay=weight_decay,
                                       strides=stride,)
        residual = self.conv2d_bn(residual,
                                  filters=filters,
                                  kernel_size=kernel_size,
                                  weight_decay=weight_decay,
                                  strides=1,)
        out = add([residual_x, residual])
        out = Activation('relu')(out)
        return out

    def resnet18(self, input_shape, weight_decay=1e-4, lr=1e-1):
        input = Input(shape=input_shape)
        x = input
        x = self.conv2d_bn_relu(x, filters=64, kernel_size=(3, 3), weight_decay=weight_decay, strides=(1, 1))
        # # conv 2
        x = self.ResidualBlock(x, filters=64, kernel_size=(3, 3), weight_decay=weight_decay, downsample=False)
        x = Dropout(0.4)(x)
        x = self.ResidualBlock(x, filters=64, kernel_size=(3, 3), weight_decay=weight_decay, downsample=False)
        # # conv 3
        x = self.ResidualBlock(x, filters=128, kernel_size=(3, 3), weight_decay=weight_decay, downsample=True)
        x = Dropout(0.4)(x)
        x = self.ResidualBlock(x, filters=128, kernel_size=(3, 3), weight_decay=weight_decay, downsample=False)
        # # conv 4
        x = self.ResidualBlock(x, filters=256, kernel_size=(3, 3), weight_decay=weight_decay, downsample=True)
        x = Dropout(0.4)(x)
        x = self.ResidualBlock(x, filters=256, kernel_size=(3, 3), weight_decay=weight_decay, downsample=False)
        # # conv 5
        x = self.ResidualBlock(x, filters=512, kernel_size=(3, 3), weight_decay=weight_decay, downsample=True)
        x = Dropout(0.4)(x)
        x = self.ResidualBlock(x, filters=512, kernel_size=(3, 3), weight_decay=weight_decay, downsample=False)
        x = AveragePooling2D(pool_size=(4, 4), padding='valid')(x)
        x = Flatten()(x)
        x = Dropout(0.5)(x)
        x = Dense(self.train_class_number, activation=None)(x)
        x = Activation('softmax')(x)
        model = Model(input, x, name='ResNet18')
        opt = SGD(lr=lr, momentum=0.9, nesterov=False)
        model.compile(optimizer=opt, loss='categorical_crossentropy', metrics=['accuracy'])
        model.summary()
        return model

    def lr_scheduler(self, epoch, lr):
        new_lr = lr * (0.1 ** (epoch // 50))
        print('new lr:%.2e' % new_lr)
        return new_lr

    def train(self, epochs=120):
        lr = 1e-1
        model = self.resnet18(input_shape=self.train_data.shape[1:], weight_decay=1e-4, lr=lr)
        reduce_lr = tf.keras.callbacks.LearningRateScheduler(self.lr_scheduler)
        # model.load_weights('/home/kengo/Documents/Detector_Evaluation/weights/svhn_resnet18.h5')
        model.fit(x=self.train_data, y=self.train_label, batch_size=128, epochs=epochs, callbacks=[reduce_lr],
                  validation_data=(self.test_data, self.test_label))
        model.save(self.model_save_path+'tf_model.h5')
        print("save path:", self.model_save_path)

    def show_model(self):
        model = tf.keras.models.load_model(self.model_save_path+'tf_model.h5')
        # model.summary()
        print("train dataset:")
        print(self.train_data.shape, self.train_label.shape)
        model.evaluate(self.train_data, self.train_label, verbose=2)

        print("test dataset:")
        print(self.test_data.shape, self.test_label.shape)
        model.evaluate(self.test_data, self.test_label, verbose=2)


