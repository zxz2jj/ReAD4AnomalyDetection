import tensorflow as tf
import os

os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"


class VGGModel(object):
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

    def create_model(self):
        model = tf.keras.Sequential()
        weight_decay = 0.0005

        model.add(tf.keras.layers.Conv2D(64, (3, 3), padding='same',
                                         input_shape=[32, 32, 3], kernel_regularizer=tf.keras.regularizers.l2(weight_decay)))
        model.add(tf.keras.layers.Activation('relu'))
        model.add(tf.keras.layers.BatchNormalization())
        model.add(tf.keras.layers.Dropout(0.3))

        model.add(tf.keras.layers.Conv2D(64, (3, 3), padding='same', kernel_regularizer=tf.keras.regularizers.l2(weight_decay)))
        model.add(tf.keras.layers.Activation('relu'))
        model.add(tf.keras.layers.BatchNormalization())

        model.add(tf.keras.layers.MaxPooling2D(pool_size=(2, 2)))

        model.add(tf.keras.layers.Conv2D(128, (3, 3), padding='same', kernel_regularizer=tf.keras.regularizers.l2(weight_decay)))
        model.add(tf.keras.layers.Activation('relu'))
        model.add(tf.keras.layers.BatchNormalization())
        model.add(tf.keras.layers.Dropout(0.4))

        model.add(tf.keras.layers.Conv2D(128, (3, 3), padding='same', kernel_regularizer=tf.keras.regularizers.l2(weight_decay)))
        model.add(tf.keras.layers.Activation('relu'))
        model.add(tf.keras.layers.BatchNormalization())

        model.add(tf.keras.layers.MaxPooling2D(pool_size=(2, 2)))

        model.add(tf.keras.layers.Conv2D(256, (3, 3), padding='same', kernel_regularizer=tf.keras.regularizers.l2(weight_decay)))
        model.add(tf.keras.layers.Activation('relu'))
        model.add(tf.keras.layers.BatchNormalization())
        model.add(tf.keras.layers.Dropout(0.4))

        model.add(tf.keras.layers.Conv2D(256, (3, 3), padding='same', kernel_regularizer=tf.keras.regularizers.l2(weight_decay)))
        model.add(tf.keras.layers.Activation('relu'))
        model.add(tf.keras.layers.BatchNormalization())
        model.add(tf.keras.layers.Dropout(0.4))

        model.add(tf.keras.layers.Conv2D(256, (3, 3), padding='same', kernel_regularizer=tf.keras.regularizers.l2(weight_decay)))
        model.add(tf.keras.layers.Activation('relu'))
        model.add(tf.keras.layers.BatchNormalization())

        model.add(tf.keras.layers.MaxPooling2D(pool_size=(2, 2)))

        model.add(tf.keras.layers.Conv2D(512, (3, 3), padding='same', kernel_regularizer=tf.keras.regularizers.l2(weight_decay)))
        model.add(tf.keras.layers.Activation('relu'))
        model.add(tf.keras.layers.BatchNormalization())
        model.add(tf.keras.layers.Dropout(0.4))

        model.add(tf.keras.layers.Conv2D(512, (3, 3), padding='same', kernel_regularizer=tf.keras.regularizers.l2(weight_decay)))
        model.add(tf.keras.layers.Activation('relu'))
        model.add(tf.keras.layers.BatchNormalization())
        model.add(tf.keras.layers.Dropout(0.4))

        model.add(tf.keras.layers.Conv2D(512, (3, 3), padding='same', kernel_regularizer=tf.keras.regularizers.l2(weight_decay)))
        model.add(tf.keras.layers.Activation('relu'))
        model.add(tf.keras.layers.BatchNormalization())

        model.add(tf.keras.layers.MaxPooling2D(pool_size=(2, 2)))

        model.add(tf.keras.layers.Conv2D(512, (3, 3), padding='same', kernel_regularizer=tf.keras.regularizers.l2(weight_decay)))
        model.add(tf.keras.layers.Activation('relu'))
        model.add(tf.keras.layers.BatchNormalization())
        model.add(tf.keras.layers.Dropout(0.4))

        model.add(tf.keras.layers.Conv2D(512, (3, 3), padding='same', kernel_regularizer=tf.keras.regularizers.l2(weight_decay)))
        model.add(tf.keras.layers.Activation('relu'))
        model.add(tf.keras.layers.BatchNormalization())
        model.add(tf.keras.layers.Dropout(0.4))

        model.add(tf.keras.layers.Conv2D(512, (3, 3), padding='same', kernel_regularizer=tf.keras.regularizers.l2(weight_decay)))
        # -11
        model.add(tf.keras.layers.Activation('relu'))
        model.add(tf.keras.layers.BatchNormalization())

        model.add(tf.keras.layers.MaxPooling2D(pool_size=(2, 2)))
        model.add(tf.keras.layers.Dropout(0.5))

        # -7
        model.add(tf.keras.layers.Flatten())
        model.add(tf.keras.layers.Dense(512, kernel_regularizer=tf.keras.regularizers.l2(weight_decay)))
        # -5
        model.add(tf.keras.layers.Activation('relu'))
        model.add(tf.keras.layers.BatchNormalization())

        model.add(tf.keras.layers.Dropout(0.5))
        model.add(tf.keras.layers.Dense(self.train_class_number, activation=None))
        model.add(tf.keras.layers.Activation('softmax'))
        sgd = tf.keras.optimizers.SGD(lr=0.1, decay=1e-6, momentum=0.9, nesterov=True)
        model.compile(loss='categorical_crossentropy', optimizer=sgd, metrics=['accuracy'])
        return model

    def train(self):
        model = self.create_model()
        # model.fit(self.train_data, self.train_label, epochs=10, verbose=2, shuffle=True, validation_split=0.2)
        model.fit(self.train_data, self.train_label, epochs=120, verbose=1,
                  validation_data=(self.test_data, self.test_label))
        model.save(self.model_save_path+'tf_model.h5')
        print("save path:", self.model_save_path+'tf_model.h5')

    def show_model(self):
        model = tf.keras.models.load_model(self.model_save_path+'tf_model.h5')
        model.summary()
        print("train dataset:")
        print(self.train_data.shape, self.train_label.shape)
        model.evaluate(self.train_data, self.train_label, verbose=2)

        print("test dataset:")
        print(self.test_data.shape, self.test_label.shape)
        model.evaluate(self.test_data, self.test_label, verbose=2)
