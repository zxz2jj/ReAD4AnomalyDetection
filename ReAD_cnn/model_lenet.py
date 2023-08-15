import tensorflow as tf
from keras.layers import Conv2D, Activation, Dropout, MaxPooling2D, Flatten, Dense


class LeNetModel(object):
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

        model.add(Conv2D(filters=40, kernel_size=(5, 5), strides=(1, 1), input_shape=self.train_data.shape[1:],
                         padding='same', data_format='channels_last', activation='relu', kernel_initializer='uniform'))
        model.add(Conv2D(filters=20, kernel_size=(5, 5), activation='relu'))
        model.add(MaxPooling2D((2, 2)))
        model.add(Dropout(0.25))

        model.add(Flatten(data_format='channels_last'))
        model.add(Dense(320, activation='relu'))
        model.add(Dropout(0.5))
        model.add(Dense(160, activation='relu'))
        model.add(Dropout(0.5))
        model.add(Dense(80, activation='relu'))
        model.add(Dropout(0.5))
        model.add(Dense(40, activation='relu'))
        model.add(Dropout(0.5))
        model.add(Dense(self.train_class_number, activation=None))
        model.add(Activation('softmax'))

        model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])
        return model

    def train(self, epochs=10):
        model = self.create_model()
        model.fit(self.train_data, self.train_label, epochs=epochs, verbose=2)
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





