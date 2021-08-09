from .Trainer import Trainer
from .DataGenerator import DataGenerator
from utils import *
from sklearn.model_selection import train_test_split, StratifiedShuffleSplit
from keras.callbacks import EarlyStopping, ReduceLROnPlateau
from keras.preprocessing.image import ImageDataGenerator
from keras.utils import np_utils


class StandardTrainer(Trainer):
    def __init__(self):
        pass

    def __str__(self):
        return "StandardTrainer"

    def train(self, model, data_train: DataSpec, data_test: DataSpec, epochs: int, batch_size: int):
        if data_train.inputs().shape[1] == 224:
            # parameters for callback functions
            es_patience = 10
            rlrop_patience = 5
            decay_rate = 0.5
            batch_size = 32
            es = EarlyStopping(monitor='val_loss', mode='min', patience=es_patience, restore_best_weights=True,
                               verbose=1)
            rlrop = ReduceLROnPlateau(monitor='val_loss', mode='min', patience=rlrop_patience,
                                      factor=decay_rate, min_lr=1e-6, verbose=1)

            train_generator = DataGenerator(data_train.inputs(), data_train.ground_truths_as_classes(),
                                            augment=True,
                                            n_classes=len(data_train.classes),
                                            batch_size=batch_size)
            validation_generator = DataGenerator(data_test.inputs(), data_test.ground_truths_as_classes(),
                                                 augment=False,
                                                 n_classes=len(data_test.classes),
                                                 batch_size=batch_size)

            history = model.fit(
                train_generator,
                steps_per_epoch=data_train.inputs().shape[0] // batch_size,
                epochs=epochs,
                validation_data=validation_generator,
                validation_steps=data_test.inputs().shape[0] // batch_size,
                verbose=1,
                use_multiprocessing=True,
                workers=4,
                callbacks=[es, rlrop]
            )
        else:
            data_train.shuffle()
            data_test.shuffle()

            history = model.fit(data_train.inputs(),
                                to_categorical(data_train.ground_truths_as_classes()),
                                validation_data=(data_test.inputs(),
                                                 to_categorical(data_test.ground_truths_as_classes())),
                                #validation_split=0.1,
                                epochs=epochs,
                                batch_size=batch_size, verbose=1)

        if VERBOSE_MODEL_TRAINING:
            print("score:", model.evaluate(data_test.inputs(), data_test.categoricals(), batch_size=batch_size))
            print(model.summary())

        return history

        # input_prediction = x_test_final[0:1, :]
        # prediction = model.predict(input_prediction, 1, 1)
        # predicted_class = np.argmax(prediction)
        # print("testing prediction:\n input: ", input_prediction,
        #       "\n output: ", prediction,
        #       "\n class (in [0,", n_classes, "]): ", predicted_class)

        # Testing
        # layer_outs = get_watched_values(model, x_train[0:1, :])
        # print(layer_outs)
