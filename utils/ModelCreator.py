from os.path import isfile
from tensorflow.python.keras.models import load_model as tf_load_model
from tensorflow.python.keras.models import model_from_json
from tensorflow.python.keras.models import Sequential, Input, Model
from tensorflow.python.keras.layers import Dense, Flatten
from time import time

from utils import *
from utils.Options import MODEL_INSTANCE_PATH
from models.ModelLoader import get_model_loader
import numpy as np

from utils import get_image_shape, DataSpec, classes2string


def get_model(model_name: str, model_path, data_train: DataSpec = None, data_test: DataSpec = None,
              class_label_map=None, model_trainer=None, n_epochs=None, batch_size=None, transfer=False,
              transfer_model_name=None, statistics: Statistics = None):
    model_constructor = get_model_loader(model_name, model_path)
    n_classes = -1 if class_label_map is None else len(class_label_map)

    loaded = False
    if isfile(MODEL_INSTANCE_PATH + model_path):
        try:
            model, history = load_model(MODEL_INSTANCE_PATH + model_path)
            loaded = True
        except:
            pass

    if len(data_train.inputs().shape) > 2:
        # images
        input_shape = get_image_shape(data_train.inputs())
    else:
        # sequential data
        input_shape = int(data_train.inputs().shape[1])

    if not loaded and not transfer:
        print("Could not load model", MODEL_INSTANCE_PATH + model_path, "- trying to copy an existing model")
        # construct raw model
        model = model_constructor(weights=None, classes=n_classes, input_shape=input_shape)

        all_labels = sorted(class_label_map.known_labels())
        classes_string = classes2string(all_labels)
        model_path_split = model_path.split('_')
        transfer_model_path = model_path
        no_transfer_model_path = ""
        for path in model_path_split[:-1]:
            if path != "transfer":
                no_transfer_model_path = no_transfer_model_path + path + "_"
        no_transfer_model_path = no_transfer_model_path + "{}.h5".format(classes_string)
        if isfile(MODEL_INSTANCE_PATH + no_transfer_model_path):
            # load pre-trained weights.
            model, history = load_model(MODEL_INSTANCE_PATH + no_transfer_model_path)
            loaded = True
            # store as the starting model for transfer
            store_model(MODEL_INSTANCE_PATH + transfer_model_path, model, history)
        else:
            print("Could not load model", MODEL_INSTANCE_PATH + model_path, "- training a new model")

            # train model
            time_training_model = time()
            history = model_trainer.train(model, data_train, data_test, epochs=n_epochs, batch_size=batch_size)
            statistics.time_training_model = time() - time_training_model

            # store model
            store_model(MODEL_INSTANCE_PATH + model_path, model, history)

    elif loaded and transfer:
        print("Transferring model weights", MODEL_INSTANCE_PATH + model_path, "- training new model")
        # construct base model
        base_model = model_constructor(weights=None, classes=n_classes, input_shape=input_shape)
        # Presumably you would want to first load pre-trained weights.
        base_model.load_weights(MODEL_INSTANCE_PATH + model_path)

        # Keep a copy of the weights of layer1 for later reference
        # initial_layer1_weights_values = base_model.layers[1].get_weights()

        model = Sequential()
        model.add(Input(shape=input_shape))
        for layer in base_model.layers[:-1]:
            model.add(layer)
        # Freeze all layers except the last one.
        for layer in model.layers[:-1]:
            layer.trainable = False
        # model.add(Flatten())
        # model.add(Dense(40, activation='relu'))
        # TODO: for each dataset and network should be a different output layer and compilation
        n_classes_new = n_classes + 1  # TODO CS: in general you want to add several new classes
        model.add(Dense(n_classes_new, activation='softmax'))#, activation='sigmoid', kernel_initializer='random_normal'))
        model.summary()

        # Recompile and train (this will only update the weights of the last layer).
        model.compile(optimizer='adam', loss='sparse_categorical_crossentropy', metrics=['accuracy'])
        # train model
        time_training_model = time()
        history = model_trainer.train(model, data_train, data_test, epochs=n_epochs, batch_size=batch_size)
        statistics.time_training_model = time() - time_training_model

        # Check that the weights of layer1 have not changed during training
        # final_layer1_weights_values = base_model.layers[1].get_weights()
        # np.testing.assert_allclose(
        #     initial_layer1_weights_values[0], final_layer1_weights_values[0]
        # )
        # np.testing.assert_allclose(
        #     initial_layer1_weights_values[1], final_layer1_weights_values[1]
        # )

        # store model
        transfer_model_name = "CNY19id1_MNIST_transfer_0-3.h5"
        store_model(MODEL_INSTANCE_PATH + transfer_model_name, model, history)

        # re-load the transferred model
        # because keras is silly and does not store the graph
        # unless one stores the model
        #model, history = load_model(model_path)

        # Check is the layer output for known classes changed during transfer
        #layer = -2
        #layer_output_base, _ = Helpers.obtain_predictions(model=base_model, data=data_train, layers=[layer])
        #layer_output_transferred, _ = Helpers.obtain_predictions(model=model, data=data_train, layers=[layer])
        #print(np.equal(layer_output_base[layer], layer_output_transferred[layer]))

    return model, history


def load_model(model_path):
    if "Doom" in model_path:
        # load json and create model
        json_file = open(MODEL_INSTANCE_PATH + 'amodel_BasicDoom_0-2.json', 'r')
        loaded_model_json = json_file.read()
        json_file.close()
        model = model_from_json(loaded_model_json)
        # load weights into new model
        model.load_weights(model_path)
        model.compile(optimizer='adam', loss='mse')
    else:
        model = tf_load_model(model_path)
    print("Loaded model from", model_path)
    history = None  # TODO store/load history?
    return model, history


def store_model(model_path, model, history):
    print("Storing model to", model_path)
    model.save(model_path)
    # TODO store/load history?
