from copy import deepcopy

from sklearn.model_selection import train_test_split

from utils import *
from trainers import *
from monitoring.online import OnlineStatistics
import os
from keras.layers import Activation


class NetworkBuilder(object):
    def __init__(self, model_constructor, model_path, model_name, data_name, flatten_layer, optimizer):
        self._model_constructor = model_constructor
        self.model_path = model_path
        self.model_name = model_name
        self.data_name = data_name
        self.flatten_layer = flatten_layer
        self.optimizer = optimizer

    def build(self, data, class_label_map: ClassLabelMap, other_labels, n_epochs, batch_size, statistics):
        print("Transferring model weights", self.model_name, " ", self.data_name, "- training new model")
        model_trainer = StandardTrainer()
        # map data labels to network classes, e.g., [0, 1, 4, 2] -> [0, 1, 2, 3]
        class_label_map_combined = class_label_map.copy()
        class_label_map_combined.add_labels(other_labels)
        ground_truths = class_label_map_combined.get_classes(data.ground_truths())
        # split data for training
        x_train, x_test, y_train, y_test = train_test_split(data.inputs(),
                                                            ground_truths,
                                                            test_size=0.2,
                                                            random_state=1)
        if len(x_train.shape) > 2:
            # images
            input_shape = get_image_shape(x_train)
        else:
            # sequential data
            input_shape = int(x_train.shape[1])
        # train network for fairer comparison from scratch
        #if len(class_label_map.known_labels()) == 99 and \
        #        not other_labels in class_label_map.known_labels():
        #    print("From-scratch model for {} classes".format(data.n_classes))
            #fairer_model = self._model_constructor(weights=None, classes=data.n_classes(), input_shape=input_shape)

        # construct base model
        print("Base model for {} classes".format(classes2string(class_label_map.known_labels())))
        base_model = self._model_constructor(weights=None, classes=len(class_label_map), input_shape=input_shape)
        # Presumably you would want to first load pre-trained weights.
        base_model.load_weights(self.model_path)
        # base_model.summary()

        model = Sequential()
        if not model.input_names:
            # something in the inputs did not connect
            # add input
            model.add(Input(shape=input_shape))
        cut_off = -1
        for layer in base_model.layers[:cut_off]:
            model.add(layer)
        # Freeze all layers except for the last one/Dense ones.
        for layer in model.layers[:self.flatten_layer]:
            layer.trainable = False

        # TODO: for each dataset and network should be a different output layer and compilation
        n_classes = data.n_classes()
        model.add(Dense(n_classes, activation='softmax'))
        # model.summary()
        #plot_model(
        #    model, to_file='model.png', show_shapes=False, show_layer_names=True,
        #    rankdir='TB', expand_nested=False, dpi=96
        #)

        # Recompile and train (this will only update the weights of the trainable layers).
        model.compile(optimizer=self.optimizer, loss='categorical_crossentropy', metrics=['accuracy'])
        # train model
        data_train = DataSpec(randomize=False, inputs=x_train, labels=y_train, uniform=True)
        data_test = DataSpec(randomize=False, inputs=x_test, labels=y_test, uniform=True)
        time_training_model = time()
        epochs = max(n_epochs, 30*(n_classes > 10 or self.data_name == 'CIFAR10'))
        history = model_trainer.train(model, data_train, data_test, epochs=epochs, batch_size=batch_size)
        statistics.time_training_model = time() - time_training_model
        #if len(class_label_map.known_labels()) == 99 and \
        #        not other_labels in class_label_map.known_labels():
            # train network for fairer comparison from scratch
            #temp_history = model_trainer.train(fairer_model, data_train, data_test, epochs=epochs, batch_size=batch_size)

        # store model
        all_labels = sorted(class_label_map.known_labels() + other_labels)
        trans_classes_string = classes2string(all_labels)#([min(all_labels), len(all_labels), max(all_labels)])
        model_path_split = self.model_path.split('_')
        transfer_model_path = ""
        for path in model_path_split[:-1]:
            transfer_model_path = transfer_model_path + path + "_"
        transfer_model_path = transfer_model_path + "{}.h5".format(trans_classes_string)
        fairer_model_path = transfer_model_path + "fairer_{}.h5".format(trans_classes_string)

        if n_classes > 3:
            os.remove(self.model_path)  # erase the previous model
        if n_classes > 2:
            store_model(transfer_model_path, model, history)
            #if len(class_label_map.known_labels()) == 99 and \
            #        not other_labels in class_label_map.known_labels():
                #store_model(fairer_model_path, fairer_model, temp_history)
        self.model_path = transfer_model_path

        # restore original labels in data
        labels = class_label_map_combined.get_labels(y_train, as_np_array=True)
        data_train = DataSpec(randomize=False, inputs=x_train, labels=labels)
        labels = class_label_map_combined.get_labels(y_test, as_np_array=True)
        data_test = DataSpec(randomize=False, inputs=x_test, labels=labels)
        return model, data_train, data_test

    def retrain_network(self, class_label_map, data_train, other_labels, statistics: OnlineStatistics):
        statistics.timer_online_retraining_networks.start()
        # retrain network, then continue outside the loop
        # retrain only for the classes that have enough data
        all_labels = sorted(class_label_map.known_labels() + other_labels)
        # TODO: how to split data on training and testing for retraining the network and the monitor?
        # TODO: should we include testing data into monitor training?
        data_filtered = data_train.filter_by_classes(classes=all_labels, copy=True)
        network, data_train_monitor, data_test_monitor =\
            self.build(data=data_filtered,
                       class_label_map=class_label_map,
                       other_labels=other_labels,
                       n_epochs=N_EPOCHS, batch_size=BATCH_SIZE,
                       statistics=statistics)
        statistics.timer_online_retraining_networks.stop()
        return network, data_filtered, data_train_monitor, data_test_monitor
