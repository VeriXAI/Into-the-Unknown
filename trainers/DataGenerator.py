import keras
import numpy as np
from keras.utils import to_categorical
import cv2
import albumentations as albu

# parameters for data
height = 32#224
width = 32#224
channels = 3
input_shape = (height, width, channels)
n_classes = 5
batch_size = 32


class DataGenerator(keras.utils.Sequence):
    'Generates data for keras'

    def __init__(self, images, labels=None, mode='fit', batch_size=batch_size,
                 dim=(height, width), channels=channels, n_classes=n_classes,
                 shuffle=True, augment=False):
        self.images = images
        self.labels = labels
        self.mode = mode
        self.batch_size = batch_size
        self.dim = dim
        self.channels = channels
        self.n_classes = n_classes
        self.shuffle = shuffle
        self.augment = augment

        self.on_epoch_end()

    def np_resize(self, img, shape):
        return cv2.resize(img, (shape[1], shape[0]), interpolation=cv2.INTER_CUBIC)

    def on_epoch_end(self):
        'Updates indexes after each epoch'
        self.indexes = np.arange(self.images.shape[0])
        if self.shuffle == True:
            np.random.shuffle(self.indexes)

    def __len__(self):
        'Denotes the number of batches per epoch'
        return int(np.floor(len(self.images) / self.batch_size))

    def __getitem__(self, index):
        'Generate one batch of data'
        batch_indexes = self.indexes[index * self.batch_size:(index + 1) * self.batch_size]

        # =========================================================== #
        # Generate mini-batch of X
        # =========================================================== #
        X = np.empty((self.batch_size, *self.dim, self.channels))
        for i, ID in enumerate(batch_indexes):
            # Generate a preprocessed image
            img = self.images[ID]
            img = img.astype(np.float32) / 255.
            img = self.np_resize(img, self.dim)
            X[i] = img

        # =========================================================== #
        # Generate mini-batch of y
        # =========================================================== #
        if self.mode == 'fit':
            y = self.labels[batch_indexes]
            y = to_categorical(y, n_classes)
            '''
            y = np.zeros((self.batch_size, self.n_classes), dtype = np.uint8)
            for i, ID in enumerate(batch_indexes):
                # one hot encoded label
                y[i, self.labels[ID]] = 1
            '''
            # Augmentation should only be implemented in the training part.
            if self.augment == True:
                X = self.__augment_batch(X)

            return X, y

        elif self.mode == 'predict':
            return X

        else:
            raise AttributeError('The mode parameters should be set to "fit" or "predict"')

    def __random_transform(self, img):
        composition = albu.Compose([albu.HorizontalFlip(p=0.5),
                                    albu.VerticalFlip(p=0.5),
                                    albu.GridDistortion(p=0.2),
                                    albu.ElasticTransform(p=0.2)])

        return composition(image=img)['image']

    def __augment_batch(self, img_batch):
        for i in range(img_batch.shape[0]):
            img_batch[i] = self.__random_transform(img_batch[i])

        return img_batch