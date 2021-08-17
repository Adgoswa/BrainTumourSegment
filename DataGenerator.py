import numpy as np
import tensorflow.keras

class DataGenerator(tensorflow.keras.utils.Sequence):
    'Generates data for Keras'
    def __init__(self, list_IDs, image_path, label_path,
                 to_fit=True, batch_size=2, dim=(128,128),
                 n_channels=4, n_classes=4, shuffle=False):
        """Initialization
        :param list_IDs: list of all 'label' ids to use in the generator
        :param image_path: path to images location
        :param label_path: path to labels or ground truth .npy files
        :param to_fit: True to return X and y, False to return X only
        :param batch_size: batch size at each iteration
        :param dim: tuple indicating image dimension
        :param n_channels: number of image channels
        :param n_classes: number of output masks
        :param shuffle: True to shuffle label indexes after every epoch
        """
        self.list_IDs = list_IDs
        self.image_path = image_path
        self.label_path = label_path
        self.to_fit = to_fit
        self.batch_size = batch_size
        self.dim = dim
        self.n_channels = n_channels
        self.n_classes = n_classes
        self.shuffle = shuffle
        self.on_epoch_end()
    
    def __len__(self):
        'Denotes the number of batches per epoch'
        return int(np.floor(len(self.list_IDs) / self.batch_size))

    def on_epoch_end(self):
        'Updates indexes after each epoch'
        self.indexes = np.arange(len(self.list_IDs))
        if self.shuffle == True:
            np.random.shuffle(self.indexes)

    def __getitem__(self, index):
        'Generate one batch of data'
        # Generate indexes of the batch
        indexes = self.indexes[index * self.batch_size:(index + 1) * self.batch_size]

        # Find list of IDs
        list_IDs_temp = [self.list_IDs[k] for k in indexes]
        #print(list_IDs_temp)
        X = self._generate_X(list_IDs_temp)
        #print('X shape', X.shape)
        y = self._generate_y(list_IDs_temp)
        #print('Y shape', y.shape)
        return X, y

    def _generate_X(self, list_IDs_temp):
        # Initialization
        X = np.empty((self.batch_size, 90, *self.dim, self.n_channels))
        for i, ID in enumerate(list_IDs_temp):
            X[i,] = np.load('./SplitDataSet/Data_4/x_' + str(ID) + '_img.npy')
        X = X.reshape([-1,128,128,4])
        return X
    
    def _generate_y(self, list_IDs_temp):
        # Initialization
        y = np.empty((self.batch_size, 90, *self.dim, self.n_channels))
        for i, ID in enumerate(list_IDs_temp):
            y[i,] = np.load('./SplitDataSet/Label_4/y_' + str(ID) + '_gt.npy')
        y = y.reshape([-1,128,128,4])
        return y
