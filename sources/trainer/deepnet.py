from sklearn.svm import SVC
from numpy import ndarray
from ..ifood.convert import img2ndarray
from ..ifood.dao import IfoodDataDao
import random
import numpy as np


class Deepnet:
    def __init__(self, batch_size, start_image_size, epoch, random_state=0):
        self.random_state = random_state
        self.start_image_size = start_image_size
        self.batch_size = batch_size
        self.epoch = epoch

    def fit(self, train_dao: IfoodDataDao, label: ndarray):
        random.seed(self.random_state)
        train_size = train_dao.get_index_size()
        train_indices = [i for i in range(train_size)]
        random.shuffle(train_indices)
        for _ in range(self.epoch):
            for start in range(0, train_size, self.batch_size):
                batch_indices = train_indices[start: start + self.batch_size]
                batch_img_arrays = [
                    img2ndarray(train_dao.get_image(i), self.start_image_size) for i in
                    batch_indices
                ]
                batch_img_arrays = np.array(batch_img_arrays)
                self._fit(batch_img_arrays, label)

    def _fit(self, X, y):
        pass
