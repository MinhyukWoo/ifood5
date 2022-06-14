from abc import ABCMeta
from abc import abstractmethod
from enum import Enum
import random

import numpy as np
from numpy import ndarray
from PIL.Image import Image


class IfoodDaoType(Enum):
    TRAIN = 0,
    VALIDATION = 1


class Dataset(metaclass=ABCMeta):
    def __init__(self, batch_size, image_size, shuffle):
        self.batch_size = batch_size
        self.image_size = image_size
        self.shuffle = shuffle
        self.__indices = self._get_indices()

    def __iter__(self):
        self.__i_start = 0
        self.__i_end = len(self.__indices)
        if self.shuffle is True:
            random.shuffle(self.__indices)
        return self

    def __next__(self):
        if self.__i_end > self.__i_start:
            batch_indices = self.__indices[
                            self.__i_start: self.__i_start + self.batch_size
                            ]
            labels = self._get_labels(batch_indices)
            image_arrays = self.__img_to_ndarray(
                self._get_images(batch_indices))
            self.__i_start += self.batch_size
            return np.array(image_arrays), np.array(labels)
        else:
            raise StopIteration

    def __img_to_ndarray(self, images: list[Image]) -> list[ndarray]:
        return [np.asarray(
            image.convert("RGB").resize(self.image_size),
            dtype=float).transpose((2, 0, 1))
                for image in images]

    @abstractmethod
    def _get_indices(self) -> list[int]:
        pass

    @abstractmethod
    def _get_labels(self, indices: list[int]) -> list[int]:
        pass

    @abstractmethod
    def _get_images(self, indices: list[int]) -> list[Image]:
        pass


class IfoodDataset(Dataset):
    def __init__(self, batch_size, image_size, shuffle, dao: IfoodDaoType):
        if dao is IfoodDaoType.TRAIN:
            from ..ifood.dao import IfoodTrainDao
            self._dao = IfoodTrainDao()
        elif dao is IfoodDaoType.VALIDATION:
            from ..ifood.dao import IfoodValidationDao
            self._dao = IfoodValidationDao()
        else:
            raise ValueError
        self._image_names = np.array(self._dao.get_img_names(), dtype=str)
        super().__init__(batch_size, image_size, shuffle)

    def _get_labels(self, indices: list[int]) -> list[int]:
        return [self._dao.get_label(self._image_names[i])
                for i in indices]

    def _get_images(self, indices: list[int]) -> list[Image]:
        return [self._dao.get_image(self._image_names[i])
                for i in indices]

    def _get_indices(self) -> list[int]:
        return [i for i in range(len(self._image_names))]
