import csv
import os.path
from typing import Tuple, List
from PIL import Image
from enum import Enum
project_path = os.path.join(__file__, '..')
annot_path = os.path.join(project_path, 'dataset', 'annot')


class IfoodClassDao:
    def __init__(self):
        self.database = dict()
        with open(os.path.join(annot_path, 'mini_class_list.csv'), 'r') as file:
            for row in csv.reader(file):
                self.database[int(row[0])] = row[1]

    def get_name(self, index: int) -> str:
        return self.database.get(index)


class IfoodDataDao:
    def _get_database(self) -> list[list[str]]:
        raise NotImplemented

    def _get_img_base_path(self):
        raise NotImplemented

    def get_img_names(self) -> tuple[str]:
        return tuple(
            row[self.Cols.IMG_NAME.value] for row in self._get_database()
        )

    def get_index_size(self) -> int:
        return len(self._get_database())

    def get_image(self, index: int | str) -> Image.Image:
        if type(index) is int:
            img_name = self._get_database()[index][self.Cols.IMG_NAME.value]
        elif type(index) is str:
            img_name = index
        else:
            raise AttributeError
        return Image.open(os.path.join(self._get_img_base_path(), img_name))

    class Cols(Enum):
        IMG_NAME = 0
        LABEL = 1


class IfoodTrainDao(IfoodDataDao):
    def __init__(self):
        print(os.path.abspath(__file__))
        with open(os.path.join(annot_path, 'mini_train_info.csv'), 'r') as file:
            self.database = [row for row in csv.reader(file)]
        self.img_base_path = os.path.join(project_path, 'dataset', 'mini_train_set')

    def _get_database(self) -> list[list[str]]:
        return self.database

    def _get_img_base_path(self):
        return self.img_base_path


class IfoodValidationDao(IfoodDataDao):
    def __init__(self):
        with open(os.path.join(annot_path, 'mini_val_info.csv'), 'r') as file:
            self.database = [row for row in csv.reader(file)]
        self.img_base_path = os.path.join(project_path, 'dataset', 'mini_val_set')

    def _get_database(self) -> list[list[str]]:
        return self.database

    def _get_img_base_path(self):
        return self.img_base_path


def __test():
    IfoodClassDao()
    trainDao = IfoodTrainDao()
    image = trainDao.get_image(0)
    print(image)


if __name__ == '__main__':
    __test()

