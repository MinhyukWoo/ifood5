import csv
import os.path
from PIL import Image
from abc import ABCMeta
from abc import abstractmethod

_project_path = os.path.join(__file__, '..')
_annot_path = os.path.join(_project_path, 'dataset', 'annot')


class IfoodClassDao:
    def __init__(self):
        self.database = dict()
        with open(os.path.join(_annot_path, 'mini_class_list.csv'), 'r') as file:
            for row in csv.reader(file):
                self.database[int(row[0])] = row[1]

    def get_name(self, index: int) -> str:
        return self.database.get(index)


class IfoodDataDao(metaclass=ABCMeta):
    @abstractmethod
    def _get_database(self) -> dict[str, int]:
        pass

    @abstractmethod
    def _get_img_base_path(self):
        pass

    def get_img_names(self) -> tuple[str]:
        return tuple(
            name for name in self._get_database().keys()
        )

    def get_image(self, image_name: str) -> Image.Image:
        if image_name in self._get_database():
            return Image.open(
                os.path.join(self._get_img_base_path(), image_name)
            )
        else:
            raise ValueError

    def get_label(self, image_name: str) -> int:
        if image_name in self._get_database():
            return self._get_database()[image_name]
        else:
            raise ValueError


class IfoodTrainDao(IfoodDataDao):
    def __init__(self):
        print(os.path.abspath(__file__))
        with open(os.path.join(_annot_path, 'mini_train_info.csv'), 'r') as f:
            self.database = dict(
                (row[0], int(row[1])) for row in csv.reader(f)
            )
        self.img_base_path = os.path.join(
            _project_path, 'dataset', 'mini_train_set'
        )

    def _get_database(self) -> dict[str, int]:
        return self.database

    def _get_img_base_path(self):
        return self.img_base_path


class IfoodValidationDao(IfoodDataDao):
    def __init__(self):
        with open(os.path.join(_annot_path, 'mini_val_info.csv'), 'r') as f:
            self.database = dict(
                (row[0], int(row[1])) for row in csv.reader(f)
            )
        self.img_base_path = os.path.join(
            _project_path, 'dataset', 'mini_val_set'
        )

    def _get_database(self) -> dict[str, int]:
        return self.database

    def _get_img_base_path(self):
        return self.img_base_path


def __test():
    IfoodClassDao()
    train_dao = IfoodTrainDao()
    names = train_dao.get_img_names()
    image = train_dao.get_image(names[0])
    image.show()
    n = train_dao.get_label(names[0])
    print(IfoodClassDao().get_name(n))
    pass


if __name__ == '__main__':
    __test()
