import random

from ifood.dao import *
from ifood.convert import img2ndarray

import numpy as np


def main():
    train_dao = IfoodTrainDao()
    train_size = train_dao.get_index_size()


if __name__ == '__main__':
    main()
