import csv
import os
from PIL import Image
import math
import numpy as np
from tqdm import tqdm


class DatasetDAO:
    def __init__(self, dataset_path, csv_path):
        self.all_img_paths = []
        self.all_labels = []

        with open(csv_path, "r") as fr:
            csv_reader = csv.reader(fr)
            for row in csv_reader:
                img_path = os.path.join(dataset_path, row[0])
                self.all_img_paths.append(img_path)
                self.all_labels.append(int(row[1]))

        idx = np.arange(len(self.all_img_paths))
        np.random.shuffle(idx)
        self.all_img_paths = np.array(self.all_img_paths)[idx]
        self.all_labels = np.array(self.all_labels)[idx]


    def get_batch_dataset(self, index, batch_size):
        batch_imgs = []

        last_index = (index+1)*batch_size
        if last_index >= len(self.all_img_paths):
            last_index = len(self.all_img_paths)

        for path in self.all_img_paths[index*batch_size:last_index]:
            image = Image.open(path)
            image = image.convert("RGB")
            image = image.resize((64, 64))
            image = np.asarray(image, dtype=np.float32)
            image = image.transpose(2, 0, 1)
            # img /= 255.0

            if image is not None:
                batch_imgs.append(image)

        return np.array(batch_imgs), np.array(self.all_labels[index*batch_size:last_index])
