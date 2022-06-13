# coding: utf-8
import sys, os
sys.path.append(os.pardir)  # 부모 디렉터리의 파일을 가져올 수 있도록 설정
import numpy as np
from sklearn.metrics import confusion_matrix, classification_report
import math
from tqdm import tqdm
import time
from .optimizer import *

# class_list = ["macaron", "dumpling", "couscous", "orzo", "gnocchi", "macaroni_and_cheese", "beet_salad",
#               "compote", "lasagna", "crab_food", "coffee_cake", "enchilada", "ceviche", "casserole",
#               "churro", "burrito", "barbecued_wing", "huitre", "chiffon_cake", "crumb_cake"]
class_list = ["macaron", "crab_food", "casserole", "churro", "huitre"]

class Trainer:
    """신경망 훈련을 대신 해주는 클래스
    """
    def __init__(self, network, train_dataset, test_dataset,
                 epochs=20, mini_batch_size=100,
                 optimizer='SGD', optimizer_param={'lr':0.01}, verbose=False):
        self.network = network
        self.train_dataset = train_dataset
        self.test_dataset = test_dataset
        self.train_num_iter = math.ceil(len(train_dataset.all_img_paths) / mini_batch_size)
        self.test_num_iter = math.ceil(len(test_dataset.all_img_paths) / mini_batch_size)
        self.train_dataset_size = len(train_dataset.all_img_paths)
        self.verbose = verbose
        self.epochs = epochs
        self.batch_size = mini_batch_size

        # optimzer
        optimizer_class_dict = {'sgd':SGD, 'momentum':Momentum, 'nesterov':Nesterov,
                                'adagrad':AdaGrad, 'rmsprpo':RMSprop, 'adam':Adam}
        self.optimizer = optimizer_class_dict[optimizer.lower()](**optimizer_param)
        
        self.current_iter = 0
        self.current_epoch = 0
        
        self.train_loss_list = []
        self.train_acc_list = []
        self.test_acc_list = []
        self.train_acc_cnt = 0
    def train_step(self, imgs, labels):
        x_batch = imgs
        t_batch = labels
        
        grads, loss, y = self.network.gradient(x_batch, t_batch)
        self.optimizer.update(self.network.params, grads)
        if t_batch.ndim != 1: t_batch = np.argmax(t_batch, axis=1)
        y = np.argmax(y, axis=1)
        self.train_acc_cnt += np.sum(y == t_batch)
        self.train_loss_list.append(loss)

        if self.verbose: print("train loss:" + str(loss))


    def train(self):
        for i in range(self.epochs):
            print(f"===== epoch {i+1} =====")
            self.train_acc_cnt = 0.0
            progress_bar = tqdm(range(self.train_num_iter))
            for j in progress_bar:
                imgs, labels = self.train_dataset.get_batch_dataset(j, self.batch_size)
                self.train_step(imgs, labels)

                now = time.localtime()
                current_time = "%04d/%02d/%02d %02d:%02d:%02d" % (
                    now.tm_year, now.tm_mon, now.tm_mday, now.tm_hour, now.tm_min, now.tm_sec)

                progress_bar.set_description(
                    '[{}] Epoch: {}/{}. Iteration: {}/{}. current batch loss: {:.4f}. '.format(
                        current_time, i + 1, self.epochs, j + 1, self.train_num_iter, self.train_loss_list[-1]))

            train_acc = self.train_acc_cnt / self.train_dataset.dataset_size

            self.train_acc_list.append(train_acc)

            total_val_cnt = 0.0
            y_list = []
            for k in range(self.test_num_iter):
                imgs, labels = self.test_dataset.get_batch_dataset(k)
                cnt, y = self.network.inference(imgs, labels)
                total_val_cnt += cnt
                y_list.extend(list(y))

            val_acc = total_val_cnt / self.test_dataset.dataset_size
            self.test_acc_list.append(val_acc)
            print("======= val acc: ", val_acc)

        print(confusion_matrix(self.test_dataset.all_labels, y_list))
        print(classification_report(self.test_dataset.all_labels, y_list, target_names=class_list))

        if self.verbose:
            print("=============== Final Test Accuracy ===============")
            print("test acc:" + str(val_acc))
