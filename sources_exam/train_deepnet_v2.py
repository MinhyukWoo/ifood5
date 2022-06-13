# coding: utf-8
import sys, os
sys.path.append(os.pardir)  # 부모 디렉터리의 파일을 가져올 수 있도록 설정
import numpy as np
import matplotlib.pyplot as plt
from deep_convnet import DeepConvNet
# from common.trainer import Trainer
from sources_exam.common.trainer_v2 import Trainer
from DatasetDAO import DatasetDAO

max_epochs = 20
batch_size = 60

train_dataset = DatasetDAO(
    dataset_path='../resources/iFood5_dataset/mini_train_set/', csv_path='../resources/iFood5_dataset/annot/mini_train_info.csv')
test_dataset = DatasetDAO(
    dataset_path='../resources/iFood5_dataset/mini_val_set/', csv_path='../resources/iFood5_dataset/annot/mini_val_info.csv')

# network = SimpleConvNet(input_dim=(3, 64, 64),
#                         conv_param = {'filter_num': 30, 'filter_size': 3, 'pad': 0, 'stride': 1},
#                         hidden_size=100, output_size=5, weight_init_std=0.01)

network = DeepConvNet(input_dim=(3, 64, 64), hidden_size=1000, output_size=5)
trainer = Trainer(network, train_dataset, test_dataset,
                  epochs=max_epochs, mini_batch_size=batch_size,
                  optimizer='Adam', optimizer_param={'lr': 0.0001})
trainer.train()

# 매개변수 보관
network.save_params("deep_convnet_params.pkl")
print("Saved Network Parameters!")

# 그래프 그리기
markers = {'train': 'o', 'test': 's'}
x = np.arange(max_epochs)
plt.plot(x, trainer.train_acc_list, marker='o', label='train', markevery=2)
plt.plot(x, trainer.test_acc_list, marker='s', label='test', markevery=2)
plt.xlabel("epochs")
plt.ylabel("accuracy")
plt.ylim(0, 1.0)
plt.legend(loc='lower right')
plt.show()