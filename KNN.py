import numpy as np
from collections import Counter
import os, copy, gzip, math
import csv

# 传入本地 mnist 数据集路径，实现加载
def load_mnist(data_folder):

    files = \
        [
            'train-labels-idx1-ubyte.gz',  # train_labels
            'train-images-idx3-ubyte.gz',  # train_images
            't10k-labels-idx1-ubyte.gz',   # test_labels
            't10k-images-idx3-ubyte.gz',   # test_images
        ]

    paths = []

    for fname in files:
        paths.append(os.path.join(data_folder, fname))

    with gzip.open(paths[0], 'rb') as lbpath:
        y_train = np.frombuffer(lbpath.read(), np.uint8, offset=8)

    with gzip.open(paths[1], 'rb') as imgpath:
        x_train = np.frombuffer(
            imgpath.read(), np.uint8, offset=16).reshape(len(y_train), 28, 28)

    with gzip.open(paths[2], 'rb') as lbpath:
        y_test = np.frombuffer(lbpath.read(), np.uint8, offset=8)

    with gzip.open(paths[3], 'rb') as imgpath:
        x_test = np.frombuffer(
            imgpath.read(), np.uint8, offset=16).reshape(len(y_test), 28, 28)

    return (x_train, y_train), (x_test, y_test)

# calculate the L0 distance between two vectors v1, v2
def dis_L0(v1, v2):
    dis = 0
    for i in range(len(v1)):
        dis += 0 if v1[i] == v2[i] else 1
    return dis

# calculate the L1 distance between two vectors v1, v2
def dis_L1(v1, v2):
    dis = 0
    for i in range(len(v1)):
        dis += abs(v1[i] - v2[i])
    return dis

# calculate the L2 distance between two vectors v1, v2
def dis_L2(x1, x2):
    dis = 0
    for i in range(len(x1)):
        dis += (x1[i] - x2[i])**2
    return math.sqrt(dis)

class Knn():
    def __init__(self, x, y, x_t, y_t, c_dis=None, k=0):

        self.x = x.reshape([len(x), -1]) / 255
        self.x_t = x_t.reshape([len(x_t), -1]) / 255
        self.y = y.reshape([len(y)])
        self.y_t = y_t.reshape([len(y_t)])

        self.len_x = len(x)
        self.len_xt = len(x_t)

        self.dis = c_dis
        self.k = k
        self.n_k = np.zeros((self.len_x, self.k, 2))
        self.y_pre = np.zeros(self.len_x)

    def get_nei_k(self):
        diss = np.zeros((self.len_x, self.len_xt, 2))
        for i in range(self.len_x):
            for j in range(self.len_xt):
                diss[i][j] = [self.dis(self.x[i], self.x_t[j]), self.y_t[j]]
            self.n_k[i] = sorted(diss[i], key=(lambda x : x[0]))[:self.k]
        return self.n_k

    def pre_vote(self, k):
        k = k if k <= self.k and k > 0 else self.k
        tmp = [-1] * k
        for i in range(len(self.x)):
            for j in range(k):
                tmp[j] = self.n_k[i][j][1]
            self.y_pre[i] = Counter(tmp).most_common(1)[0][0]
        return self.y_pre

    def c_acc(self):
        s = 0
        for i in range(self.len_x):
            s += 1 if self.y_pre[i] == self.y[i] else 0
        return s/self.len_x

def exp_knn(x, y, xt, yt, Knn=Knn, k=0, foder=None):
    header = \
        [
            'distance',
            'k',
            'accu',
        ]

    infos = []
    info = [-1] * len(header)

    knn = Knn(x, y, xt, yt, dis_L0, k)
    knn.get_nei_k()
    info[0] = 0
    for v in range(1, k+1, 1):
        info[1] = v
        knn.pre_vote(v)
        info[2] = knn.c_acc()
        infos.append(copy.deepcopy(info))

    knn = Knn(x, y, xt, yt, dis_L1, k)
    knn.get_nei_k()
    info[0] = 1
    for v in range(1, k + 1, 1):
        knn.pre_vote(v)
        info[1] = v
        info[2] = knn.c_acc()
        infos.append(copy.deepcopy(info))

    knn = Knn(x, y, xt, yt, dis_L2, k)
    knn.get_nei_k()
    info[0] = 2
    for v in range(1, k + 1, 1):
        knn.pre_vote(v)
        info[1] = v
        info[2] = knn.c_acc()
        infos.append(copy.deepcopy(info))

    os.makedirs(foder, exist_ok=True)
    with open(foder + '/{}.csv'.format(k), 'w', encoding='utf-8', newline='') as file:
        writer = csv.writer(file)
        writer.writerow(header)
        writer.writerows(infos)

(train_images, train_labels), (test_images, test_labels) = load_mnist("/root")
train_images, train_labels, test_images, test_labels = train_images[:1000], train_labels[:1000], test_images[:2000], test_labels[:2000]

foder = "./log"
exp_knn(train_images, train_labels, test_images, test_labels, Knn, 100, foder)







