import glob  # ファイル読み込みに使うやつ
import numpy as np


from pre_process import pre_process
from learning import learning


def test_process(imgname):
    pass


pre_process('image_class/data/img/maguro_sashimi', 'img1.npy')
pre_process('image_class/data/img/maguro', 'img2.npy')
# pre_process('./data/img/buri', 'img3.npy')
learning(tsnum=5, nb_epoch=100, batch_size=4, learn_schedule=0.1)
