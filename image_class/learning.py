
import numpy as np

# 自然数をベクトルに変換など便利な関数
from keras.utils import np_utils

#　学習中に行う処理
from keras.callbacks import LearningRateScheduler
from keras.callbacks import ModelCheckpoint

import constant
from build_cnn import build_cnn

# 学習
# tsnum = テストの数


def learning(tsnum=2, nb_epoch=50, batch_size=8, learn_schedule=0.9):

    # Xは画像データ
    # Yは正解ラベル

    # 入力データ（画像）ex. [[[0.2509804  0.49803922 0.18039216][0.7176471  0.81960785 0.63529414]...]]
    X_TRAIN_list = []
    Y_TRAIN_list = []  # 正解ラベル（教師データ）分類番号 ex. [0, 0, 0, 0, 1, 1, 1, 1, 2, 2...]
    X_TEST_list = []  # テスト画像データ
    Y_TEST_list = []  # テスト正解ラベル
    target = 0
    for filename in constant.FILE_NAMES:
        data = np.load(filename)  # ディレクトリ毎のnpyを読み込む
        # print(len(data)) # testデータ2枚 * 3倍 最初の次元数
        # print(data.shape)  # (6, 16, 16, 3) 6枚 16x16 3RGB
        # （水増しした画像を合わせた数 - tsnum=30）テストデータを分けるため、データを学習しないように。精度を測るため、
        trnum = data.shape[0] - tsnum

        X_TRAIN_list += [data[i]
                         for i in range(trnum)]  # 写真データ1枚ずつX_TEST_listに足していく
        # [target]の配列 * trnumの数 ex. [0] * 4 = [0, 0, 0, 0] 正解ラベルを写真の枚数だけ生成する
        Y_TRAIN_list += [target] * trnum

        # トレーニングデータで学習していない画像データをtsnum枚使用
        X_TEST_list += [data[i] for i in range(trnum, trnum+tsnum)]
        # [target]の配列 * tsnumの数 ex. [0] * 4 = [0, 0, 0, 0]
        Y_TEST_list += [target] * tsnum
        target += 1

    X_TRAIN = np.array(X_TRAIN_list + X_TEST_list)    # 連結
    Y_TRAIN = np.array(Y_TRAIN_list + Y_TEST_list)    # 連結
    print(">> 学習サンプル数 : ", X_TRAIN.shape)
    y_train = np_utils.to_categorical(Y_TRAIN, target)    # 自然数をベクトルに変換
    # [0, 1, 2] を ベクトル([1, 0, 0], [0, 1, 0], [0, 0, 1])に変換
    #  [[1. 0.]
    #  [1. 0.]
    #  [1. 0.]
    #  [1. 0.]
    #  [0. 1.]
    #  [0. 1.]
    #  [0. 1.]
    #  [0. 1.]
    #  [1. 0.]
    #  [1. 0.]
    #  [0. 1.]
    #  [0. 1.]]
    # 2（テストデータ2枚） * 2（分類数）* 1.0 / 12
    # データ全体のうちどれくらいの割合を精度確認用のテストデータにするかを指定する値
    valrate = tsnum * target * 1.0 / X_TRAIN.shape[0]

    # epoch数が増えるたびに、学習率を減らす
    # TODO: 調べる
    class Schedule(object):
        def __init__(self, init=0.001):
            self.init = init

        # 一度生成されたインスタンスを関数っぽく引数を与えて呼び出せばcallが呼び出される
        def __call__(self, epoch):
            lr = self.init
            for i in range(1, epoch + 1):
                lr *= learn_schedule
            return lr

    def get_schedule_func(init):
        return Schedule(init)

    # 学習率を変化させるコールバック関数
    # __call__で
    # TODO: ここ何やってるの
    lrs = LearningRateScheduler(get_schedule_func(0.001))
    mcp = ModelCheckpoint(filepath='best.hdf5', monitor='var_loss',
                          verbose=1, save_best_only=True, mode='auto')

    # X_TRAIN.shape[1], X_TRAIN.shape[2], X_TRAIN.shape[3]
    # ipshape=(16x16 チャネル3)
    model = build_cnn(
        ipshape=(X_TRAIN.shape[1], X_TRAIN.shape[2], X_TRAIN.shape[3]), num_classes=target)
    print(">> 学習開始")

    # TODO: 引数調べる
    hist = model.fit(X_TRAIN, y_train, batch_size=batch_size,
                     verbose=1, epochs=nb_epoch, validation_split=valrate)
    json_string = model.to_json()
    json_string += '#########' + str(constant.CLASS_NAMES)
    open('model.json', 'w').write(json_string)
    model.save_weights('last.hdf5')
