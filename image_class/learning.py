
import numpy as np

# 自然数をベクトルに変換など便利な関数
from keras.utils import np_utils

#　学習中に行う処理
from keras.callbacks import LearningRateScheduler
from keras.callbacks import ModelCheckpoint

import constant
from build_cnn import build_cnn

import matplotlib.pyplot as plt


def learning(tsnum=2, nb_epoch=50, batch_size=8, learn_schedule=0.9):
    """
    学習
    tsnum = テストの数
    """

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
        # （水増しした画像を合わせた数 - tsnum=30）テストデータを分けるため。テストデータを学習しないように。精度を測るため、
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
    y_train = np_utils.to_categorical(
        Y_TRAIN, target)    # 自然数をベクトルに変換 # 出力層に合わせる one-hot表現に
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
    # 10 * 2 * 1 / 80 = 0.25 80枚のトレーニングデータのうち、20枚はテストデータの割合。それが、結合したテストデータ分テストデータとしてとして認識される
    valrate = tsnum * target * 1.0 / X_TRAIN.shape[0]

    # epoch数が増えるたびに、学習率を減らす
    # 学習が進むにつれ重みを収束しやすく。
    class Schedule(object):
        def __init__(self, init=0.001):
            # initが最初の学習率
            self.init = init

        # 一度生成されたインスタンスを関数っぽく引数を与えて呼び出せばcallが呼び出される
        def __call__(self, epoch):
            lr = self.init  # 現在の学習率
            for i in range(1, epoch + 1):
                # 0.001 * 学習スケジュール 0.9 = 0.0009 0.0009 * 0.9 = 0.00081 ...
                lr *= learn_schedule
            return lr

    def get_schedule_func(init):
        # ここでインスタンスが返って、LearningRateScheduleでインスタンス使う。そのインスタンスを関数っぽく呼び出すと、__call__が呼び出される。（コールバック関数）
        # コールバック関数で柔軟に。名前を決めずにロジックだけ変えていける。
        return Schedule(init)

    # --------------------------------------
    # 学習率を変化させるコールバック関数
    # 学習させているときに、コールバック関数として呼び出す。
    # コールバックが呼ばれるタイミングは関数の種類で異なる
    # --------------------------------------
    lrs = LearningRateScheduler(get_schedule_func(0.001))
    # 学習途中で val_loss が最も小さくなるたびに、重みを保存する関数
    mcp = ModelCheckpoint(filepath='best.hdf5', monitor='var_loss',
                          verbose=1, save_best_only=True, mode='auto')

    # X_TRAIN.shape[1], X_TRAIN.shape[2], X_TRAIN.shape[3]
    # ipshape=(16x16 チャネル3)
    model = build_cnn(
        ipshape=(X_TRAIN.shape[1], X_TRAIN.shape[2], X_TRAIN.shape[3]), num_classes=target)
    print(">> 学習開始")

    # * 訓練データ
    # * 正解ラベル
    # * batch_size: サンプル数ごとの勾配更新
    # * verbose: 0, 1または2．詳細表示モード．0とすると標準出力にログを出力しません． 1の場合はログをプログレスバーで標準出力，2の場合はエポックごとに1行のログを出力します
    # * epochs: エポック数
    # * validation_split: 0から1までの浮動小数点数． 訓練データの中で検証データとして使う割合． 訓練データの中から検証データとして設定されたデータは，訓練時に使用されず，各エポックの最後に計算される損失関数や何らかのモデルの評価関数で使われます．0.1を設定すると、10%が検証のために使われる。validation splitからデータを抽出する際にはデータがシャッフルされない。入力データの最後のx%のsampleに対して行われる。
    hist = model.fit(X_TRAIN, y_train, batch_size=batch_size,
                     verbose=1, epochs=nb_epoch, validation_split=valrate, callbacks=[lrs, mcp])

    # ----------------- 訓練の履歴の可視化 -----------------
    plt.plot(hist.history['accuracy'])
    plt.plot(hist.history['val_accuracy'])
    plt.title("Model accuracy")
    plt.ylabel('Accuracy')
    plt.xlabel('Epoch')
    plt.legend(['Train', 'Test'], loc='upper left')
    plt.show()

    plt.plot(hist.history['loss'])
    plt.plot(hist.history['val_loss'])
    plt.title('Model loss')
    plt.ylabel('Loss')
    plt.xlabel('Epoch')
    plt.legend(['Train', 'Test'], loc='upper left')
    plt.show()

    # ----------------- 学習モデルの保存 -----------------
    json_string = model.to_json()
    json_string += '#########' + str(constant.CLASS_NAMES)
    open('model.json', 'w').write(json_string)
    model.save_weights('last.hdf5')
