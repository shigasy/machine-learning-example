# 層
from keras.layers.convolutional import Conv2D
from keras.layers.pooling import MaxPooling2D
from keras.layers.core import Activation
from keras.layers.core import Dense
from keras.layers.core import Dropout
from keras.layers.core import Flatten

# モデルそのもの
from keras.models import Sequential

# 最適化のアルゴリズム 学習の調整手法
from keras.optimizers import Adam

from keras.utils.vis_utils import plot_model


def build_cnn(ipshape=(32, 32, 3), num_classes=3):
    """
    cnnの構築
    ipshape = 入力層 32x32 3チャネル
    """
    model = Sequential()  # 分岐したり合流したりしない、単純なモデル定義

    # ----------------- 層1 -----------------

    # 畳み込み3x3のフィルター 24回
    # padding='same'は画像の周りを0で埋める
    # データの縦横のサイズを変化させないという特徴
    # 畳み込み回数分、色んなフィルターで特徴を抽出
    # 3 * 3 のRGBのフィルターを24個用意して、チャネル数フィルター通して、全て足す。合計、3*3の24枚出力
    # https://kenyu-life.com/2019/03/07/convolutional_neural_network/
    model.add(Conv2D(24, 3, padding='same', input_shape=ipshape))

    model.add(Activation('relu'))  # 活性化関数にrelu関数

    # ----------------- 層2 -----------------
    # 色んな重みのフィルターを48枚用意する。1枚ごとに、24input全てフィルター掛けて足す。それが48枚出来る
    model.add(Conv2D(48, 3))  # 3x3フィルター48回
    model.add(Activation('relu'))

    # プーリング層: 画像データを2x2の送料域に分割し、その中の最大値を出力
    model.add(MaxPooling2D(pool_size=(2, 2)))
    model.add(Dropout(0.5))  # 50%を0に置き換えて、過学習を抑える

    # ----------------- 層3 -----------------
    model.add(Conv2D(96, 3, padding='same'))
    model.add(Activation('relu'))

    # ----------------- 層4 -----------------
    model.add(Conv2D(96, 3))
    model.add(Activation('relu'))
    model.add(MaxPooling2D(pool_size=(2, 2)))
    model.add(Dropout(0.5))

    # ----------------- 層5 -----------------
    model.add(Flatten())  # 2x2 の96チャネルを平らに = 384要素
    model.add(Dense(128))  # FlattenとDenseで128個の1次元配列に
    model.add(Activation('relu'))
    model.add(Dropout(0.5))

    # ----------------- 層6 -----------------
    model.add(Dense(num_classes))
    model.add(Activation('softmax'))

    # 最適化関数
    # * lr: 0以上の浮動小数点数．学習率
    # * beta_1: adam関数の分母のβ 浮動小数点数, 0 < beta < 1. 一般的に1に近い値です
    # * beta_2: dam関数の分子のβ 浮動小数点数, 0 < beta < 1. 一般的に1に近い値です
    # * epsilon: 各更新の学習率減衰
    adam = Adam(lr=0.001, beta_1=0.9, beta_2=0.999, epsilon=1e-08)

    # クロスエントロピー誤差
    # モデルコンパイル
    model.compile(loss='categorical_crossentropy',
                  optimizer=adam, metrics=['accuracy'])

    plot_model(model, "model.png", show_shapes=True, show_layer_names=True)

    return model
