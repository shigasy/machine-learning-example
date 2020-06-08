# -*- coding: utf-8 -*-
# 入力層：2 － 中間層：16(ReLU) － 出力層：2（Softmax）

import keras
# keras.layers モジュールから、Dense関数のみ読み込み
from keras.layers import Dense
from keras.layers import Activation
from keras.utils.vis_utils import plot_model
from keras.utils import np_utils
import numpy as np

# 学習データ: 入力
in_data = [
    [0, 0],
    [1, 0],
    [0, 1],
    [1, 1]
]
#学習データ: 正解
out_data = [0, 1, 1, 0]  # xor

# numpy.ndarray
# 多次元配列を表現できる
# 数値計算のためのメソッド・関数が豊富で、高速な演算が可能
# 行列演算や画像処理など様々な場面で使える
in_data = np.array(in_data)

# 正解データを one-hot 表現へ変換
out_data = np_utils.to_categorical(out_data, np.max(out_data) + 1)

# モデルの生成
model = keras.models.Sequential()

# 入力層: ユニット2 - 中間層: ユニット16
model.add(Dense(units=16, input_dim=2))  # units: 次の層（中間層）のユニット数 input_dim: 入力数
model.add(Activation('relu'))  # 活性化関数（sigmoid, relu, softmaxなど）

# 出力層
model.add(Dense(2))
model.add(Activation('softmax'))  # 活性化関数 足したら1.0の確率に変換

# モデルのコンパイル
model.compile(
    loss='categorical_crossentropy',      # 損失関数: クロスエントロピー誤差

    # optimizer = 'sgd'                   # SGD（確率的勾配降下法）
    optimizer='adam',                     # Adam
    metrics=['accuracy']  # モデルの精度表示
)
# モデルの画像出力(graphviz をインストールしていない場合はコメントにする)
plot_model(model, "model.png", show_shapes=True, show_layer_names=True)

# 学習
model.fit(in_data, out_data, epochs=100)
# model.load_weights('weights.hdf5')  # 学習済みの重みデータの読み込み。読み込む場合はfit()不要

# 予測
result = model.predict(in_data)             # 出力層の値
# [[0.52221036 0.47778964] 0 の方が重みが大きい
#  [0.45873725 0.54126275] 1 の方が重みが大きい
#  [0.47307363 0.52692634] 1 の方が重みが大きい
#  [0.53368354 0.46631646]] 0 の方が重みが大き良い
result2 = model.predict_classes(in_data)   # 出力層の値を分類に変換した値
# [0, 1, 1, 0]
print(result)
print(result2)

# 重みの保存
model.save_weights('weights.hdf5')
