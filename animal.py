import keras
from keras.layers import Dense
from keras.layers import Activation
import numpy as np
import pandas as pd
from sklearn.preprocessing import MinMaxScaler

# 二次元の表形式のデータ（テーブルデータ）を表す
data = pd.DataFrame(
    [
        # 足の数, 体重kg, R,   G,   B,   種類(0:象 1:キリン 2:白熊 3:黒熊 4:犬 5:クモ 6:タコ　7:イカ 8:虫)

        # 象さん
        [4,     6000,  210, 217, 206, 0],
        [4,     5800,  214, 213, 209, 0],
        [4,     5600,  206, 217, 208, 0],
        [4,     5400,  209, 214, 209, 0],
        [4,     5200,  208, 210, 215, 0],

        # キリンさん
        [4,      840,  203, 173,  52, 1],
        [4,      820,  194, 206,  49, 1],
        [4,      800,  196, 182,  60, 1],
        [4,      780,  181, 187,  68, 1],
        [4,      760,  196, 189,  60, 1],

        # 白熊さん
        [4,      490,  253, 253, 253, 2],
        [4,      470,  234, 235, 231, 2],
        [4,      450,  241, 242, 240, 2],
        [4,      430,  255, 255, 255, 2],
        [4,      410,  248, 250, 248, 2],

        # 黒熊さん
        [4,      490,   13,  13,  13, 3],
        [4,      470,   14,  15,  11, 3],
        [4,      450,   11,  12,  10, 3],
        [4,      430,   15,  15,  15, 3],
        [4,      410,   18,  10,  18, 3],

        # 犬さん
        [4,       14,  193,  97,   0, 4],
        [4,       12,  213, 106,   0, 4],
        [4,       10,  208, 121,   4, 4],
        [4,        8,  209, 153,   3, 4],
        [4,        6,  252, 184,   3, 4],

        # クモさん
        [8,     0.01,   14,  14,  12, 5],
        [8,     0.02,   12,  14,  14, 5],
        [8,     0.01,   56,  57,  49, 5],
        [8,     0.02,   58,  55,  48, 5],
        [8,     0.01,   38,  36,  32, 5],

        # タコさん
        [8,       18,  222,  33,  33, 6],
        [8,       16,  220,  23,  23, 6],
        [8,       14,  224,  50,  50, 6],
        [8,       12,  230,  50,  23, 6],
        [8,       10,  232,  40,  40, 6],

        # イカさん  色は白熊、体重はタコと同じにした
        [10,      18,  253, 253, 253, 7],
        [10,      16,  234, 235, 231, 7],
        [10,      14,  241, 242, 240, 7],
        [10,      12,  255, 255, 255, 7],
        [10,      10,  248, 250, 248, 7],

        # 虫さん   クモと足の数だけ違う
        [6,     0.01,   14,  14,  12, 8],
        [6,     0.02,   12,  14,  14, 8],
        [6,     0.01,   56,  57,  49, 8],
        [6,     0.02,   58,  55,  48, 8],
        [6,     0.01,   38,  36,  32, 8],
    ],
    columns=['legs', 'weight', 'R', 'G', 'B', 'kind']
)
#     legs   weight    R    G    B  kind
# 0      4  6000.00  210  217  206     0
# 1      4  5800.00  214  213  209     0
# ...
df = pd.DataFrame(data)

# データから分類以外の列を取得
in_data = df.iloc[:, 0:5]

# データから分類の列を取得
out_data = df['kind']

# 正規化（各データが 0 ～ 1 の範囲になるようにスケール変換します）列ごとに
# 正規化を行う理由は、出力に依存して、パラメータ間で更新幅に偏りが生じるから
scaler = MinMaxScaler()
in_data = scaler.fit_transform(in_data)
# in_data = pd.DataFrame(scaler.fit_transform(in_data))
#            0         1         2         3         4
# 0   0.000000  1.000000  0.815574  0.844898  0.807843
# 1   0.000000  0.966667  0.831967  0.828571  0.819608
# 2   0.000000  0.933333  0.799180  0.844898  0.815686
# 3   0.000000  0.900000  0.811475  0.832653  0.819608
# 4   0.000000  0.866666  0.807377  0.816327  0.843137
# 5   0.000000  0.139999  0.786885  0.665306  0.203922
# 6   0.000000  0.136665  0.750000  0.800000  0.192157
# 7   0.000000  0.133332  0.758197  0.702041  0.235294
# 8   0.000000  0.129999  0.696721  0.722449  0.266667
# 9   0.000000  0.126665  0.758197  0.730612  0.235294
# 10  0.000000  0.081665  0.991803  0.991837  0.992157
# 11  0.000000  0.078332  0.913934  0.918367  0.905882
# 12  0.000000  0.074998  0.942623  0.946939  0.941176
# 13  0.000000  0.071665  1.000000  1.000000  1.000000
# 14  0.000000  0.068332  0.971311  0.979592  0.972549
# 15  0.000000  0.081665  0.008197  0.012245  0.050980
# 16  0.000000  0.078332  0.012295  0.020408  0.043137
# 17  0.000000  0.074998  0.000000  0.008163  0.039216
# 18  0.000000  0.071665  0.016393  0.020408  0.058824
# 19  0.000000  0.068332  0.028689  0.000000  0.070588
# 20  0.000000  0.002332  0.745902  0.355102  0.000000
# 21  0.000000  0.001998  0.827869  0.391837  0.000000
# 22  0.000000  0.001665  0.807377  0.453061  0.015686
# 23  0.000000  0.001332  0.811475  0.583673  0.011765
# 24  0.000000  0.000998  0.987705  0.710204  0.011765
# 25  0.666667  0.000000  0.012295  0.016327  0.047059
# 26  0.666667  0.000002  0.004098  0.016327  0.054902
# 27  0.666667  0.000000  0.184426  0.191837  0.192157
# 28  0.666667  0.000002  0.192623  0.183673  0.188235
# 29  0.666667  0.000000  0.110656  0.106122  0.125490
# 30  0.666667  0.002998  0.864754  0.093878  0.129412
# 31  0.666667  0.002665  0.856557  0.053061  0.090196
# 32  0.666667  0.002332  0.872951  0.163265  0.196078
# 33  0.666667  0.001998  0.897541  0.163265  0.090196
# 34  0.666667  0.001665  0.905738  0.122449  0.156863
# 35  1.000000  0.002998  0.991803  0.991837  0.992157
# 36  1.000000  0.002665  0.913934  0.918367  0.905882
# 37  1.000000  0.002332  0.942623  0.946939  0.941176
# 38  1.000000  0.001998  1.000000  1.000000  1.000000
# 39  1.000000  0.001665  0.971311  0.979592  0.972549
# 40  0.333333  0.000000  0.012295  0.016327  0.047059
# 41  0.333333  0.000002  0.004098  0.016327  0.054902
# 42  0.333333  0.000000  0.184426  0.191837  0.192157
# 43  0.333333  0.000002  0.192623  0.183673  0.188235
# 44  0.333333  0.000000  0.110656  0.106122  0.125490

in_data = np.array(in_data)
out_data = np.array(out_data)


model = keras.models.Sequential()
model.add(Dense(units=16, input_dim=5))
model.add(Activation('relu'))
model.add(Dense(9))  # 出力層: 9種類のためのユニット数は9
model.add(Activation('softmax'))

model.compile(
    loss='sparse_categorical_crossentropy',
    optimizer='adam',
    metrics=['accuracy']
)
model.fit(in_data, out_data, epochs=1000)
# model.load_weights('weights_animal.hdf5')

# テストデータ(学習データと微妙に変えています)
test_data = np.array(
    [
        # 象さん
        [4,     5700,  208, 213, 209],
        [4,     5500,  200, 217, 208],
        [4,     5650,  206, 210, 211],
        [4,     5450,  209, 212, 213],
        [4,     5250,  211, 210, 210],

        # キリンさん
        [4,      845,  205, 176,  48],
        [4,      815,  190, 206,  49],
        [4,      805,  197, 181,  52],
        [4,      785,  180, 187,  60],
        [4,      720,  200, 189,  68],

        # 白熊さん
        [4,      495,  250, 255, 253],
        [4,      475,  240, 235, 240],
        [4,      455,  235, 245, 245],
        [4,      435,  245, 255, 250],
        [4,      400,  255, 252, 245],

        # 黒熊さん
        [4,      495,   14,  14,  14],
        [4,      465,   11,  15,  14],
        [4,      455,   10,  12,  17],
        [4,      437,   13,  15,  13],
        [4,      413,   15,  10,  16],

        # 犬さん
        [4,       12.5,  193, 100,   2],
        [4,       10,    213, 107,   1],
        [4,        9,    208, 118,   3],
        [4,        7,    209, 163,   4],
        [4,        5,    252, 174,   5],

        # クモさん
        [8,     0.03,   12,  12,  11],
        [8,     0.01,   16,  16,  13],
        [8,     0.01,   56,  55,  50],
        [8,     0.01,   58,  57,  46],
        [8,     0.01,   36,  36,  33],

        # タコさん
        [8,       17,  220,  30,  31],
        [8,       15,  222,  27,  20],
        [8,       13,  225,  55,  55],
        [8,       11,  231,  50,  21],
        [8,        9,  232,  45,  45],

        # イカさん
        [10,      17,  251, 250, 250],
        [10,      15,  231, 240, 240],
        [10,      15,  240, 245, 242],
        [10,      13,  253, 255, 253],
        [10,      11,  250, 250, 250],

        # 虫さん
        [6,     0.01,   12,  11,  12],
        [6,     0.02,   12,  12,  14],
        [6,     0.02,   56,  48,  49],
        [6,     0.02,   56,  50,  50],
        [6,     0.02,   40,  40,  40],
    ]
)

test_data = pd.DataFrame(scaler.transform(test_data))
test_data = np.array(test_data)

# 予測
# result = model.predict(test_data)             # 出力層の値
result = model.predict_classes(test_data)

print(result)

model.save_weights('weights_animal.hdf5')
