# Keras 画像分類

## 前処理
学習させる前に画像データの前処理が必要

```python
# constant.py
FILE_NAMES = ["img1.npy", "img2.npy"]　# 学習するデータの名前
CLASS_NAMES = ["マグロの刺し身", "マグロ"] # 分類するデータのラベル
hw = {"height": 128, "width": 128} # 学習データの画像サイズ
```
設定後に下記を実行し、ディレクトリ名を入力
```shell
$ python pre_process.py
```


## 学習
学習し、モデルと重みを保存する。
| ファイル名 | 役割                       |
| ---------- | -------------------------- |
| model.json | 学習モデル自体のデータ     |
| model.png  | 学習モデルの図で表したもの |
| last.hdf5  | 学習済みモデルの重み       |

```shell
$ python main.py
```

## 判定
予測したCLASS_NAMESが出力される。
| 1枚ずつ判定 |
| ----------- |
```shell
$ python test_process.py
```


| フォルダ内の画像一括判定 |
| ------------------------ |
CLASS_NAMES[0]と判断した割合が出力される。
```shell
$ python test_process.py
````
