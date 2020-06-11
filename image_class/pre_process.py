import glob  # ファイル読み込みに使うやつ
import numpy as np

# 前処理
from keras.preprocessing.image import load_img, img_to_array
from keras.preprocessing.image import random_rotation, random_shift, random_zoom

import constant

import os.path as op


from PIL import Image


def pre_process(dirname, filename, var_amount=3):
    """
    画像データの前処理

    ## 機能
    サイズを16x16に統一
    画像を回転させて学習用データを増やす(var_amount=3倍)1枚につき3枚に
    """
    num = 0  # 画像ファイル数
    arrlist = []  # 画像ファイルをnumpy型に変換して入れるリスト
    files = glob.glob(dirname + "/*.jpg")  # ファイル名抽出

    for imgfile in files:
        img = load_img(imgfile, target_size=(
            constant.hw["height"], constant.hw["width"]))  # 画像ファイルの読み込み
        array = img_to_array(img) / 255  # 画像ファイルのnumpy化
        # [[[ 37.  89.  27.]
        # [183. 200. 146.]
        # [183. 194. 151.]
        # [220. 221. 181.]
        # [226. 226. 188.]
        # [233. 243. 209.]
        # [237. 241. 206.]
        # [234. 229. 197.]
        # [210. 212. 173.]
        # [208. 218. 183.]
        # [206. 215. 184.]
        # [178. 197. 165.]
        # [166. 202. 128.]
        # [118. 120. 107.]
        # [111. 115. 100.]
        # [ 68.  75.  67.]][...]] 16x16 の RGB を / 255
        arrlist.append(array)                 # numpy型データをリストに追加
        # for i in range(var_amount - 1):
        #     arr2 = array
        #     arr2 = random_rotation(arr2, rg=360)
        #     arrlist.append(arr2)
        num += 1

    nplist = np.array(arrlist)  # ディレクトリ内の16x16 rgbデータをnumpy配列に
    np.save(filename, nplist)  # numpyリストをfilenameで保存
    print(">> " + dirname + "から" + str(num) + "個のファイル読み込み成功")


def main():
    """
    前処理をして、FILE_NAMESに書いてあるimg1やimg2などの.npyを更新
    """
    i = 0
    for filename in constant.FILE_NAMES:
        while True:
            dirname = input(">>「" + constant.CLASS_NAMES[i] + "」の画像があるディレクトリ:")
            if op.isdir(dirname):
                break
            print("そのディレクトリは存在しません。")
        print(dirname)
        # Pythonのスコープはモジュールと関数しかない。
        pre_process(dirname, filename, var_amount=3)
        i += 1


if __name__ == "__main__":
    main()
