import numpy as np
from keras.models import model_from_json

from keras.preprocessing.image import load_img, img_to_array

import constant
import os.path as op


def test_process(imgname):
    """
    画像を読み込んで、学習結果を元に判定
    imgname = imgの場所
    """

    modelname_text = open("model.json").read()
    # モデルと分類テキストをsplit #########['マグロの刺し身', 'マグロ']
    json_strings = modelname_text.split('#########')
    textlist = json_strings[1].replace(
        "[", "").replace("]", "").replace("\'", "").replace(",", "").split()  # 文字列をリスト化するため
    model = model_from_json(json_strings[0])  # モデルの読み込み
    model.load_weights("last.hdf5")  # 　重みの読み込み
    img = load_img(imgname, target_size=(
        constant.hw["height"], constant.hw["width"]))
    TEST = img_to_array(img) / 255  # テストしたい画像の数値

    pred = model.predict(np.array([TEST]), batch_size=1, verbose=0)  # 学習結果が出る
    print(">> 計算結果↓\n" + str(pred))
    # >> 計算結果↓
    # [[0.52773863 0.4722614 ]]

    print(">> この画像は「" + textlist[np.argmax(pred)
                                 ].replace(",", "") + "」です。")  # argmaxで配列内の一番大きな要素のインデックスを返す。[[0.52773863 0.4722614 ]]だった場合、textlist[0]を表示
    # >> この画像は「マグロの刺し身」です。

    return np.argmax(pred)


def main():
    while True:
        while True:
            imgname = input("\n>> 入力したい画像ファイル(「END」で終了) ： ")
            if op.isfile(imgname) or imgname == "END":
                break
            print(">> そのファイルは存在しません")
        if imgname == "END":
            break
        test_process(imgname)


if __name__ == "__main__":
    main()
