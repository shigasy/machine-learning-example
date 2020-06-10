from test_process import test_process
import glob
import os.path as op

dirname = input("フォルダ名:")
files = glob.glob(dirname + "/*.jpg")
cn1 = 0
cn2 = 0
for imgname in files:
    kind = test_process(imgname)
    if kind == 1:
        cn2 += 1
    cn1 += 1
print("cn2である正答率は" + str(cn2*1.0/cn1) + "です。")  # cn2である割合
