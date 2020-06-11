# callはインスタンスを関数っぽく呼び出したもの


class A:
    def __init__(self, a):
        self.a = a
        print("A init")

    def __call__(self, b):
        print("A call")
        print(self.a + b)

    def hoge(self):
        print('hoge')


a = A(1)
print('---')
a(2)
print('---')
a.hoge()
