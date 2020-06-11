
from learning import learning


def main():
    """
    learn_scheduleが大きいほど、エポック数による学習率を減らす割合が小さくなる
    """
    learning(tsnum=10, nb_epoch=100, batch_size=8, learn_schedule=0.9)
    # learning(tsnum=10, nb_epoch=100, batch_size=8, learn_schedule=0.9)


if __name__ == "__main__":
    main()
