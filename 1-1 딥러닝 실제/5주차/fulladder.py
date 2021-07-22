import numpy as np


def AND(x1, x2):
    x = np.array([x1, x2])
    w = np.array([0.5, 0.5])
    b = -0.7
    tmp = np.sum(w * x) + b
    if tmp <= 0:
        return 0
    else:
        return 1


def NAND(x1, x2):
    x = np.array([x1, x2])
    w = np.array([-0.5, -0.5])
    b = 0.7
    tmp = np.sum(w * x) + b
    if tmp <= 0:
        return 0
    else:
        return 1


def OR(x1, x2):
    x = np.array([x1, x2])
    w = np.array([0.5, 0.5])
    b = -0.2
    tmp = np.sum(w * x) + b
    if tmp <= 0:
        return 0
    else:
        return 1


def XOR(x1, x2):
    s1 = NAND(x1, x2)
    s2 = OR(x1, x2)
    y = AND(s1, s2)
    return y


def HA(x1, x2):
    s = XOR(x1, x2)
    c = AND(x1, x2)
    return s, c


def FA(x1, x2, Cin):
    s, c1 = HA(x1, x2)
    s, c2 = HA(s, Cin)
    Cout = OR(c1, c2)
    return s, Cout


if __name__ == '__main__':
    print("X, Y, Cin -> S, Cout")
    for xs in [(0, 0, 0), (1, 0, 0), (0, 1, 0), (1, 1, 0), (0, 0, 1),
               (1, 0, 1), (0, 1, 1), (1, 1, 1)]:
        s, Cout = FA(xs[0], xs[1], xs[2])
        print(str(xs) + " -> " + str(s) + ", " + str(Cout))
