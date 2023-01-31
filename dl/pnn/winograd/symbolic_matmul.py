#!/usr/bin/env python3
# _*_ coding: utf-8 _*_

from winograd import *
from sympy import symbols, Matrix, Poly, zeros, eye, Indexed, simplify, IndexedBase, init_printing, pprint, Rational, \
    Idx


def Filter2DVerify(m, r, AT, G, BT):
    alpha = m + r - 1

    di = IndexedBase('d')
    gi = IndexedBase('g')

    d = Matrix(alpha, alpha, lambda i, j: di[i, j])
    g = Matrix(r, r, lambda i, j: gi[i, j])

    print("input:", d)

    V = BT * d * BT.T
    # Print Transformed data:
    print("Input Transformed: ")
    for i in range(alpha * alpha):
        print(V[i])
    print("*" * 10)

    U = G * g * G.T
    # print("Weight transformed: ", U)

    mi = IndexedBase('src')
    M_inter = Matrix(alpha, alpha, lambda i, j: mi[i, j])
    transformed_m = AT * M_inter * AT.T
    print("OutputTransformed LMS: ")
    for i in range(m * m):
        print(transformed_m[i])
    print("*" * 10)

    M = U.multiply_elementwise(V)
    Y = simplify(AT * M * AT.T)
    return Y


def FilterNotEqualVerify(m, n, r, k):
    AT_M, G_M, BT_M, f = cookToomFilter((0, 1, -1, 2, -2, Rational(1 / 2), Rational(-1 / 2),), m, r)
    AT_N, G_N, BT_N, f = cookToomFilter((0, 1, -1, 2, -2, Rational(1 / 2), Rational(-1 / 2),), n, k)

    alphaY = m + r - 1
    alphaX = n + k - 1;

    di = IndexedBase('d')
    gi = IndexedBase('g')

    d = Matrix(alphaY, alphaX, lambda i, j: di[i, j])
    g = Matrix(r, k, lambda i, j: gi[i, j])

    V = BT_M * d * BT_N.T
    # Print Transformed data:
    print("Input Transformed: ")
    for i in range(alphaY * alphaX):
        print(V[i])
    print("*" * 10)

    U = G_M * g * G_N.T

    mi = IndexedBase('src')
    M_inter = Matrix(alphaY, alphaX, lambda i, j: mi[i, j])
    transformed_m = AT_M * M_inter * AT_N.T
    print("OutputTransformed LMS: ")
    for i in range(m * n):
        print(transformed_m[i])
    print("*" * 10)


def Filter1DVerify(n, r, AT, G, BT):
    alpha = n + r - 1

    di = IndexedBase('d')
    gi = IndexedBase('g')
    d = Matrix(alpha, 1, lambda i, j: di[i])
    g = Matrix(r, 1, lambda i, j: gi[i])

    V = BT * d
    print("Input Transformed: ", V);
    U = G * g
    M = U.multiply_elementwise(V)

    mi = IndexedBase('m')
    M_inter = Matrix(alpha, 1, lambda i, j: mi[i])
    transformed_m = AT * M_inter

    print("OutputTransformed:")
    for i in range(n):
        print(transformed_m[i])

    Y = simplify(AT * M)

    return Y


def foo():
    m = 2
    r = 2
    # showCookToomFilter((0, 1, -1, 2, -2, Rational(1 / 2), Rational(-1 / 2),), m, r)
    AT, G, BT, f = cookToomFilter((0, 1, -1, 2, -2, Rational(1 / 2), Rational(-1 / 2),), m, r)
    # print(Filter1DVerify(m, r, AT, G, BT))
    print(Filter2DVerify(m, r, AT, G, BT))
    # showCookToomFilter((0, 1, -1, 2, -2, Rational(1 / 2), Rational(-1 / 2),), 4, 3)


def parseLine(line):
    '''
    d[0, 0] - 21*d[0, 2]/4 + 21*d[0, 4]/4 - d[0, 6] - 21*d[2, 0]/4 + 441*d[2, 2]/16 - 441*d[2, 4]/16 + 21*d[2, 6]/4 + 21*d[4, 0]/4 - 441*d[4, 2]/16 + 441*d[4, 4]/16 - 21*d[4, 6]/4 - d[6, 0] + 21*d[6, 2]/4 - 21*d[6, 4]/4 + d[6, 6]
    :param line:
    :return:
    '''
    line = line.replace(", ", "")
    line = line.replace("[", "")
    line = line.replace("]", "")
    line = line.replace("d", "src")
    line = line.replace("/4", " *0.25F")
    line = line.replace("/16", " *0.0625F")
    line = line.replace("/8", " *0.125F")
    line = line.replace("/2", " *0.5F")
    return line


def parseRes():
    txt = "/Users/bob/Desktop/tmp/res.txt"
    lines = []
    with open(txt, 'r') as f:
        for line in f.readlines():
            lines.append(line)
    res = [parseLine(x) for x in lines[1:]]
    srcs = []
    for i in range(8):
        for j in range(8):
            srcs.append("src{}{}".format(i, j))

    # print srcs
    # for s in srcs:
    #     print(s,end=',')

    # print load
    # for i in range(64):
    #     print("{} = SimdVec::load(src + {} * ALIGN)".format(srcs[i], i), end=";\n")

    # print calc
    for i in range(8):
        for j in range(8):
            pos = i * 8 + j
            print("SimdVec res{}{} = {};".format(i, j, res[pos]));
        for k in range(8):
            tmp = i * 8 + k
            print("SimdVec::store(dst + {} * dstStep, res{}{});".format(tmp, i, k))


if __name__ == '__main__':
    # foo()
    FilterNotEqualVerify(4, 4, 2, 2)
    # parseRes()
