#!/usr/bin/env python3
# _*_ coding: utf-8 _*_

from winograd import *
from sympy import symbols, Matrix, Poly, zeros, eye, Indexed, simplify, IndexedBase, init_printing, pprint, Rational


def foo():
    showCookToomFilter((0, 1, -1, 2, -2, Rational(1 / 2), Rational(-1 / 2),), 4, 3)


if __name__ == '__main__':
    foo()
