#!/usr/bin/env python2
# -*- coding: utf-8 -*-

import multiprocessing as mp

L = 20000
proc = 8

def subcalc(p): # p = 0,1,...,7
    subtotal = 0

    ini = L * p / proc
    fin = L * (p+1) / proc

    for i in range(ini, fin):
        for j in range(L):
            subtotal += i * j
    return subtotal

pool = mp.Pool(proc)
callback = pool.map(subcalc, range(8))
total = sum(callback)
print (total)
