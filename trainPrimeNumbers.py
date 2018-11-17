from __future__ import absolute_import, division, print_function

import os
import matplotlib.pyplot as plt

import tensorflow as tf
import tensorflow.contrib.eager as tfe

def decimalToBinary(dec, numberOfBits):
    answer = []
    while(dec > 0):
        answer.append(dec % 2)
        dec = dec // 2
    zeros = []
    for i in range(0, numberOfBits-len(answer)):
        zeros.append(0)
    return zeros + answer

def binaryToDecimal(bin):
    place = 1
    sum = 0
    for i in range(len(bin)-1, -1, -1):
        sum += (place * bin[i])
        place *= 2
    return sum