#!/usr/bin/env
import numpy as np 
import os

def makeDir(dirname):
    if os.path.exists(dirname) == False:
        if os.path.exists(os.path.split(dirname)[0]) == False:
            print('making directory: ', os.path.split(dirname)[0])
            os.mkdir(os.path.split(dirname)[0])
        print('making directory: ', dirname)
        os.mkdir(dirname)

def round2dec(num2round):
    roundednum = np.round(num2round,2)
    return roundednum
