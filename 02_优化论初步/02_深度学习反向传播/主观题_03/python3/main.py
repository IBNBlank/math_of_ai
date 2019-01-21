# -*- coding: utf-8 -*-
# @Author: IBNBlank
# @Date:   2019-01-01 20:39:53
# @Last Modified by:   IBNBlank
# @Last Modified time: 2019-01-01 23:09:51

import numpy as np

import datafile as df
import BP_BACK as BP

if __name__ == '__main__':
    df.datafile()
    data = np.load("data.npy")

    BP.BP_BACK(data, 0.9)