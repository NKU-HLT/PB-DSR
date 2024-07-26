# Copyright (c) Facebook, Inc. and its affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

import re


SPACE_NORMALIZER = re.compile(r"\s+")


def tokenize_line(line):
    line = SPACE_NORMALIZER.sub(" ", line)
    line = line.strip()
    #------wsy fix----------------------------------------------------------------------------------
    # return line.split()
    # 注意
    # choose=1
    choose=2 # pb-dsr
    # choose=3
    if choose==1: # sn
        return line.split()# 此处根据空格切分之后是单词，单词怎么可能映射到词典呢？可能，修改词典对应即可
    elif choose==2: # word asr 或 kws
        line=line.lower()# 增加 了.lower() word asr
        return line.split()
    elif choose==3: # char asr 或 中文dsr（每个字都在字典里）
        # return line.replace(" ",".")
        return line.upper()
    #----------------------------------------------------------------------------------------------
