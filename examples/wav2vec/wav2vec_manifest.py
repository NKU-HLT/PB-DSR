#!/usr/bin/env python3
# Copyright (c) Facebook, Inc. and its affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.
"""
Data pre-processing: build vocabularies and binarize training data.
"""

import argparse
import glob
import os
import random

import soundfile


def get_parser():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        #---wsy fix--------------------------------------------------------------------
        # "root", 
        "--root", 
        # 注意:
        # default="/home/wangshiyao/wangshiyao_space/LRDWWS/test-b/eval/wav/DM0041",
        # default="/home/wangshiyao/wangshiyao_space/TOGRO",
        # default="/home/wangshiyao/wangshiyao_space/vits/wsy/torgo_syn/remove0",
        # default="/home/wangshiyao/wangshiyao_space/vits/wsy/torgo_syn/remove1",
        # default="/home/wangshiyao/wangshiyao_space/vits/wsy/torgo_syn/remove2",
        # default="/home/wangshiyao/wangshiyao_space/vits/wsy/torgo_syn/remove3",
        # default="/home/wangshiyao/wangshiyao_space/vits/wsy/torgo_syn/remove4",
        # default="/home/wangshiyao/wangshiyao_space/vits/wsy/torgo_syn/remove5",
        # default="/home/wangshiyao/wangshiyao_space/vits/wsy/torgo_syn/remove6",
        # default="/home/wangshiyao/wangshiyao_space/vits/wsy/torgo_syn/remove7",
        # default="/home/wangshiyao/wangshiyao_space/vits/wsy/torgo_syn/remove8",
        # default="/home/wangshiyao/wangshiyao_space/vits/wsy/torgo_syn/remove9",
        # default="/home/wangshiyao/wangshiyao_space/vits/wsy/torgo_syn/remove10",
        # default="/home/wangshiyao/wangshiyao_space/vits/wsy/torgo_syn/remove11",
        # default="/home/wangshiyao/wangshiyao_space/vits/wsy/torgo_syn/remove12",
        # default="/home/wangshiyao/wangshiyao_space/vits/wsy/torgo_syn/remove13",
        # default="/home/wangshiyao/wangshiyao_space/vits/wsy/torgo_syn/remove14",
        default="/home/wangshiyao/wangshiyao_space/dataset/cdsd_cutting/Audio",
        #-----------------------------------------------------------------------------
        metavar="DIR", help="root directory containing flac files to index"
    )
    parser.add_argument(
        "--valid-percent",
        #------wsy fix---------------
        # default=0.01,
        # 注意：
        default=0, # 全部作为训练集
        # default=1, # 全部作为验证集
        #---------------------------
        type=float,
        metavar="D",
        help="percentage of data to use as validation set (between 0 and 1)",
    )
    parser.add_argument(
        "--dest", 
        #--wsy fix----------------------------------------------------------------------
        # 注意：
        # default="/home/wangshiyao/wangshiyao_space/fairseq/wsy/data/lrdwwk/eval",
        # default="/home/wangshiyao/wangshiyao_space/fairseq/wsy/private/data/torgo",
        # default="/home/wangshiyao/wangshiyao_space/fairseq/wsy/private/data/torgo/syn",
        default="/home/wangshiyao/wangshiyao_space/fairseq/wsy/private/data/cdsd",
        #------------------------------------------------------------------------------
        type=str, metavar="DIR", help="output directory"
    )
    parser.add_argument(
        "--ext", 
        #-----wsy fix-----
        # default="flac", 
        default="wav",
        #------------------
        type=str, metavar="EXT", help="extension to look for"
    )
    parser.add_argument("--seed", default=42, type=int, metavar="N", help="random seed")
    parser.add_argument(
        "--path-must-contain",
        default=None,
        type=str,
        metavar="FRAG",
        help="if set, path must contain this substring for a file to be included in the manifest",
    )
    return parser


def main(args):
    assert args.valid_percent >= 0 and args.valid_percent <= 1.0

    if not os.path.exists(args.dest):
        os.makedirs(args.dest)

    dir_path = os.path.realpath(args.root)
    search_path = os.path.join(dir_path, "**/*." + args.ext)
    rand = random.Random(args.seed)

    valid_f = (
        open(os.path.join(args.dest, "valid.tsv"), "w")
        if args.valid_percent > 0
        else None
    )

    with open(os.path.join(args.dest, "train.tsv"), "w") as train_f:
        print(dir_path, file=train_f)

        if valid_f is not None:
            print(dir_path, file=valid_f)

        #-------wsy add for torgo-------------------------------------------------------
        bad_files=[
                "/llm/nankai/wangshiyao_space/TOGRO/F01/Session1/wav_headMic/0068.wav",
                "/llm/nankai/wangshiyao_space/TOGRO/F01/Session1/wav_headMic/0067.wav",
                ]
        count=0
        #----------------------------------------------------------------------------------
        for fname in glob.iglob(search_path, recursive=True):
            file_path = os.path.realpath(fname)

            if args.path_must_contain and args.path_must_contain not in file_path:
                continue
            #------wsy add for torgo-----------
            # if "wav_arrayMic" not in fname:
            #     continue
            # if "wav_headMic"not in fname:
            #     continue
            # elif fname in bad_files:
            #     continue
            count+=1
            if count%100==0:
                print(count)
            #----------------------------------------------------------------------------------
            frames = soundfile.info(fname).frames
            dest = train_f if rand.random() > args.valid_percent else valid_f
            print(
                "{}\t{}".format(os.path.relpath(file_path, dir_path), frames), file=dest
            )
    if valid_f is not None:
        valid_f.close()


if __name__ == "__main__":
    parser = get_parser()
    args = parser.parse_args()
    main(args)