import os
task="s2st"
# task="none" 

## asr记录oov
# record_oov=True
record_oov=False
oov_dir="oov_dir"

## 数据增强相关
# 注意
# addwavaug=True 
addwavaug=False
aug_choose=1 # wavaugment：pitch+混响
# aug_choose=2 # 速度扰动

## 对比学习相关：
# 注意
# use_cl_loss=True
use_cl_loss=False

## pb-dsr相关
# 注意
# prepare_datastore=True
prepare_datastore=False
# o_datastore_dir="datastore_dir"
o_datastore_dir="testpkl_dir"
if prepare_datastore and (not os.path.exists(o_datastore_dir)):
    os.makedirs(o_datastore_dir)
# 注意
# knn_aug=True
knn_aug=False

# 是否在词典中添加sos,eos,pad,blank
add_special_symbols=True # 增加sos,eos,pad,blank
# add_special_symbols=False # 这个设置不了，因为所使用的框架是基于ctc的，而ctc需要配置这些。

print("record_oov:",record_oov,"addwavaug:",addwavaug,"aug_choose:",aug_choose,"use_cl_loss:",use_cl_loss,"prepare_datastore:",prepare_datastore,"knn_aug:",knn_aug,"add_special_symbols:",add_special_symbols)
