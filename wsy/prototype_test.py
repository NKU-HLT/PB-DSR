import argparse
import os
os.environ["CUDA_VISIBLE_DEVICES"] = "0" # 注意

import torch 
import pickle as pkl
import numpy as np
import faiss
from glob import glob
import librosa
import torch.nn.functional as F

import warnings
warnings.filterwarnings("ignore")

# 控制参数：=============
# 注意 设置说话人id
ltr_name="M08" 

# 注意 设置度量策略
policy="4" 

# 注意 是否是remove测试
remove_one=True 

# 注意 测试模式
# test_mode="base_model" 
test_mode="removeone or removeoneft"

# 注意 设置特征类型：使用完整hubert特征或仅使用第一帧，支持集与测试集的设置需要一样
# only_first_frame=False 
only_first_frame=True

# 注意 设置特征是否平均
average_feature=True 

topk=3 # 预测错误时的输出的topk设置

# 以下设置不需要修改：
top=1 
is_tensor=False
my_dim=768
feature_suffix=".pkl"
print("removeone",remove_one,"only_first_frame",only_first_frame,"average_feature",average_feature,"test_mode",test_mode)

def get_parser(): # 配置参数
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--choose",
    )
    parser.add_argument(
        "--ltr_name",
    )
    parser.add_argument(
        "--datastore_dir",
    )
    parser.add_argument(
        "--test_dir",
    )
    parser.add_argument(
        "--policy",
    )
    return parser

def get_matching_set(ref_feature_paths):  # from knn-vc 获取特征集（原始代码是将tensor进行了级联，但此处修改为了返回list）
    feats = []
    count=0
    max_len=0
    if average_feature:
        for p in ref_feature_paths:
            if "B3" in os.path.basename(p):
                break
            f=open(p,'rb')
            feat1=pkl.load(f).detach().cpu().numpy()
            f.close()
            f=open(p.replace("B1","B3"),'rb')
            feat2=pkl.load(f).detach().cpu().numpy()
            f.close()
            if feat1.shape[0]<feat2.shape[0]:
                feat1=np.pad(feat1, ((0,feat2.shape[0]-feat1.shape[0]),(0,0)),'constant',constant_values=0)
            elif feat1.shape[0]>feat2.shape[0]:
                feat2=np.pad(feat2, ((0,feat1.shape[0]-feat2.shape[0]),(0,0)),'constant',constant_values=0)
            feat=np.mean([feat1,feat2],axis=0)
            feats.append(feat)
            tmp=feats[-1].shape[0]
            if tmp>max_len:
                max_len=tmp
            count+=1
    else:
        for p in ref_feature_paths:
            if p.endswith(".pkl"):
                f=open(p,'rb')
                if is_tensor:
                    feats.append(pkl.load(f)) 
                else:
                    feats.append(pkl.load(f).detach().cpu().numpy()) 
                f.close()
                if is_tensor:
                    tmp=feats[-1].size()[0] 
                else:
                    tmp=feats[-1].shape[0]
            elif p.endswith(".wav"):
                y,sr=librosa.load(p)
                tmp=y.shape[0]
                if is_tensor:
                    feats.append(torch.from_numpy(y).cuda())
                else:
                    feats.append(y)
            if tmp>max_len:
                max_len=tmp
            count+=1
    return feats,max_len    

def my_pad(features,max_len,dim,reshape=True): # 将特征进行pad（faiss需要使用）
    batch_size=len(features) 
    if is_tensor:
        pad_features = torch.zeros(batch_size, max_len, dim)
    else:
        pad_features = np.zeros((batch_size, max_len, dim))
    for x in range(batch_size):
        feature=features[x]
        if is_tensor:
            pad_features[x].narrow(0, 0, feature.size(0)).copy_(feature)
        else:
            pad_features[x][:feature.shape[0]]=feature
    if reshape:
        pad_features=pad_features.reshape(batch_size,max_len*dim)
    return pad_features

def test_ssl_and_knn(source_feature,ref_features): # 逐个与参考特征计算余弦相似度，获取最相似的数据索引
    dists=[]
    count=0
    for ref_feature in ref_features:
        if ref_feature.all()==None:
            dists.append(0)
            continue
        if not only_first_frame:
            if is_tensor:
                if len(ref_feature.size())==2:
                    ref_feature=ref_feature.transpose(0,1).cuda()
                else:
                    ref_feature=ref_feature.unsqueeze(0)
                s=source_feature.size()[1]
                r=ref_feature.size()[1]
            else:
                ref_feature=ref_feature.T
                s=source_feature.shape[1]
                r=ref_feature.shape[1]
            # 填充:
            if s<r:  
                if is_tensor:
                    source_feature=torch.nn.functional.pad(source_feature,pad=(r-s,0))
                else:
                    pad_feature=np.zeros((source_feature.shape[0],r))
                    pad_feature[:,:s]=source_feature
                    source_feature=pad_feature
            elif s>r:
                if is_tensor:
                    ref_feature=torch.nn.functional.pad(ref_feature,pad=(s-r,0))
                else:
                    pad_feature=np.zeros((ref_feature.shape[0],s))
                    pad_feature[:,:r]=ref_feature
                    ref_feature=pad_feature
            if is_tensor:
                tmp=torch.cosine_similarity(source_feature,ref_feature) # 求余弦相似度是越大越好
            else:
                tmp=torch.cosine_similarity(torch.from_numpy(source_feature),torch.from_numpy(ref_feature)) # 求余弦相似度是越大越好
        else:
            tmp=torch.cosine_similarity(torch.from_numpy(source_feature[np.newaxis,:]),torch.from_numpy(ref_feature[np.newaxis,:])) # 求余弦相似度是越大越好
        dists.append(torch.sum(tmp).cpu())
        count+=1
    arr = np.array(dists)
    best=arr.argsort()[-topk:][::-1]
    return best

def create_index(datas_embedding,choose=1): # 索引创建
    if choose==1:# 暴力检索的方法FlatL2，L2代表构建的index采用的相似度度量方法为L2范数，即欧氏距离
        index = faiss.IndexFlatL2(datas_embedding.shape[1]) # 传入一个向量的维度，创建一个空的索引
    elif choose==2:# 建立Inner product索引, 实际是余弦距离
        index = faiss.IndexFlatIP(datas_embedding.shape[1]) 
    elif choose==3: # 余弦相似度
        faiss.normalize_L2(datas_embedding) # 需要先经过L2归一化
        index = faiss.IndexFlatIP(datas_embedding.shape[1])
    index.add(datas_embedding) # 把向量数据加入索引
    return index

def data_recall(faiss_index, query_embedding, top_k): # 搜索
    Distance, Index = faiss_index.search(query_embedding, top_k)
    return Index, Distance

def test_knn(pkl_dir,test_dir,ltr_path,ref_ltr_paths,test_case,dim,faiss_choose=1,ltr_name=None,policy=None): # 测试knn的主函数
    # 读取datastore和测试数据的特证文件路径
    pkl_files=glob(pkl_dir)
    pkl_files = [f for f in pkl_files if not "UW" in os.path.basename(f)] # 去除UW
    if average_feature:
        pkl_files.sort(key=lambda x: os.path.basename(x))
    testpkl_files=glob(test_dir)
    # 样本真值（label）：wavname: ltr
    ltr_dic={} 
    with open(ltr_path,'r') as f:
        for line in f:
            paras=line.replace("\n","").split(" ")
            ltr_dic[paras[0].replace(".wav","")]=paras[1]
    # 参考：wavname: ltr
    ref_ltr_dic={} 
    for tmp in ref_ltr_paths:
        with open(tmp,'r') as f:
            for line in f:
                paras=line.replace("\n","").split(" ")
                ref_ltr_dic[paras[0].replace(".wav","")]=paras[1]
    # 加载参考特征：
    if not only_first_frame:
        ref_features,max_len=get_matching_set(pkl_files) 
    else:
        ref_features=get_matching_set_ff(pkl_files)
    count=0
    wer=0
    if test_case==1: # ref_features=my_pad(ref_features,max_len,dim=dim,reshape=False) 
        print("policy：",policy,"ltr_name：",ltr_name)
        for source_feature_path in testpkl_files:
            filename=os.path.basename(source_feature_path).replace(feature_suffix,"")
            print("filename：",filename)
            if source_feature_path.endswith(".pkl"):
                f=open(source_feature_path,'rb')
                if not only_first_frame:
                    if is_tensor:
                        source_feature=pkl.load(f) # source_feature=pkl.load(f).transpose(0,1).detach().numpy()
                        source_feature=source_feature.transpose(0,1)
                    else:
                        source_feature=pkl.load(f).detach().cpu().numpy().T
                else:
                    if is_tensor:
                        source_feature=pkl.load(f) # source_feature=pkl.load(f).transpose(0,1).detach().numpy()
                        source_feature=source_feature[0]
                    else:
                        source_feature=pkl.load(f).detach().cpu().numpy()[0]
                f.close()
            elif source_feature_path.endswith(".wav"):
                source_feature,sr=librosa.load(source_feature_path)
                source_feature=source_feature[np.newaxis,:]
            index=test_ssl_and_knn(source_feature,ref_features) # [128,4] 
            # print("index：",index)  
            label=ltr_dic[filename].lower()
            print("label:",label) # √
            count+=1
            # if len(index)==0:
            #     continue
            if top==1:
                ref_name=os.path.basename(pkl_files[index[0]]).replace(feature_suffix,"")
                pred=ref_ltr_dic[ref_name].lower() 
                print("pred:",ref_name," ",pred)
            elif top==3:
                tmp_pred_dic={}
                ref_name=None
                for tmp in index[:3]:
                    if tmp not in tmp_pred_dic:
                        tmp_pred_dic[tmp]=1
                    else: # 说明有重复的了
                        ref_name=os.path.basename(pkl_files[tmp]).replace(feature_suffix,"") 
                        break
                if not ref_name: # 如果没有重复的，则取第一个
                    ref_name=os.path.basename(pkl_files[index[0]]).replace(feature_suffix,"")
                pred=ref_ltr_dic[ref_name].lower()
                print("pred:",ref_name," ",pred)
            if pred!=label:
                wer+=1  
                print("相似的top5数据是：",end="")
                for tmp in index:
                    ref_name=os.path.basename(pkl_files[tmp]).replace(feature_suffix,"")
                    print("pred:",ref_name,"(",ref_ltr_dic[ref_name].lower(),end=") ")
                print()
    elif test_case==2:
        print("policy：",policy,"ltr_name：",ltr_name)
        if only_first_frame:
            pass
        elif policy=="4" or policy=="6": # 此处仅针对单说话人数据作为datastore的情况
            if ltr_name=="M09":
                max_len=261 
            elif ltr_name=="M07":
                max_len=653 
            elif ltr_name=="F02":
                max_len=490 
            elif ltr_name=="M01":
                max_len=2846 
        if not only_first_frame:
            global my_dim
            ref_features=my_pad(ref_features,max_len,dim=my_dim) # [batch, maxlen*dim]
        if is_tensor:
            faiss_index = create_index(ref_features.detach().numpy(),faiss_choose) 
        else:
            ref_features=ref_features.astype('float32') # 解决报错TypeError: in method 'fvec_renorm_L2', argument 3 of type 'float *'
            faiss_index = create_index(ref_features,faiss_choose) 
        count=0        
        for source_feature_path in testpkl_files:
            filename=os.path.basename(source_feature_path).replace(feature_suffix,"")
            print("filename：",filename)
            label=ltr_dic[filename].lower()
            print("label：",label)
            if not only_first_frame:
                if is_tensor:
                    pad_feature = torch.zeros(1,max_len, dim) # 构音障碍语音一般比正常语音长，所以maxlen可能有问题。
                else:
                    pad_feature = np.zeros((1,max_len, dim))
            if feature_suffix==".pkl":
                f=open(source_feature_path,'rb') 
                if not only_first_frame:
                    if is_tensor:
                        source_feature=pkl.load(f)
                    else:
                        source_feature=pkl.load(f).detach().cpu().numpy()
                else:
                    if is_tensor:
                        source_feature=pkl.load(f)[0]
                    else:
                        source_feature=pkl.load(f).detach().cpu().numpy()[0]
                    query=source_feature[np.newaxis,:]
                f.close()
            if not only_first_frame:
                try:
                    if is_tensor:
                        pad_feature[0].narrow(0, 0, source_feature.size(0)).copy_(source_feature)
                    else:
                        pad_feature[0][:source_feature.shape[0]]=source_feature
                except:
                    print("error")
                    print(source_feature.shape) 
                    exit()
                # start_time=time.time()
                if is_tensor:
                    query=pad_feature.reshape(1,-1).detach().numpy()
                else:
                    query=pad_feature.reshape(1,-1).astype('float32')
            if faiss_choose==3:
                faiss.normalize_L2(query) # 该函数是在原始数据上做修改，没有返回值
            sim_data_Index = data_recall(faiss_index,query,topk) 
            # print("cost time：",time.time()-start_time) # 挺快的，2s以内
            if top==1:
                ref_name=os.path.basename(pkl_files[sim_data_Index[0][0]]).replace(feature_suffix,"")
                pred=ref_ltr_dic[ref_name].lower()
                print("pred：",ref_name," ",pred)
            elif top==3:
                index=sim_data_Index[0]
                tmp_pred_dic={}
                ref_name=None
                for tmp in index[:3]:
                    if tmp not in tmp_pred_dic:
                        tmp_pred_dic[tmp]=1
                    else: # 说明有重复的了
                        ref_name=os.path.basename(pkl_files[tmp]).replace(feature_suffix,"") 
                        break
                if not ref_name: # 如果没有重复的，则取第一个
                    ref_name=os.path.basename(pkl_files[index[0]]).replace(feature_suffix,"")
                pred=ref_ltr_dic[ref_name].lower()
                print("pred:",ref_name," ",pred)
            count+=1 
            if count%100==0:
                print(count)
            if pred!=label:
                wer+=1  
                print("相似的top数据是：",end="")
                for index in sim_data_Index[0]: 
                    ref_name=os.path.basename(pkl_files[index]).replace(feature_suffix,"")
                    print("pred:",ref_name,"(",ref_ltr_dic[ref_name].lower(),end=") ")
                print()
    print(wer)
    print(count)
    print("success!")   
 
def only_get_error(origin_path,result_path):
    count=0
    results=[]
    with open(origin_path,"r") as f:
        for line in f:
            if count==0: # filename
                results.append(line)
                count+=1
            elif count==1: # label
                label=line.replace("\n","").split(": ")[1]
                count+=1
                results.append(line)
            elif count==2: # top10 输出
                tmp="相似的top10数据是："+line.replace("pred: "," ")
                count+=1
            elif count==3: # pred
                pred=line.replace("\n","").split("   ")[1]
                results.append(line)
                if label!=pred:
                    results.append(tmp)
                count=0
    with open(result_path,"w") as f:
        f.writelines(results)

def get_matching_set_ff(ref_feature_paths): # 仅使用第一帧特征相关函数
    feats = []
    count=0
    if average_feature:
        for p in ref_feature_paths:
            if "B3" in os.path.basename(p):
                break
            f=open(p,'rb')
            feat1=pkl.load(f).detach().cpu().numpy()[0]
            f.close()
            f=open(p.replace("B1","B3"),'rb')
            feat2=pkl.load(f).detach().cpu().numpy()[0]
            f.close()
            feat=np.mean([feat1,feat2],axis=0)
            feats.append(feat)
            count+=1
    else:
        for p in ref_feature_paths: 
            f=open(p,'rb')
            feats.append(pkl.load(f).detach().cpu().numpy()[0]) 
            f.close()
            count+=1
    return np.array(feats)

if __name__=="__main__":
    parser = get_parser()
    args = parser.parse_args()
    choose=int(args.choose) # 得到的是字符串
    ltr_name=args.ltr_name
    if choose==1: # 测试自己写的计算余弦相似度的knn或faiss中的knn
        print(ltr_name)
        # 针对base model的
        if test_mode=="base_model":# 只用M5数据
            pkl_dir=os.path.join(args.datastore_dir,ltr_name+"*M5.pkl")
            test_dir=os.path.join(args.test_dir,ltr_name+"*.pkl")
        else:
            pkl_dir=os.path.join(args.datastore_dir,"*M5*.pkl")
            test_dir=os.path.join(args.test_dir,"*.pkl")
            print("len(pkl_dir)：",len(glob(pkl_dir)))
            print("len(test_dir)：",len(glob(test_dir)))
        policy=args.policy
        print("policy：",policy)
        dim=my_dim
        if policy=="1":
            test_case=1
            faiss_choose=1 # 1：L2距离，2：余弦距离，3：余弦相似度
        elif policy=="4":
            test_case=2
            faiss_choose=1 
        elif policy=="5":
            test_case=2
            faiss_choose=2
        elif policy=="6":
            test_case=2
            faiss_choose=3
        ref_ltr_paths=["ltr_withwavname/train.ltr","valid.ltr"] # 注意 设置支持集转录路径，可以是多个文件，每一行是"文件名 转录" 如: F04_B2_UW3_M2.wav NUREMBERG
        ltr_path="ltr_withwavname/all/test.ltr" 
        test_knn(pkl_dir,test_dir,ltr_path,ref_ltr_paths,test_case,dim,faiss_choose,ltr_name=ltr_name,policy=policy)
    elif choose==2: # 处理输出的文件
        origin_path=""
        result_path=""
        only_get_error(origin_path,result_path)
    print("success!")