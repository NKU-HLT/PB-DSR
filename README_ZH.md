# PB-DSR #
这是我们的论文“Enhancing Dysarthric Speech Recognition for Unseen Speakers via Prototype-Based Adaptation”的代码实现。

## 环境 ##
```
python==3.9
pip install -r requirements.txt
```

## 使用 ##
### 数据配置 ###
1. 下载[UASPEECH](https://www.isca-archive.org/interspeech_2008/kim08c_interspeech.pdf)数据集。

2. 参考我们的论文中“Dataset”小节对数据进行划分。

### 训练DSR模型 ###
1. 修改配置文件 ```examples/hubert/config/finetune/base_10h.yaml``` 中的参数，尤其是标注了“**注意**”的地方。

2. 设置Hubert路径：在文件 ```fairseq/models/hubert/hubert_asr.py``` 中设置 ```cfg.w2v_path``` 的值（344行代码）。

3. 训练指令：
    ```
    python fairseq_cli/hydra_train.py --config-dir [PB-DSR DIR]/examples/hubert/config/finetune --config-name base_10h task.data=[YOUR data_dir] task.label_dir=[YOUR label_dir] model.w2v_path=[Hubert path]
    ```

4. 恢复训练或微调：设置 ```fairseq/trainer.py``` 中 ```filename``` 的值（457行代码），设置为加载训练的模型路径。

#### 增加监督对比学习损失训练时的额外配置 ####
1. 修改 ```fairseq/tasks/hubert_pretraining.py``` 文件中的 ```index=3``` 。(仅使用CTC损失训练时，```index=0``` )。

2. 设置 ```wsy/hparam.py``` 文件中的 ```use_cl_loss``` 为 ```True```。（仅使用CTC损失训练时，```use_cl_loss=False```）。

### 使用DSR模型进行推理 ###
1. 修改 ```wsy/config/config.yaml``` 中的参数，同样注意标注了“**注意**”的地方。

2. 推理指令：
    ```
    python examples/speech_recognition/new/infer.py
    ```

### PB-DSR ###
#### 支持集抽取特征 ####
1. 设置 ```wsy/hparam.py``` 中的 ```prepare_datastore=True``` 以及保存特征的目录 ```o_datastore_dir``` 。
2. 执行 ```使用DSR模型进行推理``` 处的操作。

#### 测试集抽取特征 ####
1. 设置```wsy/hparam.py```中的```prepare_datastore=True```以及设置保存特征的目录```o_datastore_dir``` 。
2. 执行```使用DSR模型进行推理```处的操作。

#### 基于原型的分类 ####
1. 设置 ```wsy/prototype_test.py``` 里标注了 ```注意``` 的地方。
2. 执行以下指令：可以修改 ```speaker_name``` ， ```result_dir``` ；```datastore_dir```和```test_dir```是上方保存支持集和测试集特征的目录
    ```
    speaker_name=M04 
    datastore_dir=datastore_dir_path
    test_dir=test_dir_path
    policy=4
    result_dir=result_dir_path
    python wsy/prototype_test.py --choose 1 --ltr_name $speaker_name --datastore_dir $datastore_dir --test_dir $test_dir --policy $policy >> result_dir/${speaker_name}_$policy.txt
    ```

## 注意 ##
1. 代码中的注释以中文为主。

2. 如有技术问题，欢迎提issue，我们尽可能解答疑问。

3. 如果本仓库对您的学习有帮助，可以star⭐鼓励一下我们，谢谢：）

## 致谢 ##
本工作基于[Fairseq](https://github.com/facebookresearch/fairseq)进行构建，我们修复了原始仓库中的一些bug，使得可以基于Hubert构建构音障碍语音识别（DSR）模型。

## 引用 ##
```
@article{wang2024enhancing,
  title={Enhancing Dysarthric Speech Recognition for Unseen Speakers via Prototype-Based Adaptation},
  author={Wang, Shiyao and Zhao, Shiwan and Zhou, Jiaming and Kong, Aobo and Qin, Yong},
  journal={arXiv preprint arXiv:2407.18461},
  year={2024}
}
```