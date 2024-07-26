# PB-DSR
This is code of our paper：“Enhancing Dysarthric Speech Recognition for Unseen Speakers via Prototype-Based Adaptation”

本工作基于fairseq仓库进行构建，修复了原始仓库中的一些bug，使得可以基于hubert构建构音障碍语音识别（DSR）模型。

# Environment
python==3.9
pip install -r requirements.txt

# Usage
## 训练
1. 将本目录下“/home/wangshiyao/wangshiyao_space/fairseq”替换为您的路径
2. 根据需要修改“examples/hubert/config/finetune/base_10h.yaml”中的参数，尤其注意标注了“注意”的地方，需要修改为您的实际路径或下载模型
3. 训练指令：python fairseq_cli/hydra_train.py --config-dir /home/wangshiyao/wangshiyao_space/fairseq/examples/hubert/config/finetune --config-name base_10h task.data=/home/wangshiyao/wangshiyao_space/fairseq/wsy/data/lrdwwk task.label_dir=/home/wangshiyao/wangshiyao_space/fairseq/wsy/data/lrdwwk model.w2v_path=/home/wangshiyao/wangshiyao_space/fairseq/wsy/dm/chinese-hubert-base.pt >> wsy/training_log/
(注意，指令中的路径需要修改为您的实际路径)

## 推理
1. 根据实际修改“wsy/config/config.yaml”中的参数，同样注意标注了“注意”的地方，需要修改为您的实际路径或下载模型
2. 推理指令：python examples/speech_recognition/new/infer.py

原型代码后续发布

## 注意 ##
1. 代码中的注释以中文为主
2. 如有技术问题，欢迎提issue，尽可能解答疑问。
3. 如果本仓库对您的学习有帮助，可以star鼓励一下，谢谢：）

# Citation
1. 论文：
    1. 论文目前已提交到arxiv，7.29会公开。欢迎引用
2. 感谢fairseq仓库