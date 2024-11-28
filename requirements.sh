pip3 install torch torchvision torchaudio
pip install hydra-core
pip install librosa==0.9.2 
pip install bitarray==2.6.0
pip install tqdm
pip install sacrebleu -i https://pypi.tuna.tsinghua.edu.cn/simple
mkdir tmp
cd tmp
git clone https://github.com/facebookresearch/fairseq.git
# todo
cd fairseq
pip install ./
