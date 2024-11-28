pip3 install torch torchvision torchaudio
pip install hydra-core
pip install librosa==0.9.2 
pip install bitarray==2.6.0
pip install tqdm
pip install sacrebleu
mkdir tmp
cd tmp
git clone https://github.com/facebookresearch/fairseq.git
cd fairseq
pip install pip==24.0
pip install omegaconf==2.0.5
pip install fairseq==0.12.2
pip install ./ -i https://pypi.tuna.tsinghua.edu.cn/simple
pip uninstall fairseq
cd ../../
mv README.md PBDSR-READMEmd
cp tmp/fairseq/README.md .
python setup.py build_ext --inplace