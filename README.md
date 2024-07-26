# PB-DSR 
[切换中文版](./README_ZH.md)

This is code of our paper: "**Enhancing Dysarthric Speech Recognition for Unseen Speakers via Prototype-Based Adaptation**" (accepted by Interspeech 2024)

# Environment
```
python==3.9
pip install -r requirements.txt
```

## Usage ##
### Data configuration ###
1. Download[UASPEECH](https://www.isca-archive.org/interspeech_2008/kim08c_interspeech.pdf) dataset.
2. Refer to the "Dataset" section of our paper to partition the data. 

### Training Dysarthric Speech Recognition(DSR) model ###
1. Replace the actual path of PB-DSR with "[PB-DSR DIR]" in the code of this repository.
2. Modify the parameters in the ```examples/hubert/config/finetune/base_10h.yaml``` configuration file, especially those marked with "**注意**", to your actual path.
3. Set hubert path: Set the value of ```cfg.w2v_path``` in the file ```fairseq/models/hubert/hubert_asr.py``` (344 lines of code).
4. Training instructions: Pay attention to changing the path in the instructions to your actual path
    ```
    python fairseq_cli/hydra_train.py --config-dir [PB-DSR DIR]/examples/hubert/config/finetune --config-name base_10h task.data=[YOUR data_dir] task.label_dir=[YOUR label_dir] model.w2v_path=[hubert path]
    ```
5. Resume training or fine-tuning: It is necessary to set the value of ```filename``` in ```fairseq/trainer.py``` (457 lines of code) as the path to load the trained model.

#### Additional configuration for supervised contrastive learning loss training ####
1. Modify ```index=3``` in the ```fairseq/tasks/hubert_pretraining.py``` file. (When training only with CTC loss, ```index=0```).
2. Set the ```use_cl-loss``` in the ```wsy/hparam.py``` file to ```True```. When training only with CTC loss, ```use_cl_loss=False```.

### Using DSR model for inference ###
1. Modify the parameters in ```wsy/config/config.yaml``` according to the actual situation, and also pay attention to the places marked with '**注意**', which need to be modified to your actual configuration path.
2. Inference instruction:
    ```
    python examples/speech_recognition/new/infer.py
    ```

### PB-DSR ###
We will update PB-DSR detailed operations and code on July 30, 2024

## Note ##
1. The comments in the code are mainly in Chinese.
2. If you have any technical questions, please feel free to raise them and we will try our best to answer them.
3. If this repository is helpful for your learning, you can **star0A67F42B.png** encourage us. Thank you：）

## Citation ##
1. Paper: The paper has been submitted to arxiv and will be made public on July 29, 2024.
2. This work is based on the [Fairseq](https://github.com/facebookresearch/fairseq) repository and fixes some bugs in the original repository, enabling the construction of a speech recognition (DSR) model for articulation disorders based on Hubert.