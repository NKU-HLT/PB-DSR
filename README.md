# PB-DSR #
[切换中文版](./README_ZH.md)

This is code of our paper: "[**Enhancing Dysarthric Speech Recognition for Unseen Speakers via Prototype-Based Adaptation**](https://arxiv.org/abs/2407.18461#)" (accepted by Interspeech 2024)

We modified and adapted the framework to the Dysarthria Wake-Up Word Spotting task, and won second place in the SLT 2024 Low-Resource Dysarthria Wake-Up Word Spotting (LRDWWS) Challenge. For more implementation details of our system, please refer to our paper [PB-LRDWWS SYSTEM FOR THE SLT 2024 LOW-RESOURCE DYSARTHRIA WAKE-UP WORD SPOTTING CHALLENGE](https://arxiv.org/abs/2409.04799).

## Create Environment ##
```
conda create -n PBDSR python=3.9
conda activate PBDSR
bash requirements.sh
```

## Paper reproduction ##
### Data configuration ###
1. Download[UASPEECH](https://www.isca-archive.org/interspeech_2008/kim08c_interspeech.pdf) dataset.

2. Refer to the "Dataset" section of our paper to partition the data, and configure metadata (*. tsv, *. ltr, and dictionary) according to the [Fairseq](https://github.com/facebookresearch/fairseq) format.

### Training Dysarthric Speech Recognition (DSR) model ###
1. Modify the parameters in the ```examples/hubert/config/finetune/base_10h.yaml``` configuration file, especially those marked with "**注意**".

2. Set Hubert path: Set the value of ```cfg.w2v_path``` in the file ```fairseq/models/hubert/hubert_asr.py``` (344 lines of code).

3. Training instructions: 
    ```
    export WORK_DIR=[PB-DSR DIR]
    
    python fairseq_cli/hydra_train.py --config-dir [PB-DSR DIR]/examples/hubert/config/finetune --config-name base_10h task.data=[YOUR data_dir] task.label_dir=[YOUR label_dir] model.w2v_path=[Hubert path]
    ```
    Set ` [PB-DSR DIR] ` in the instruction as the code directory, ` [YOUR data-dir] ` and ` [YOUR label-dir] ` as the metadata directory, and ` [Hubert path] ` as the Hubert model path.

4. Resume training or fine-tuning: Set the value of ```filename``` in ```fairseq/trainer.py``` (457 lines of code) as the path to load the trained model.

#### Additional configuration for supervised contrastive learning loss training ####
1. Modify ```index=3``` in the ```fairseq/tasks/hubert_pretraining.py``` file. (When training only with CTC loss, ```index=0```).

2. Set the ```use_cl_loss``` in the ```wsy/hparam.py``` file to ```True```. When training only with CTC loss, ```use_cl_loss=False```.

### Using DSR model for inference ###
1. Modify the parameters in ```wsy/config/config.yaml```, and also pay attention to those marked with '**注意**'.
   
2. Inference instruction:
    ```
    python examples/speech_recognition/new/infer.py
    ```

### PB-DSR ###
#### Support set feature extraction ####
1. Set ```prepare_datastore=True``` and the directory ```o_datastore_dir``` where features are saved in ```wsy/hparam.py``` .
2. Execute the operation at ```Using DSR model for inference``` .

#### Extract features from the test set ####
1. Set ```prepare_datastore=True``` and the directory ```o_datastore_dir``` where features are saved in ```wsy/hparam.py``` .
2. Execute the operation at ```Using DSR model for inference``` .

#### Prototype-based classification ####
1. The places marked with ```注意``` in the ```wsy/prototype_test.py``` settings.
2. Execute the following command: You can modify ```speaker_name```, ```result_dir``` . ```datastore_dir``` and ```test_dir``` are the directories above where the support set and test set features are stored.
    ```
    speaker_name=M04 
    datastore_dir=datastore_dir_path
    test_dir=test_dir_path
    policy=4
    result_dir=result_dir_path
    python wsy/prototype_test.py --choose 1 --ltr_name $speaker_name --datastore_dir $datastore_dir --test_dir $test_dir --policy $policy >> result_dir/${speaker_name}_$policy.txt
    ```

## Note ##
1. The comments in the code are mainly in Chinese.
2. If you have any technical questions, please feel free to create a new issue and we will try our best to answer them.
3. If this repository is helpful for your learning, you can **star⭐** encourage us. Thank you：）

## Acknowledgement ##
This work is based on [Fairseq](https://github.com/facebookresearch/fairseq) repository and we have fixed some bugs in the original repository, allowing us to build Dysarthric Speech Recognition (DSR) model based on Hubert.

## Citation ##
```
@article{wang2024enhancing,
  title={Enhancing Dysarthric Speech Recognition for Unseen Speakers via Prototype-Based Adaptation},
  author={Wang, Shiyao and Zhao, Shiwan and Zhou, Jiaming and Kong, Aobo and Qin, Yong},
  journal={arXiv preprint arXiv:2407.18461},
  year={2024}
}
```