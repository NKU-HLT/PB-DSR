# PB-DSR

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

## citation 
