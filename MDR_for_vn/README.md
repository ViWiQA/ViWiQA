# MDR for Vietnamese
This directory is for training MDR model for Vietnamese

# Requirements
Clone the official [MDR repository](https://github.com/facebookresearch/multihop_dense_retrieval) and install the required packages in `requirements.txt`

# Data
Download the processed data `mdr_vn_data.zip`. The data includes the following files.
- vimqa_dev_with_neg_bridge.json
- vimqa_train_with_neg_bridge.json
- wiki_corpus.jsonl

# Training
The training scripts is similar to the training in section [Train models from scratch](https://github.com/facebookresearch/multihop_dense_retrieval#train-models-from-scratch).

Move the files in `scripts` to the cloned MDR repository.

## 1. Retriever training
Run the training script
```bash
bash vimqa_train.sh
```

After training, the model will be saved in the `logs` directory of MDR.

## 2. Finetune the question encoder with frozen memory bank
Set the `CHECKPOINT_PT` in `vimqa_finetune.sh` to the path of the trained model (from the previous step), then run the fine-tune script.
```bash
bash vimqa_finetune.sh
```
After fine-tuning, the model will be saved in the `logs` directory of MDR.

## 3. Encoding the corpus for retrieval
Set the `MODEL_CHECKPOINT` to the fine-tune model (in step 2) and specify the path to save the encode corpus (index) in `SAVE_PATH`, then run the script.
```bash
bash vimqa_encode_corpus.sh
```