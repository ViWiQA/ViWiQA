RUN_ID=1
TRAIN_DATA_PATH="mdr_vn_data/vimqa_train_with_neg_bridge.json"
DEV_DATA_PATH="mdr_vn_data/vimqa_dev_with_neg_bridge.json"
CHECKPOINT_PT="path/to/checkpoint.pt"

CUDA_VISIBLE_DEVICES=0,2 python scripts/train_momentum.py \
    --do_train \
    --prefix ${RUN_ID} \
    --predict_batch_size 500 \
    --model_name xlm-roberta-base \
    --train_batch_size 32 \
    --learning_rate 2e-5 \
    --train_file ${TRAIN_DATA_PATH} \
    --predict_file ${DEV_DATA_PATH} \
    --seed 16 \
    --eval-period -1 \
    --max_c_len 300 \
    --max_q_len 70 \
    --max_q_sp_len 350 \
    --momentum \
    --k 76800 \
    --m 0.999 \
    --temperature 1 \
    --init-retriever ${CHECKPOINT_PT}
