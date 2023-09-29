RUN_ID=2
TRAIN_DATA_PATH="mdr_vn_data/vimqa_train_with_neg_bridge.json"
DEV_DATA_PATH="mdr_vn_data/vimqa_dev_with_neg_bridge.json"

CUDA_VISIBLE_DEVICES=0 python scripts/train_mhop.py \
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
    --shared-encoder \
    --warmup-ratio 0.1
