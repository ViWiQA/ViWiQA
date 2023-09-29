EXPERIMENT_NAME=xlmr-large
INITIAL_MODEL=deepset/xlm-roberta-large-squad2
EVAL_STEP=200
GPU_IDS=1

TRAIN_DATA=dataset/train_hf_viquad.json
VALIDATION_DATA=dataset/dev_hf_viquad.json
TEST_DATA=dataset/test_hf_viquad.json

OUTPUT_DIR=models

CUDA_VISIBLE_DEVICES=$GPU_IDS python run_qa.py \
    --output_dir $OUTPUT_DIR/$EXPERIMENT_NAME \
    --model_name_or_path $INITIAL_MODEL \
    --do_train \
    --do_predict \
    --train_file $TRAIN_DATA \
    --validation_file $VALIDATION_DATA \
    --test_file $TEST_DATA \
    --max_seq_length 512 \
    --max_answer_length 500 \
    --pad_to_max_length false \
    --group_by_length true \
    --num_train_epochs 5 \
    --learning_rate 1e-5 \
    --warmup_ratio 0.1 \
    --metric_for_best_model f1 \
    --per_device_train_batch_size 16 \
    --per_device_eval_batch_size 16 \
    --evaluation_strategy steps \
    --eval_steps $EVAL_STEP \
    --save_steps $EVAL_STEP \
    --logging_strategy steps \
    --logging_steps $EVAL_STEP \
    --load_best_model_at_end \
    --report_to tensorboard
