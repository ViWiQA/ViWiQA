EXPERIMENT_NAME=xlmr-base
INITIAL_MODEL=xlm-roberta-base
EVAL_STEP=200
GPU_IDS=1
TRAIN_DATA=data/train_ce.json
VALIDATION_DATA=data/dev_ce.json
OUTPUT_DIR=models

CUDA_VISIBLE_DEVICES=$GPU_IDS python run_glue.py \
    --output_dir $OUTPUT_DIR/$EXPERIMENT_NAME \
    --model_name_or_path $INITIAL_MODEL \
    --do_train \
    --train_file $TRAIN_DATA \
    --validation_file $VALIDATION_DATA \
    --max_seq_length 512 \
    --pad_to_max_length false \
    --group_by_length true \
    --num_train_epochs 10 \
    --learning_rate 2e-5 \
    --warmup_ratio 0.2 \
    --per_device_train_batch_size 32 \
    --per_device_eval_batch_size 32 \
    --evaluation_strategy steps \
    --eval_steps $EVAL_STEP \
    --save_steps $EVAL_STEP \
    --logging_strategy steps \
    --logging_steps $EVAL_STEP \
    --load_best_model_at_end \
    --report_to tensorboard
