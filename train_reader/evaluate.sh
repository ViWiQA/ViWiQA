# Path to the model to evaluate. For example: models/xlmr-large-2
MODEL_PATH=path-to-the-model-to-evaluate
GPU_IDS=0

TRAIN_PATH=dataset/train_hf_viquad.json
TEST_PATH=dataset/test_hf_viquad.json

OUTPUT_DIR=predict

CUDA_VISIBLE_DEVICES=$GPU_IDS python run_qa.py \
    --output_dir $OUTPUT_DIR \
    --model_name_or_path $MODEL_PATH \
    --do_eval \
    --train_file $TRAIN_PATH \
    --validation_file $TEST_PATH \
    --max_seq_length 400 \
    --max_answer_length 500 \
    --pad_to_max_length false \
    --group_by_length true

python squad_eval.py origin_test_ViQuAD.json predict/eval_predictions.json
