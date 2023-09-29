MODEL_CHECKPOINT="path/to/finetune_checkpoint.pt"
CORPUS_PATH="mdr_vn_data/wiki_corpus.jsonl"
SAVE_PATH="encoded_corpus/xlm-roberta-base-encoded"

CUDA_VISIBLE_DEVICES=1 python scripts/encode_corpus.py \
    --do_predict \
    --predict_batch_size 250 \
    --model_name xlm-roberta-base \
    --predict_file ${CORPUS_PATH} \
    --init_checkpoint ${MODEL_CHECKPOINT} \
    --embed_save_path ${SAVE_PATH} \
    --max_c_len 300 \
    --num_workers 20
