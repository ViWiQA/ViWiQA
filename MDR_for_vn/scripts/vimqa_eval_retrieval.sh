SAVE_RETRIEVAL_FOR_QA="encoded_corpus/xlm-roberta-base-encoded/retrieval_result/test_retrieval_duplicate.json"

CUDA_VISIBLE_DEVICES=0 python scripts/eval/eval_mhop_retrieval.py \
    vimqa_scripts/vimqa_test_duplicate_qas_val.jsonl \
    encoded_corpus/xlm-roberta-base-encoded.npy \
    encoded_corpus/xlm-roberta-base-encoded/id2doc.json \
    checkpoint_q_best.pt \
    --batch-size 16 \
    --beam-size 1 \
    --topk 1 \
    --shared-encoder \
    --model-name xlm-roberta-base \
    --gpu \
    --save-path ${SAVE_RETRIEVAL_FOR_QA}
