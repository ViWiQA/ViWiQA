python -m pyserini.index.lucene \
    --collection JsonCollection \
    --input corpora/wiki_vn_eol_jsonl \
    --language vi \
    --index indexes/wiki_vn_eol_index \
    --generator DefaultLuceneDocumentGenerator \
    --threads 10 \
    --storePositions --storeDocvectors --storeRaw
