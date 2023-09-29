import sys
import csv
from colbert import Searcher
import pandas as pd
import torch
import argparse
import faiss
import json
import numpy as np
from apex import amp
from transformers import AutoConfig, AutoTokenizer
from mdr.retrieval.models.mhop_retriever import RobertaRetriever
from mdr.retrieval.utils.utils import load_saved, move_to_cuda


INDEX_PATH_COLBERT = (
    "experiments/indexes/uit_viquad_colbert_xml_base_maxsteps200000_bsize8"
)
COLLECTIONS = "data_colbert/int_collections.tsv"
QUERY_ID = "1"

TOP_K = 10

PRETRAINED_MODEL = "xlm-roberta-base"
MODEL_PATH = "xlm-roberta-base-finetuned/checkpoint_q_best.pt"
CORPUS_DICT = (
    "multihop_dense_retrieval/encoded_corpus/xlm-roberta-base-encoded/id2doc.json"
)
INDEX_PATH_MDR = "multihop_dense_retrieval/encoded_corpus/xlm-roberta-base-encoded.npy"


def get_colbert_components():
    searcher = Searcher(index=INDEX_PATH_COLBERT)
    collections = {}
    with open(COLLECTIONS) as file:
        tsv_file = csv.reader(file, delimiter="\t")
        for line in tsv_file:
            collections[line[0]] = line[1]
    return searcher, collections


def colbert_search(searcher, collections, text, top_k=TOP_K):
    q = {QUERY_ID: text}
    ranking = searcher.search_all(q, k=top_k)
    results = [[collections[str(r[0])], r[2]] for r in ranking.data[QUERY_ID]]
    return colbert_to_dataframe(results)


def colbert_to_dataframe(results):
    passages = []
    scores = []
    for r in results:
        passages.append(r[0])
        scores.append(r[1])
    d = {"passage": passages, "score": scores}
    return pd.DataFrame(data=d)


def get_mdr_components():
    parser = argparse.ArgumentParser()
    parser.add_argument("--topk", type=int, default=3, help="topk paths")
    parser.add_argument("--max-q-len", type=int, default=70)
    parser.add_argument("--max-c-len", type=int, default=300)
    parser.add_argument("--max-q-sp-len", type=int, default=350)
    parser.add_argument(
        "--sp-pred", action="store_true", help="whether to predict sentence sp"
    )
    parser.add_argument(
        "--sp-weight", default=0, type=float, help="weight of the sp loss"
    )
    parser.add_argument("--max-ans-len", default=30, type=int)
    parser.add_argument("--save-prediction", default="", type=str)
    parser.add_argument("--index-gpu", default=-1, type=int)
    args = parser.parse_args()
    args.model_name = PRETRAINED_MODEL
    args.model_path = MODEL_PATH
    args.corpus_dict = CORPUS_DICT
    args.indexpath = INDEX_PATH_MDR

    device = "cuda" if torch.cuda.is_available() else "cpu"

    print("Initializing retrieval module...")
    bert_config = AutoConfig.from_pretrained(args.model_name)
    tokenizer = AutoTokenizer.from_pretrained(args.model_name)
    retriever = RobertaRetriever(bert_config, args)
    retriever = load_saved(retriever, args.model_path, exact=False)
    retriever.to(device)
    retriever = amp.initialize(retriever, opt_level="O1")
    retriever.eval()

    print("Loading index...")
    index = faiss.IndexFlatIP(768)
    xb = np.load(args.indexpath).astype("float32")
    index.add(xb)
    if args.index_gpu != -1:
        res = faiss.StandardGpuResources()
        index = faiss.index_cpu_to_gpu(res, args.index_gpu, index)

    print("Loading documents...")
    id2doc = json.load(open(args.corpus_dict))

    print("Index ready...")

    return retriever, index, id2doc, tokenizer, args


def mdr_search(retriever, index, id2doc, tokenizer, args, query, top_k=TOP_K):
    args.topk = top_k
    query = query[:-1] if query.endswith("?") else query
    with torch.no_grad():
        print("Retrieving")
        q_encodes = tokenizer.batch_encode_plus(
            [query],
            max_length=args.max_q_len,
            pad_to_max_length=True,
            return_tensors="pt",
        )
        q_encodes = move_to_cuda(dict(q_encodes))
        q_embeds = (
            retriever.encode_q(
                q_encodes["input_ids"],
                q_encodes["attention_mask"],
                q_encodes.get("token_type_ids", None),
            )
            .cpu()
            .numpy()
        )
        scores_1, docid_1 = index.search(q_embeds, args.topk)

        query_pairs = []  # for 2nd hop
        for _, doc_id in enumerate(docid_1[0]):
            doc = id2doc[str(doc_id)][1]
            if doc.strip() == "":
                # roberta tokenizer does not accept empty string as segment B
                doc = id2doc[str(doc_id)][0]
                scores_1[b_idx][_] = float("-inf")
            query_pairs.append((query, doc))

        q_sp_encodes = tokenizer.batch_encode_plus(
            query_pairs,
            max_length=args.max_q_sp_len,
            pad_to_max_length=True,
            return_tensors="pt",
        )
        q_sp_encodes = move_to_cuda(dict(q_sp_encodes))
        q_sp_embeds = (
            retriever.encode_q(
                q_sp_encodes["input_ids"],
                q_sp_encodes["attention_mask"],
                q_sp_encodes.get("token_type_ids", None),
            )
            .cpu()
            .numpy()
        )
        scores_2, docid_2 = index.search(q_sp_embeds, args.topk)

        scores_2 = scores_2.reshape(1, args.topk, args.topk)
        docid_2 = docid_2.reshape(1, args.topk, args.topk)
        path_scores = np.expand_dims(scores_1, axis=2) + scores_2
        search_scores = path_scores[0]
        ranked_pairs = np.vstack(
            np.unravel_index(
                np.argsort(search_scores.ravel())[::-1], (args.topk, args.topk)
            )
        ).transpose()
        chains = []
        topk_docs = {}
        for _ in range(args.topk):
            path_ids = ranked_pairs[_]
            doc1_id = str(docid_1[0, path_ids[0]])
            doc2_id = str(docid_2[0, path_ids[0], path_ids[1]])
            chains.append([id2doc[doc1_id], id2doc[doc2_id]])
            topk_docs[id2doc[doc1_id][0]] = id2doc[doc1_id][1]
            topk_docs[id2doc[doc2_id][0]] = id2doc[doc2_id][1]
    return mdr_to_dataframe(chains)


def mdr_to_dataframe(results):
    passages1 = []
    passages2 = []
    for r in results:
        passages1.append(r[0][1])
        passages2.append(r[1][1])
    d = {"passage1": passages1, "passage2": passages2}
    return pd.DataFrame(data=d)
