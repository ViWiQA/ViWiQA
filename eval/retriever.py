from typing import List
from pyserini.search.lucene import LuceneSearcher
import json
import torch
from torch.utils.data import DataLoader, Dataset
from transformers import (
    AutoTokenizer,
    AutoModelForSequenceClassification,
    pipeline,
    QuestionAnsweringPipeline,
)
from eval_utils import has_answers


device = "cuda" if torch.cuda.is_available() else "cpu"
RETRIEVAL_CROSS_ENC_NAME = "path/to/ce_model"
QA_NAME = "path/to/qa_model"
LUCENE_INDEX = "indexes/wiki_vn_index"

retrieve_ce_tokenizer = None
retrieve_ce_model = None
question_answerer: QuestionAnsweringPipeline = None
searcher: LuceneSearcher = None


def init_ce():
    global retrieve_ce_tokenizer
    global retrieve_ce_model
    retrieve_ce_tokenizer = AutoTokenizer.from_pretrained(RETRIEVAL_CROSS_ENC_NAME)
    retrieve_ce_model = AutoModelForSequenceClassification.from_pretrained(
        RETRIEVAL_CROSS_ENC_NAME
    )
    retrieve_ce_model.to(device)


def init_qa():
    global question_answerer
    question_answerer = pipeline(
        "question-answering",
        model=QA_NAME,
        device=0 if torch.cuda.is_available() else -1,
    )


def init_bm25():
    global searcher
    searcher = LuceneSearcher(LUCENE_INDEX)
    searcher.set_language("vi")


def init_all():
    init_ce()
    init_bm25()
    init_qa()


class QuestionContextDataset(Dataset):
    def __init__(self, question: str, contexts: List[str]):
        self.question = question
        self.contexts = contexts

    def __getitem__(self, i):
        return self.question, self.contexts[i]

    def __len__(self):
        return len(self.contexts)


def _predict_relevance_score(s1, s2, tokenizer, model):
    with torch.no_grad():
        inp = tokenizer(
            s1, s2, return_tensors="pt", padding=True, truncation="longest_first"
        ).to(device)
        out = model(**inp).logits[:, 1].tolist()
    return out


def _rerank(question, contexts, tokenizer, model):
    ds = QuestionContextDataset(question, contexts)
    loader = DataLoader(ds, batch_size=50)
    score = []
    for s1, s2 in loader:
        pred = _predict_relevance_score(s1, s2, tokenizer=tokenizer, model=model)
        score.extend(pred)
    sort = sorted(list(zip(contexts, score)), key=lambda x: x[1], reverse=True)
    return [x[0] for x in sort]


def retrieve_bm25(query, top_k, include_title=False):
    hits = searcher.search(query, k=top_k)
    ret = []
    for x in hits:
        retr = json.loads(x.raw)
        if include_title:
            content = retr["title"] + ": " + retr["contents"]
        else:
            content = retr["contents"]
        ret.append(content)
    return ret


def retrieve_ce(question, top_k):
    contexts = retrieve_bm25(question, top_k)
    return _rerank(question, contexts, retrieve_ce_tokenizer, retrieve_ce_model)


def rerank_rider(contexts, preds, n_pred=30):
    has_ans = []
    not_has_ans = []
    for ctx in contexts:
        has_answer_flag = has_answers(set(preds[:n_pred]), ctx)
        if has_answer_flag:
            has_ans.append(ctx)
        else:
            not_has_ans.append(ctx)
    ret = has_ans + not_has_ans
    assert len(ret) == len(contexts)
    return ret


def get_qa_preds(question, top_k):
    global question_answerer
    contexts = retrieve_bm25(question, top_k)
    preds = question_answerer(
        question=[question] * len(contexts),
        context=contexts,
        max_answer_len=500,
        max_seq_len=512,
        max_question_len=128,
        device=0,
    )
    preds.sort(key=lambda x: x["score"], reverse=True)
    preds = [x["answer"] for x in preds]
    return preds, contexts


def retrieve_rider(question, top_k):
    preds, contexts = get_qa_preds(question, top_k)
    return rerank_rider(contexts, preds)


def retrieve_ce_rider(question, top_k):
    global retrieve_ce_tokenizer
    global retrieve_ce_model
    preds, _ = get_qa_preds(question, top_k)
    contexts = retrieve_ce(question, top_k)
    return rerank_rider(contexts, preds)
