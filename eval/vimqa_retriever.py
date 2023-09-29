from collections import namedtuple
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
from tqdm import tqdm
from eval_utils import has_answers

device = "cuda" if torch.cuda.is_available() else "cpu"
RETRIEVAL_CROSS_ENC_NAME = "path/to/ce_model"
QA_NAME = "path/to/qa_model"
LUCENE_INDEX = "indexes/wiki_vn_index"
WIKI_GRAPH = "path/to/wiki_graph/graph.json"
PARAGRAPHS_FILE = "paragraphs.jsonl"

retrieve_ce_tokenizer = None
retrieve_ce_model = None
question_answerer: QuestionAnsweringPipeline = None
searcher: LuceneSearcher = None
wiki_graph: dict = None
title2passages: dict = None

Passage = namedtuple("Passage", ["title", "content"])


def init_wiki_graph():
    global wiki_graph
    global title2passages
    with open(WIKI_GRAPH) as f:
        wiki_graph = json.load(f)

    title2passages = {}
    with open(PARAGRAPHS_FILE) as f:
        for line in tqdm(f):
            para_info = json.loads(line)
            title = para_info["title"]
            paras = para_info["paragraphs"]
            passages = [Passage(title, para) for para in paras]
            title2passages[title.lower()] = passages


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
    init_wiki_graph()


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
        out = model(**inp).logits.squeeze().tolist()
    return out


def _rerank(question: str, passages: List[Passage], tokenizer, model):
    ds = QuestionContextDataset(question, [x.content for x in passages])
    loader = DataLoader(ds, batch_size=50)
    score = []
    for s1, s2 in loader:
        pred = _predict_relevance_score(s1, s2, tokenizer=tokenizer, model=model)
        if type(pred) == float:
            score.append(pred)
        else:
            score.extend(pred)
    assert len(passages) == len(score)
    sort = sorted(list(zip(passages, score)), key=lambda x: x[1], reverse=True)
    return [x[0] for x in sort]


def retrieve_bm25(query: str, top_k: int):
    hits = searcher.search(query, k=top_k)
    ret = []
    for x in hits:
        retr = json.loads(x.raw)
        ret.append(Passage(retr["title"], retr["contents"]))
    return ret


def retrieve_ce(question, top_k):
    passages = retrieve_bm25(question, top_k)
    return _rerank(question, passages, retrieve_ce_tokenizer, retrieve_ce_model)


def rerank_rider(passages, preds, n_pred=30):
    has_ans = []
    not_has_ans = []
    for p in passages:
        has_answer_flag = has_answers(set(preds[:n_pred]), p.content)
        if has_answer_flag:
            has_ans.append(p)
        else:
            not_has_ans.append(p)
    ret = has_ans + not_has_ans
    assert len(ret) == len(passages)
    return ret


def get_qa_preds(question, top_k):
    global question_answerer
    passages = retrieve_bm25(question, top_k)
    preds = question_answerer(
        question=[question] * len(passages),
        context=[x.content for x in passages],
        max_answer_len=500,
        max_seq_len=512,
        max_question_len=128,
        device=0,
    )
    preds.sort(key=lambda x: x["score"], reverse=True)
    preds = [x["answer"] for x in preds]
    return preds, passages


def retrieve_rider(question, top_k):
    preds, passages = get_qa_preds(question, top_k)
    return rerank_rider(passages, preds)


def retrieve_ce_graph(question, top_k):
    global wiki_graph
    global title2passages
    passages = retrieve_ce(question, top_k)
    for p in passages:
        if p.title.lower() in wiki_graph:
            top_passage = p
            break
    link_titles = wiki_graph[top_passage.title.lower()]
    link_passages = []
    for title in link_titles:
        if title in title2passages:
            link_passages.extend(title2passages[title][:2])
    if len(link_passages) == 0:
        return passages[:2]
    rerank_link_passages = _rerank(
        question, link_passages, retrieve_ce_tokenizer, retrieve_ce_model
    )
    top_link_passage = rerank_link_passages[0]

    return [top_passage, top_link_passage]


def retrieve_ce_rider(question, top_k):
    global retrieve_ce_tokenizer
    global retrieve_ce_model

    preds, _ = get_qa_preds(question, top_k)
    passages = retrieve_ce(question, top_k)
    return rerank_rider(passages, preds)
