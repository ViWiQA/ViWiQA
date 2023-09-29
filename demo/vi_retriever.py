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


class QuestionContextDataset(Dataset):
    def __init__(self, question: str, contexts: List[str]):
        self.question = question
        self.contexts = contexts

    def __getitem__(self, i):
        return self.question, self.contexts[i]

    def __len__(self):
        return len(self.contexts)


class ViRetriever:
    def __init__(
        self, model_path, lucene_searcher=None, lucene_index=None, device="cuda"
    ):
        self.tokenizer = AutoTokenizer.from_pretrained(model_path)
        self.model = AutoModelForSequenceClassification.from_pretrained(model_path)
        self.model.to(device)
        self.device = device
        if lucene_searcher is not None:
            self.searcher = lucene_searcher
        elif lucene_index is not None:
            self.searcher = LuceneSearcher(lucene_index)
            self.searcher.set_language("vi")
        else:
            raise ValueError("Must provide lucene searcher or index")

    def _predict_relevance_score(self, s1, s2, tokenizer, model):
        with torch.no_grad():
            inp = tokenizer(
                s1, s2, return_tensors="pt", padding=True, truncation="longest_first"
            ).to(self.device)
            out = model(**inp).logits.squeeze().tolist()
        return out

    def _rerank(self, question, contexts):
        ds = QuestionContextDataset(question, contexts)
        loader = DataLoader(ds, batch_size=50)
        score = []
        for s1, s2 in loader:
            pred = self._predict_relevance_score(
                s1, s2, tokenizer=self.tokenizer, model=self.model
            )
            score.extend(pred)
        sort = sorted(list(zip(contexts, score)), key=lambda x: x[1], reverse=True)
        return [x[0] for x in sort]

    def retrieve(self, query, top_k=150):
        hits = self.searcher.search(query, k=top_k)
        contexts = [json.loads(x.raw)["contents"] for x in hits]
        return self._rerank(query, contexts)


class BM25Retriever:
    def __init__(self, lucene_searcher=None, lucene_index=None):
        if lucene_searcher is not None:
            self.searcher = lucene_searcher
        elif lucene_index is not None:
            self.searcher = LuceneSearcher(lucene_index)
            self.searcher.set_language("vi")
        else:
            raise ValueError("Must provide lucene searcher or index")

    def retrieve(self, query, top_k=150):
        hits = self.searcher.search(query, k=top_k)
        return [json.loads(x.raw)["contents"] for x in hits]
