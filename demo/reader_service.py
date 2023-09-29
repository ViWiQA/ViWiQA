import torch
from transformers import pipeline, QuestionAnsweringPipeline


device = "cuda" if torch.cuda.is_available() else "cpu"
QA_NAME = "path/to/model"


def init_qa():
    question_answerer: QuestionAnsweringPipeline = pipeline(
        "question-answering",
        model=QA_NAME,
        device=0 if torch.cuda.is_available() else -1,
    )
    return question_answerer


def get_qa_preds(question_answerer, question, contexts):
    preds = question_answerer(
        question=question,
        context=contexts,
        max_answer_len=500,
        max_seq_len=512,
        max_question_len=128,
        device=0,
    )
    return preds, contexts
