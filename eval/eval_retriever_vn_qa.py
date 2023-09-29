from retriever import *
from eval_utils import has_answers
from vimqa_retriever import retrieve_ce_graph


def clean_ds(squad_ds):
    qids = set()
    for d in squad_ds["data"]:
        for para in d["paragraphs"]:
            new_qas = []
            for qa in para["qas"]:
                qid = qa["id"]
                if qid in qids:
                    print("Remove duplicate ID:", qid)
                    continue
                else:
                    qids.add(qid)
                    answers = [
                        dict(t) for t in {tuple(d.items()) for d in qa["answers"]}
                    ]
                    new_qas.append(
                        {"id": qid, "answers": answers, "question": qa["question"]}
                    )
            para["qas"] = new_qas
    squad_ds["version"] = "cleaned"
    return squad_ds


def eval_retrieval_accuracy(eval_file, retrieve_method, acc_k=1, num_retrieve=100):
    with open(eval_file) as f:
        eval_ds = json.load(f)

    correct_count = 0
    q_count = 0
    qa_data = []
    for d in tqdm(eval_ds["data"]):
        title = d["title"]
        for para in tqdm(d["paragraphs"]):
            for qa in para["qas"]:
                q_count += 1
                question = qa["question"]
                answers = qa["answers"]
                ans_text = [x["text"] for x in answers]
                ans_start = [x["answer_start"] for x in answers]
                qid = qa["id"]

                contexts = retrieve_method(question, num_retrieve)
                for ctx in contexts[:acc_k]:
                    if has_answers(ans_text, ctx):
                        correct_count += 1
                        break
                if len(contexts) > 0:
                    qa_data.append(
                        {
                            "id": qid,
                            "context": contexts[0],
                            "question": question,
                            "title": title,
                            "answers": {"answer_start": ans_start, "text": ans_text},
                        }
                    )
                else:
                    print(f'No context, id={d["id"]}')
    print("Acc", round(correct_count / q_count * 100, 2))
    return qa_data


def eval_retrieval_accuracy_vimqa(
    eval_file, retrieve_method, acc_k=1, num_retrieve=100
):
    with open(eval_file) as f:
        eval_ds = json.load(f)

    correct_count = 0
    q_count = 0
    qa_data = []
    for d in tqdm(eval_ds):
        q_count += 1
        qid = d["_id"]
        question = d["question"]
        contexts = retrieve_method(question, num_retrieve, include_title=True)
        answer = d["answer"]
        for ctx in contexts[:acc_k]:
            if has_answers(answer, ctx):
                correct_count += 1
                break
        if len(contexts) >= 2:
            qa_data.append(
                {
                    "id": qid,
                    "context": "đúng không - " + " ".join(contexts[:2]),
                    "question": question,
                    "title": "",
                    "answers": {"answer_start": [0], "text": [answer]},
                }
            )
        else:
            print(f"No context, id={qid}")
    print("Acc", round(correct_count / q_count * 100, 2))
    return qa_data


init_bm25()
init_ce()
INP = "dev_ViQuAD.json"
OUT = "eval_out.json"

for k in [1, 5, 10, 20, 50]:
    print("-" * 30)
    print("k=", k)
    eval_retrieval_accuracy(INP, retrieve_method=retrieve_ce, acc_k=k, num_retrieve=100)
