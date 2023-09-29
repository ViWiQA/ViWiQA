from vimqa_retriever import *
from eval_utils import has_answers


id2mdr = {}
with open("retrieval_result/duplicate.json") as f:
    for line in f:
        d = json.loads(line)
        id2mdr[d["_id"]] = d


def retrieve_mdr(qid) -> List[Passage]:
    mdr = id2mdr[qid]
    return [Passage(x["title"], x["text"]) for x in mdr["candidate_chains"][0]]


def eval_retrieval_accuracy_vimqa(
    eval_file, retrieve_method, acc_k=1, num_retrieve=100
):
    with open(eval_file) as f:
        eval_ds = json.load(f)

    correct_1 = 0
    correct_2 = 0
    q_ans_count = 0
    q_ans_correct = 0
    q_count = 0
    qa_data = []
    for d in tqdm(eval_ds):
        q_count += 1
        qid = d["_id"]
        question = d["question"]
        titles = [x[0] for x in d["context"]]
        passages = retrieve_mdr(qid)
        retr_titles = [x.title for x in passages[:acc_k]]
        correct = len(set(titles).intersection(set(retr_titles)))
        if correct == 1:
            correct_1 += 1
        elif correct == 2:
            correct_2 += 1
            correct_1 += 1

        answer = d["answer"]
        if answer not in ["đúng", "không"]:
            q_ans_count += 1
            for p in passages[:acc_k]:
                ctx = p.title + ": " + p.content
                if has_answers(answer, ctx):
                    q_ans_correct += 1
                    break
        if len(passages) >= 2:
            qa_data.append(
                {
                    "id": qid,
                    "context": "đúng không - "
                    + " ".join([x.title + ": " + x.content for x in passages][:2]),
                    "question": question,
                    "title": "",
                    "answers": {"answer_start": [0], "text": [answer]},
                }
            )
        else:
            print(f"No context, id={qid}")
    print("Accuracy 1 context", round(correct_1 / q_count * 100, 2))
    print("Accuracy 2 context", round(correct_2 / q_count * 100, 2))
    print("Accuracy has answer", round(q_ans_correct / q_ans_count * 100, 2))
    print("Not Yes/No question count:", q_ans_count)
    return qa_data


INP = "vimqa_dev_gold_only.json"
OUT = "vimqa/test_vimqa_ctx_ce_graph.json"
qa_data = eval_retrieval_accuracy_vimqa(
    INP, retrieve_method=None, acc_k=2, num_retrieve=100
)
