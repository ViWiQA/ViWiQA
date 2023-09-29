from vimqa_retriever import retrieve_bm25, init_bm25
import json
import util
from tqdm import tqdm

INP = "vimqa/vimqa_train.json"
OUT = "vimqa_ce/train_ce_vimqa.json"


def sample_data(data_fp, neg_ratio=7):
    with open(data_fp) as f:
        ds = json.load(f)

    data = []
    for d in tqdm(ds):
        answer = d["answer"].strip()
        if answer in ["đúng", "không"]:
            # Ignore Yes/No question
            continue
        qid = d["_id"]
        question = d["question"]
        titles = [x[0].strip() for x in d["context"]]
        passages = retrieve_bm25(question, top_k=100)
        all_context = " ".join([" ".join(x[1]) for x in d["context"]])
        pos = util.split_on_window(all_context, limit=100, step=50)
        neg = []
        for p in passages:
            ctx = p.content
            if answer in ctx or answer in titles:
                pos.append(ctx)
            else:
                neg.append(ctx)
            if len(neg) // len(pos) == neg_ratio:
                break
        data.extend(
            [{"sentence1": question, "sentence2": ctx, "label": 1.0} for ctx in pos]
        )
        data.extend(
            [{"sentence1": question, "sentence2": ctx, "label": 0.0} for ctx in neg]
        )
    return data


if __name__ == "__main__":
    init_bm25()
    data = sample_data(INP)
    with open(OUT, "w") as f:
        f.write("\n".join([json.dumps(x, ensure_ascii=False) for x in data]))
