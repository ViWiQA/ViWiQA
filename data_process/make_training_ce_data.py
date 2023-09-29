from retriever import retrieve_bm25, init_bm25
import json
import util
from tqdm import tqdm

INP = "viquad/test_ViQuAD.json"
OUT = "ce_viquad_eol/test.json"


def sample_data(data_fp, neg_ratio=7):
    with open(data_fp) as f:
        ds = json.load(f)

    data = []
    for d in tqdm(ds["data"]):
        for para in d["paragraphs"]:
            for qa in para["qas"]:
                answers = [x["text"] for x in qa["answers"]]
                qid = qa["id"]
                question = qa["question"]
                gold_ctx = para["context"]
                passages = retrieve_bm25(question, top_k=100)
                pos = [gold_ctx]
                neg = []
                for ctx in passages:
                    if any([ans in ctx for ans in answers]):
                        pos.append(ctx)
                    else:
                        neg.append(ctx)
                    if len(neg) // len(pos) == neg_ratio:
                        break
                data.extend(
                    [
                        {"sentence1": question, "sentence2": ctx, "label": 1.0}
                        for ctx in pos
                    ]
                )
                data.extend(
                    [
                        {"sentence1": question, "sentence2": ctx, "label": 0.0}
                        for ctx in neg
                    ]
                )
    return data


if __name__ == "__main__":
    init_bm25()
    data = sample_data(INP)
    with open(OUT, "w") as f:
        f.write("\n".join([json.dumps(x, ensure_ascii=False) for x in data]))
