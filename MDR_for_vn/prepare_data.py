import json
from tqdm import tqdm

DATASET = "vimqa_dev.json"
OUTPUT_FILE = "vimqa_dev_with_neg_bridge.json"

process_data = []

with open(DATASET, "r") as f:
    data = json.load(f)

for sample in tqdm(data):
    # Get supporting title
    supporting = set(x[0] for x in sample["supporting_facts"])
    if len(supporting) != 2:
        continue

    # Join texts to paragraph with title
    passages = []
    for p in sample["context"]:
        x = {"title": p[0], "text": " ".join(p[1])}
        passages.append(x)

    # Get positive and negative paragraphs
    pos_para = []
    neg_para = []
    for p in passages:
        if p["title"] in supporting:
            pos_para.append(p)
            bridge = p["title"]
        else:
            neg_para.append(p)

    a = {
        "question": sample["question"],
        "answers": [sample["answer"]],
        "type": sample["type"],
        "pos_paras": pos_para,
        "neg_paras": neg_para,
        "_id": sample["_id"],
    }
    if sample["type"] == "bridge":
        a["bridge"] = bridge
    process_data.append(a)


with open(OUTPUT_FILE, "w") as f:
    f.write("\n".join([json.dumps(x, ensure_ascii=False) for x in process_data]))
