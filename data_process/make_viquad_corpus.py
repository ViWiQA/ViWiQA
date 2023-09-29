from collections import defaultdict
import json
from tqdm import tqdm

TRAIN = "viquad/train_ViQuAD.json"
DEV = "viquad/dev_ViQuAD.json"
TEST = "viquad/test_ViQuAD.json"


cur_id = 0


def split_on_window(sequence, limit, step):
    ret = []
    split_sequence = sequence.split()
    l, r = 0, limit
    while r < len(split_sequence):
        s = " ".join(split_sequence[l:r])
        ret.append(s)
        l += step
        r += step
    if l < len(split_sequence):
        s = " ".join(split_sequence[l:r])
        ret.append(s)
    return ret


def get_data_sliding_window(f_path, window_size=100, stride=50):
    global cur_id
    with open(f_path) as f:
        ds = json.load(f)
    data = []
    title2ctx = defaultdict(list)
    for d in tqdm(ds["data"]):
        title = d["title"]
        for para in tqdm(d["paragraphs"]):
            context = para["context"]
            title2ctx[title].append(context)

    for title, ctxs in title2ctx.items():
        all_ctx = "\n".join(ctxs)
        window_ctxs = split_on_window(all_ctx, window_size, stride)
        for ctx in window_ctxs:
            data.append(
                {
                    "contents": ctx,
                    "title": title,
                    "id": f"viquad_corpus_w{window_size}_{cur_id}",
                }
            )
            cur_id += 1
    return data


def get_data(f_path):
    global cur_id
    with open(f_path) as f:
        ds = json.load(f)
    data = []
    for d in tqdm(ds["data"]):
        title = d["title"]
        for para in tqdm(d["paragraphs"]):
            context = para["context"]
            data.append(
                {"contents": context, "title": title, "id": f"viquad_corpus_{cur_id}"}
            )
            cur_id += 1
    return data


mydata = (
    get_data_sliding_window(DEV)
    + get_data_sliding_window(TEST)
    + get_data_sliding_window(TRAIN)
)
print("Done")
with open("viquad_corpus_w100.jsonl", "w") as f:
    f.write("\n".join([json.dumps(x, ensure_ascii=False) for x in mydata]))
