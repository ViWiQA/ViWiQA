import json
from tqdm import tqdm

data = []
cur_id = 0
with open("paragraphs.jsonl") as f:
    for line in tqdm(f):
        para_info = json.loads(line)
        for para in para_info["paragraphs"]:
            data.append(
                {
                    "contents": para,
                    "title": para_info["title"],
                    "id": f"wiki_vn_{cur_id}",
                }
            )
            cur_id += 1
print("Done")
with open("wiki_vn_para_100.jsonl", "w") as f:
    f.write("\n".join([json.dumps(x, ensure_ascii=False) for x in data]))
