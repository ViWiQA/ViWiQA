import json
import csv
import re

RETRIEVED_PASSAGES = "path/to/retrieve_data.tsv"
COLLECTIONS = "path/to/collections.tsv"
DATA = "path/to/test_queries_with_answers.json"
N = 20


def make_id_int(id, id_set):
    suggestion = re.sub("[^0-9]", "", id)
    while suggestion in id_set:
        suggestion = suggestion + "1"
    id_set.add(suggestion)
    return suggestion


retrieval = {}
with open(RETRIEVED_PASSAGES, "r") as file:
    tsv_file = csv.reader(file, delimiter="\t")
    for line in tsv_file:
        qid = line[0]
        pid = line[1]
        rank = int(line[2]) - 1
        passages = retrieval.get(qid, [])
        if not passages:
            passages = [0 for _ in range(N)]
            retrieval[qid] = passages
        passages[rank] = pid

collections = {}
with open(COLLECTIONS, "r") as file:
    tsv_file = csv.reader(file, delimiter="\t")
    for line in tsv_file:
        collections[line[0]] = line[1]

data = {}
with open(DATA, "r") as f:
    data = json.load(f)


def evaluate(top_k=1):
    correct_count = 0
    qid_set = set()
    for k, v in data.items():
        qid = str(int(make_id_int(k, qid_set)))
        # print('q:' ,v)
        # print('retrieve:' )
        # print([collections[p] for p in retrieval[qid][:top_k]])
        for p in retrieval[qid][:top_k]:
            if any([v["answers"][0] in collections[p]]):
                correct_count += 1
                break
    return len(data), correct_count, correct_count / len(data)


k = [1, 5, 10, 20]
for i in k:
    print(i, evaluate(i))
