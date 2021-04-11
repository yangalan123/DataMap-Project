import json
import pickle
import numpy as np
label_dict = pickle.load(open("label_dict.pkl", "rb"))
# inv_map = dict()
# for k, v in label_dict.items():
#     try:
#         inv_map[v] = int(k)
#     except:
#         inv_map[v] = k


# with open("emotions.txt", "r", encoding='utf-8') as f_in:
#     id2emotions = f_in.readlines()
#     id2emotions = [x.strip() for x in id2emotions]
uid2results = dict()
uid2label = dict()

for i in range(20):
    with open(f"training_dynamics/dynamics_epoch_{i}.jsonl", "rb") as f_in:
        for line in f_in:
            _data = json.loads(line.strip())
            _uuid = _data["guid"]
            _logits = np.array(_data[f"logits_epoch_{i}"])
            _orders = list(np.argsort(-_logits))
            _top3 = [x for x in _orders[:3]]
            if _uuid not in uid2results:
                uid2results[_uuid] = []
            uid2results[_uuid].append({
                "top3": list(_top3),
            })
            if _uuid not in uid2label:
                uid2label[_uuid] = _data["gold"]
            else:
                assert uid2label[_uuid] == _data["gold"]

new_id2label = dict()
bufs = []
# dataset_name = "GoEmotions"
dataset_name = "PathosLogos"
# metric = "variability"
metric = "confidence"
with open(f"cartography/filtered/cartography_{metric}_0.01/{dataset_name}/train.tsv", "r", encoding='utf-8') as f_in:
    with open(f"cartography/filtered/cartography_{metric}_0.01/{dataset_name}/train_simp.tsv", "w", encoding='utf-8') as f_out:
        for i, line in enumerate(f_in):
            if i == 0:
                content = line.strip().split("\t")
                f_out.write("\t".join(content[1: -1] + ["label_change", ]) + "\n")
                continue
            content = line.strip().split("\t")
            bufs.append(content)
            uuid = content[0]
            assert uuid in uid2results
            assert uuid in uid2label
            if uid2label[uuid] not in new_id2label:
                new_id2label[uid2label[uuid]] = content[2]
            else:
                assert content[2] == new_id2label[uid2label[uuid]]
        # just for debugging
        if 1 not in new_id2label:
            new_id2label[1] = "logos"
        id_set = set(new_id2label.keys())
        value_set = set(new_id2label.values())
        assert len(id_set) == len(value_set)
        for content in bufs:
            uuid = content[0]
            _changes = [str(tuple([new_id2label[y] for y in x["top3"]])) for x in uid2results[uuid]]

            f_out.write("\t".join(content[1: -1]) + "\t" + "->".join(_changes) + "\n")
            # f_out.write("\t".join(content[1: -1]) + "\n")
            # for item in uid2results[uuid]:
            #     assert content[2] == item["gold"]



