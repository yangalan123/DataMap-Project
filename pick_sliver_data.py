import os
data_dir = "pathsLogos/pathos_premise"
thres = ["0.01", "0.05", "0.10", "0.17", "0.25", "0.33", "0.50", "0.75"]
metrics = ["confidence", "variability"]
output = ["hard_to_learn.txt", "ambiguous.txt"]
with open(os.path.join(data_dir, "train.tsv"), "r") as f_in:
    buf = f_in.readlines()

pseudo_labeled = buf[-38:]
guid2metrics = dict()
for line in pseudo_labeled:
    content = line.strip().split("\t")
    guid = content[0]
    guid2metrics[guid] = dict()
    guid2metrics[guid]["content"] = content[1]


for _metric in metrics:
    for _thre in thres:
        with open(f"cartography/filtered/cartography_{_metric}_{_thre}/{data_dir}/train.tsv", "r",
                  encoding='utf-8') as f_in:
            for i, line in enumerate(f_in):
                if i == 0:
                    continue
                content = line.strip().split("\t")
                guid = content[0]
                if guid in guid2metrics:
                    if _metric not in guid2metrics[guid]:
                        guid2metrics[guid][_metric] = float(_thre)

res = []
for guid in guid2metrics:
    _res = [guid, ]
    for _metric in metrics:
        if _metric not in guid2metrics[guid]:
            _res.append(1)
        else:
            _res.append(guid2metrics[guid][_metric])
    res.append(tuple(_res))
for i, output_filename in enumerate(output):
    res.sort(key=lambda x: x[i + 1])
    with open(os.path.join(data_dir, output_filename), "w", encoding='utf-8') as f_out:
        f_out.write("text\tconfidence_level\tvariability_level\n")
        for item in res:
            guid = item[0]
            text = guid2metrics[guid]["content"]
            f_out.write(f"{text}\t{item[1]}\t{item[2]}\n")





