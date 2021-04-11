import os
from shutil import copyfile
with open("emotions.txt", "r", encoding='utf-8') as f_in:
    id2emotions = f_in.readlines()
    id2emotions = [x.strip() for x in id2emotions]

data_dir = "GoEmotions"

splits = ["train", "dev", "test"]
for _split in splits:
    with open(os.path.join(data_dir, f"{_split}.tsv"), "r", encoding='utf-8') as f_in:
        buf = f_in.readlines()
    copyfile(os.path.join(data_dir, f"{_split}.tsv"), os.path.join(data_dir, f"{_split}_backup.tsv"))
    with open(os.path.join(data_dir, f"{_split}.tsv"), "w", encoding='utf-8') as f_out:
        headers = buf[0].strip().split("\t")
        headers = headers[: -1] + ["human_label", ] + headers[-1: ]
        headers = '\t'.join(headers)
        f_out.write(f"{headers}\n")
        for i in range(1, len(buf)):
            fields = buf[i].strip().split("\t")
            _label = int(fields[-1])
            fields = fields[: -1] + [id2emotions[_label], ] + fields[-1: ]
            content = "\t".join(fields)
            f_out.write(f"{content}\n")


