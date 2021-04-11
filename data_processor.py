import os
from shutil import copyfile
import pickle
from config import Args
args = Args()
data_dir = args.dir_processed_data_for_datamap
with open(os.path.join(data_dir, "label_dict.pkl"), "rb") as f_in:
    label_dict = pickle.load(f_in)

splits = ["train", "dev", "test"]
for _split in splits:
    with open(os.path.join(data_dir, f"{_split}.tsv"), "r", encoding='utf-8') as f_in:
        buf = f_in.readlines()
    # rule out empty lines
    buf = [x for x in buf if len(x.strip()) > 0]
    copyfile(os.path.join(data_dir, f"{_split}.tsv"), os.path.join(data_dir, f"{_split}_backup.tsv"))
    with open(os.path.join(data_dir, f"{_split}.tsv"), "w", encoding='utf-8') as f_out:
        headers = buf[0].strip().split("\t")
        # headers = headers[: -1] + ["human_label", ] + headers[-1: ]
        headers = headers[: -1] + ["human_label", "label"]
        headers = '\t'.join(headers)
        f_out.write(f"{headers}\n")
        for i in range(1, len(buf)):
            fields = buf[i].strip().split("\t")
            fields = fields + [str(label_dict[fields[-1]]), ]
            content = "\t".join(fields)
            f_out.write(f"{content}\n")


