import glob
import os

def augment_from_argument_data():
    prefix_path = "ArgumentAnnotatedEssays-1.0/ArgumentAnnotatedEssays-1.0/brat-project/brat-project"
    res = []
    for filename in glob.glob(os.path.join(prefix_path, "*.ann")):
        with open(filename, "r", encoding='utf-8') as f_in:
            for line in f_in:
                content = line.strip().split("\t")
                if "Premise" in content[1]:
                    res.append("\t".join([content[-1], "pathos"]))
    return res


if __name__ == '__main__':
    train_file = os.path.join("PathosLogos", "origin", "PathosLogos", "train.tsv")
    with open(train_file, "r", encoding='utf-8') as f_in:
        buf = f_in.readlines()
        buf = [x.strip() for x in buf]
    count_pathos = 0
    count_logos = 0
    for line in buf:
        contents = line.split("\t")
        if contents[-1] == "pathos":
            count_pathos += 1
        elif contents[-1] == "logos":
            count_logos += 1
        else:
            raise Exception(f"wtf{contents[-1]}")

    # res = augment_from_argument_data()
    with open("pred_res.tsv", "r", encoding='utf-8') as f_in:
        res = f_in.readlines()
        res = [x.strip() for x in res]
        res = [x for x in res if x.split("\t")[-1] == "pathos"]
    # res = res[: count_logos - count_pathos]
    print(len(res))
    # exit()
    with open(os.path.join("PathosLogos", "train_aug_model_pred.tsv"), "w") as f_out:
        # for line in buf + res:
        for line in buf + res:
            f_out.write(line + "\n")

