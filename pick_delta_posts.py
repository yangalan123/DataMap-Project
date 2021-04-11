def read_tsv(filepath):
    guid2content = dict()
    with open(filepath, "r", encoding='utf-8') as f_in:
        buf = f_in.readlines()
        for i in range(1, len(buf)):
            _buf = buf[i].strip().split("\t")
            guid = _buf[0]
            guid2content[guid] = buf[i]
    return guid2content


in_1_dict = read_tsv("cartography/filtered/cartography_variability_0.50/GoEmotions/train.tsv")
in_2_dict = read_tsv("cartography/filtered/cartography_variability_0.75/GoEmotions/train.tsv")

