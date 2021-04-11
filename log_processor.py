import json
import os
import pickle
from config import Args
args = Args()
log_path = args.dir_processed_data_for_datamap
label_dict = pickle.load(open(os.path.join(log_path, "label_dict.pkl"), "rb"))
output_path = "training_dynamics"
os.makedirs(output_path, exist_ok=True)
Epoch = 20
for _epoch in range(Epoch):
    with open(os.path.join(log_path, f"train_res_dict_epoch{_epoch}.json"), "r", encoding='utf-8') as f_in:
        data = json.load(f_in)
        f_out = open(os.path.join(output_path, f"dynamics_epoch_{_epoch}.jsonl"), "w", encoding='utf-8')
        for _id in data:
            new_dict = {
                "guid": _id,
                f"logits_epoch_{_epoch}": data[_id]["logits"],
                "gold": label_dict[data[_id]["label"]]
            }
            f_out.write(f"{json.dumps(new_dict)}\n")
        f_out.close()
