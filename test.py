from transformers import AutoTokenizer
from model import BertClassifier
from config import QuestionArgs
import torch
import random
import pickle
from torch.optim import Adam
from torch.nn import CrossEntropyLoss
from loguru import logger
import json
import os
from datetime import datetime
import numpy as np
import uuid


log_path = "pathsLogos/running_2021-04-09-02_19_58"
user_label_dict = pickle.load(open(os.path.join(log_path, "label_dict.pkl"), "rb"))
def load_data(tokenizer: AutoTokenizer, args):
    def get_data(filename, output_dir="processed_data"):
        os.makedirs(output_dir, exist_ok=True)
        user_input = []
        user_labels = []
        with open(filename, "r", encoding='utf-8') as f_in:
            for line in f_in:
                #text, labels, comment_id = line.split("\t")
                #try:
                _temp = line.split("\t")
                labels = "Dummy"
                text = "".join(_temp[:-1])
                #except:
                   #print(line)
                   #print(line.split("\t"))
                   #exit()
                # user_input.append(q + " [SEP] " + a1 + a2)
                labels = labels.split(",")
                # ignore multi-label for now
                if len(labels) > 1:
                    continue
                guid = str(uuid.uuid4())
                user_input.append((text, guid))
                user_labels.append(labels[0])
        return user_input, user_labels
    def create_batches(inputs, labels, ids):
        buf_input = []
        buf_label = []
        buf_ids = []
        ret = []
        for i, _id in enumerate(ids):
            buf_input.append(inputs[_id][0])
            buf_label.append(labels[_id])
            # buf_ids.append(_id)
            buf_ids.append(inputs[_id][1])
            if (i + 1) % args.BATCH_SIZE == 0:
                batch_inputs = tokenizer(buf_input, padding=True, return_tensors='pt', truncation=True)
                batch_labels = torch.LongTensor(buf_label)
                ret.append({"batch_inputs": batch_inputs, "batch_labels": batch_labels, "id": buf_ids.copy()})
                buf_label.clear()
                buf_input.clear()
                buf_ids.clear()
        if len(buf_input) > 0:
            batch_inputs = tokenizer(buf_input, padding=True, return_tensors="pt")
            batch_labels = torch.LongTensor(buf_label)
            ret.append({"batch_inputs": batch_inputs, "batch_labels": batch_labels, "id": buf_ids.copy()})
            buf_label.clear()
            buf_input.clear()
            buf_ids.clear()
        return ret
    user_input_test, user_labels_test = get_data("PathosLogos/all_argument_aug.tsv")
    all_user_inputs = user_input_test
    all_user_labels = user_labels_test
    for i in range(len(all_user_labels)):
        all_user_labels[i] = user_label_dict[random.sample(list(user_label_dict.keys()), 1)[0]]

    test_ids = list(range(len(user_input_test)))
    test_data = create_batches(all_user_inputs, all_user_labels, test_ids)

    print(f"test: {args.BATCH_SIZE * (len(test_data) - 1) + len(test_data[-1]['id'])}")
    return test_data, all_user_inputs, all_user_labels

def batch_iteration(data, model, criterion, optimizer, mode, record=False):
    assert mode in {"eval", "train"}
    res_dict = {}
    loss = 0
    acc = 0
    count = 0
    for _batch in data:
        # print(type(_batch["batch_inputs"]))
        inputs = _batch["batch_inputs"]
        _cuda_inputs = dict()
        for key in inputs:
            try:
                _cuda_inputs[key] = inputs[key].cuda()
            except:
                pass
        logits = model(_cuda_inputs)
        labels = _batch["batch_labels"].cuda()
        ids = _batch["id"]
        _loss = criterion(logits, labels)
        loss += _loss.cpu().item()
        prediction = torch.argmax(logits, dim=-1)
        acc += (prediction == labels).sum().cpu().item()
        count += len(labels)
        if mode == "train":
            _loss.backward()
            optimizer.step()
            optimizer.zero_grad()
        if record:
            for _id, _label, _prediction, _output in zip(ids, labels.cpu(), prediction.cpu(), logits.cpu()):
                res_dict[_id] = {
                    # "label": _label.item(),
                    "prediction": _prediction.item(),
                    "logits": _output.tolist()
                }
    return loss / count, acc / count, res_dict


args = QuestionArgs()
random.seed(args.SEED)
torch.manual_seed(args.SEED)
torch.cuda.manual_seed(args.SEED)
tokenizer = AutoTokenizer.from_pretrained(args.BASE_MODEL)
test_data, all_user_inputs, all_user_labels \
    = load_data(tokenizer, args)
key_to_id = dict()
for i, item in enumerate(all_user_inputs):
    #print(item)
    _guid = item[1]
    key_to_id[_guid]=i
id2label = dict()
for k, v in user_label_dict.items():
    id2label[v] = k
# model = BertClassifier(len(user_label_dict), args).cuda()
# optimizer = Adam(model.parameters(), lr=2e-5)
criterion = CrossEntropyLoss()
# log_name = "running_{time}".format(time=datetime.now().strftime("%Y-%m-%d-%H_%M_%S"))
# log_path = os.path.join("logs", log_name)
# log_path = os.path.join("logs/running_2021-03-21-21_57_15")
# os.makedirs(log_path, exist_ok=True)
# logger.add(os.path.join(log_path, "log.txt"))
# logger.info("Config: {}".format(json.dumps(vars(args), indent=4)))
all_data = test_data
# model_path = os.path.join(log_path, "model.pt")
# K_fold = 10
K_fold = 1
# group_size = len(all_user_labels) // K_fold
# assert group_size % args.BATCH_SIZE == 0
# valid_num = group_size // args.BATCH_SIZE
# assert ((K_fold - 1) * group_size) % args.BATCH_SIZE == 0
# train_num = ((K_fold - 1) * group_size) // args.BATCH_SIZE
# all_k_fold_accs = []
# all_k_fold_accs_dev = []
for no_fold in range(K_fold):
    # _k_fold_test_data = all_data[no_fold * valid_num: (no_fold + 1) * valid_num]
    # _k_fold_train_data = all_data[0: no_fold * valid_num] + all_data[(no_fold + 1) * valid_num:]
    # random.shuffle(_k_fold_train_data)
    # _k_fold_valid_data = _k_fold_train_data[0: valid_num]
    # _k_fold_train_data = _k_fold_train_data[valid_num: ]
    model = BertClassifier(len(user_label_dict), args).cuda()
    optimizer = Adam(model.parameters(), lr=2e-5)
    # model_path = os.path.join(log_path, "model_{}_fold.pt".format(no_fold))
    model_path = os.path.join(log_path, "model.pt".format(no_fold))
    # logger.info(f"Doing K-fold Evaluation: {no_fold + 1} / {K_fold}")
    best_acc = -99999

    # for _epoch in range(args.EPOCHS):
    # for _epoch in range(0):
    #     train_loss, train_acc, train_res_dict = batch_iteration(train_data, model, criterion, optimizer, "train", True)
    # #     train_loss, train_acc, _ = batch_iteration(_k_fold_train_data, model, criterion, optimizer, "train", False)
    #     with torch.no_grad():
    #         # valid_loss, valid_acc, _ = batch_iteration(_k_fold_valid_data, model, criterion, optimizer, "eval", False)
    #         valid_loss, valid_acc, _ = batch_iteration(valid_data, model, criterion, optimizer, "eval", False)

        # logger.info("Epoch: {}, train_loss: {}, train_acc: {}, valid_loss: {}, valid_acc: {}".format(
        #     _epoch + 1, train_loss, train_acc, valid_loss, valid_acc
        # ))
        # if valid_acc > best_acc:
        #     best_acc = valid_acc
        #     model.save_pretrained(model_path)
        #     logger.info("Best updated at Epoch: {}, best_acc:{}".format(_epoch + 1, best_acc))
        # for key in train_res_dict:
        #     train_res_dict[key]["inputs"] = all_user_inputs[key_to_id[key]]
        #     train_res_dict[key]["label"] = id2label[train_res_dict[key]["label"]]
        #     train_res_dict[key]["prediction"] = id2label[train_res_dict[key]["prediction"]]

        # json.dump(train_res_dict, open(os.path.join(log_path, "train_res_dict_epoch{}.json".format(_epoch)), "w", encoding='utf-8'), indent=4)
    model.from_pretrained(model_path)
    with torch.no_grad():
        # test_loss, test_acc, _ = batch_iteration(_k_fold_test_data, model, criterion, optimizer, "eval", False)
        test_loss, test_acc, res_dict = batch_iteration(test_data, model, criterion, optimizer, "eval", True)
        # logger.info("Training finished, test_acc:{}, test_loss:{}".format(test_acc, test_loss))
        with open("pred_res.tsv", "w", encoding='utf-8') as f_out:
            for _id in res_dict:
                _text = all_user_inputs[key_to_id[_id]][0]
                f_out.write(f"{_text}\t{id2label[res_dict[_id]['prediction']]}")


        # all_loss, all_acc, res_dict = batch_iteration(_k_fold_train_data + _k_fold_valid_data + _k_fold_test_data, model, criterion, optimizer, "eval", record=True)
    # all_k_fold_accs.append(test_acc)
    # all_k_fold_accs_dev.append(best_acc)


#     for key in res_dict:
#         res_dict[key]["inputs"] = all_user_inputs[key]
#         res_dict[key]["label"] = id2label[res_dict[key]["label"]]
#         res_dict[key]["prediction"] = id2label[res_dict[key]["prediction"]]
#
#     json.dump(res_dict, open(os.path.join(log_path, "res_dict_fold{}.json".format(no_fold)), "w", encoding='utf-8'), indent=4)
#
# print(all_k_fold_accs)
# print(np.mean(all_k_fold_accs), np.std(all_k_fold_accs))
# print(all_k_fold_accs_dev)
# print(np.mean(all_k_fold_accs_dev), np.std(all_k_fold_accs_dev))




