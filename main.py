import json
import os
import pickle
import random
import uuid
from datetime import datetime

import torch
from loguru import logger
from torch.nn import CrossEntropyLoss
from torch.optim import Adam
from transformers import AutoTokenizer

from config import Args
from model import BertClassifier


def load_data(tokenizer: AutoTokenizer, args):
    def get_data_emotions(filename, output_path="processed_data/train.tsv"):
        user_input = []
        user_labels = []
        f_out = open(output_path, "w", encoding='utf-8')
        # the output file is prepared according to datamap format
        # the first field should be guid, and the last field should be label (string)
        # later we will use log_processor for further processing
        f_out.write("guid\ttext\tlabel\n")
        with open(filename, "r", encoding='utf-8') as f_in:
            for line in f_in:
                # text, labels, comment_id = line.split("\t")
                _temp = line.strip().split("\t")
                labels = _temp[-1]
                text = "".join(_temp[:-1])
                labels = labels.split(",")
                # ignore multi-label, since datamap do not have support for multi-label problem
                if len(labels) > 1:
                    continue
                guid = str(uuid.uuid4())
                user_input.append((text, guid))
                user_labels.append(labels[0])
                f_out.write("\t".join([guid, text, labels[0]]) + "\n")
        f_out.close()
        return user_input, user_labels

    def create_batches(inputs, labels, ids):
        buf_input = []
        buf_label = []
        buf_ids = []
        ret = []
        for i, _id in enumerate(ids):
            buf_input.append(inputs[_id][0])
            buf_label.append(labels[_id])
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

    try:
        user_label_dict = pickle.load(open(os.path.join(args.dir_processed_data_for_datamap, "label_dict.pkl"), "rb"))
        train_data, dev_data, test_data = pickle.load(
            open(os.path.join(args.dir_processed_data_for_datamap, f"data_split_bs{args.BATCH_SIZE}.pkl"), "rb"))
        all_user_inputs, all_user_labels = pickle.load(
            open(os.path.join(args.dir_processed_data_for_datamap, "all_data.pkl"), "rb"))
    except:
        print("File Loading Error, try to re-create dataset....")
        os.makedirs(args.dir_processed_data_for_datamap, exist_ok=True)
        user_input_train, user_labels_train = get_data_emotions(args.origin_train_file, args.output_train_file)
        user_input_dev, user_labels_dev = get_data_emotions(args.origin_dev_file, args.output_dev_file)
        user_input_test, user_labels_test = get_data_emotions(args.origin_test_file, args.output_test_file)
        user_labels_set = set(user_labels_train) | set(user_labels_dev) | set(user_labels_test)
        all_user_inputs = user_input_train + user_input_dev + user_input_test
        all_user_labels = user_labels_train + user_labels_dev + user_labels_test
        user_label_dict = dict()
        for i, label in enumerate(user_labels_set):
            user_label_dict[label] = i

        for i in range(len(all_user_labels)):
            all_user_labels[i] = user_label_dict[all_user_labels[i]]

        train_ids = list(range(len(user_input_train)))
        dev_ids = list(range(len(user_input_dev)))
        dev_ids = [x + len(user_input_train) for x in dev_ids]
        test_ids = list(range(len(user_input_test)))
        test_ids = [x + len(user_input_train) + len(user_input_dev) for x in test_ids]
        train_data = create_batches(all_user_inputs, all_user_labels, train_ids)
        dev_data = create_batches(all_user_inputs, all_user_labels, dev_ids)
        test_data = create_batches(all_user_inputs, all_user_labels, test_ids)
        pickle.dump(user_label_dict, open(os.path.join(args.dir_processed_data_for_datamap, "label_dict.pkl"), "wb"))
        pickle.dump([train_data, dev_data, test_data],
                    open(os.path.join(args.dir_processed_data_for_datamap, f'data_split_bs{args.BATCH_SIZE}.pkl'),
                         "wb"))
        pickle.dump([all_user_inputs, all_user_labels],
                    open(os.path.join(args.dir_processed_data_for_datamap, "all_data.pkl"), "wb"))

    print(f"Successfully load {len(all_user_inputs)} data!")
    print(f"train: {args.BATCH_SIZE * (len(train_data) - 1) + len(train_data[-1]['id'])}")
    print(f"dev: {args.BATCH_SIZE * (len(dev_data) - 1) + len(dev_data[-1]['id'])}")
    print(f"test: {args.BATCH_SIZE * (len(test_data) - 1) + len(test_data[-1]['id'])}")
    return train_data, dev_data, test_data, user_label_dict, all_user_inputs, all_user_labels


def batch_iteration(data, model, criterion, optimizer, mode, record=False):
    assert mode in {"eval", "train"}
    res_dict = {}
    loss = 0
    acc = 0
    count = 0
    for _batch in data:
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
                    "label": _label.item(),
                    "prediction": _prediction.item(),
                    "logits": _output.tolist()
                }
    return loss / count, acc / count, res_dict


if __name__ == '__main__':

    args = Args()
    random.seed(args.SEED)
    torch.manual_seed(args.SEED)
    torch.cuda.manual_seed(args.SEED)
    tokenizer = AutoTokenizer.from_pretrained(args.BASE_MODEL)
    train_data, valid_data, test_data, user_label_dict, all_user_inputs, all_user_labels \
        = load_data(tokenizer, args)
    key_to_id = dict()
    for i, item in enumerate(all_user_inputs):
        _guid = item[1]
        key_to_id[_guid] = i
    id2label = dict()
    for k, v in user_label_dict.items():
        id2label[v] = k
    criterion = CrossEntropyLoss()
    log_name = "running_{time}".format(time=datetime.now().strftime("%Y-%m-%d-%H_%M_%S"))
    log_path = os.path.join(args.dir_processed_data_for_datamap, "logs", log_name)
    os.makedirs(log_path, exist_ok=True)
    logger.add(os.path.join(log_path, "log.txt"))
    logger.info("Config: {}".format(json.dumps(vars(args), indent=4)))
    all_data = train_data + valid_data + test_data
    model = BertClassifier(len(user_label_dict), args).cuda()
    optimizer = Adam(model.parameters(), lr=2e-5)
    best_acc = -99999

    for _epoch in range(args.EPOCHS):
        train_loss, train_acc, train_res_dict = batch_iteration(train_data, model, criterion, optimizer, "train", True)
        with torch.no_grad():
            valid_loss, valid_acc, _ = batch_iteration(valid_data, model, criterion, optimizer, "eval", False)

        logger.info("Epoch: {}, train_loss: {}, train_acc: {}, valid_loss: {}, valid_acc: {}".format(
            _epoch + 1, train_loss, train_acc, valid_loss, valid_acc
        ))
        if valid_acc > best_acc:
            best_acc = valid_acc
            model.save_pretrained(args.model_path)
            logger.info("Best updated at Epoch: {}, best_acc:{}".format(_epoch + 1, best_acc))
        for key in train_res_dict:
            train_res_dict[key]["inputs"] = all_user_inputs[key_to_id[key]]
            train_res_dict[key]["label"] = id2label[train_res_dict[key]["label"]]
            train_res_dict[key]["prediction"] = id2label[train_res_dict[key]["prediction"]]

        json.dump(train_res_dict,
                  open(os.path.join(args.dir_processed_data_for_datamap, "train_res_dict_epoch{}.json".format(_epoch)),
                       "w", encoding='utf-8'), indent=4)
    model.from_pretrained(args.model_path)
    with torch.no_grad():
        test_loss, test_acc, test_res_dict = batch_iteration(test_data, model, criterion, optimizer, "eval", True)
        logger.info("Training finished, test_acc:{}, test_loss:{}".format(test_acc, test_loss))
        valid_loss, valid_acc, valid_res_dict = batch_iteration(valid_data, model, criterion, optimizer, "eval", True)
        json.dump(valid_res_dict,
                  open(os.path.join(args.dir_processed_data_for_datamap, "dev_res_dict.json"),
                       "w", encoding='utf-8'), indent=4)
        json.dump(test_res_dict,
                  open(os.path.join(args.dir_processed_data_for_datamap, "test_res_dict.json"),
                       "w", encoding='utf-8'), indent=4)
