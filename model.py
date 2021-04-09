from torch import nn
from transformers import AutoModel
import torch
class BertClassifier(nn.Module):
    def __init__(self, num_label, arg):
        super(BertClassifier, self).__init__()
        self.base_model = AutoModel.from_pretrained(arg.BASE_MODEL).eval()
        self.hidden_size = self.base_model.config.hidden_size
        self.output = nn.Linear(self.hidden_size, num_label)
        # self.output0 = nn.Linear(self.base_model.config.hidden_size, 1024)
        # self.output1 = nn.Linear(1024, num_label)
        self.criterion = nn.CrossEntropyLoss()
        self.dropout = nn.Dropout(0.1)
        self.transfer = nn.Sigmoid()
        # annealing part: encourage the model to explore more
        self.alpha = 2e-4
        # self.beta = 1 + 1e-3
        self.arg = arg
        # attention part
        # self.multi_head = nn.MultiheadAttention(self.hidden_size, arg.num_head)
        # self.Q_transform = nn.Linear(self.hidden_size, self.hidden_size)
        # self.K_transform = nn.Linear(self.hidden_size, self.hidden_size)
        # self.V_transform = nn.Linear(self.hidden_size, self.hidden_size)

    def forward(self, inputs, **kwargs):
        #print(inputs.shape)
        # hiddens = []
        # for i in range(0, len(inputs["input_ids"]), self.arg.bert_batch_size):
        #     # given our GPU memory is so limited
        #     #hidden = self.base_model(inputs[i: i + arg.bert_batch_size].cuda())[0][:, 0, :]
        #     #hidden = self.base_model(inputs["input_ids"][i: i + arg.bert_batch_size],
        #     #attention_mask=inputs["attention_mask"][i: i + arg.bert_batch_size],
        #     #)[0][:, 0, :]
        #     hidden = self.base_model(inputs["input_ids"][i: i + self.arg.bert_batch_size],
        #                              )[0][:, 0, :]
        #     hiddens.append(hidden)
        hiddens = self.base_model(**inputs)[1]
        # print(hiddens.shape)
        # logits = self.output1(self.dropout(self.transfer(self.output0(hidden))))
        # hiddens = hiddens.mean(dim=0, keepdim=True)
        logits = self.output(hiddens)
        # hiddens = self.multi_head(
        #     self.Q_transform(hiddens).unsqueeze(1),
        #     self.K_transform(hiddens).unsqueeze(1),
        #     self.V_transform(hiddens).unsqueeze(1)
        # )
        # _hiddens = hiddens.unsqueeze(1)
        # hiddens, weight = self.multi_head(
        #     _hiddens,
        #     _hiddens,
        #     _hiddens
        # )

        # logits = self.output(hiddens.squeeze(1).mean(dim=0, keepdim=True)) / alpha
        # loss = self.criterion(logits, labels)
        return logits

    def save_pretrained(self, path):
        torch.save(self.state_dict(), path)

    def from_pretrained(self, path):
        self.load_state_dict(torch.load(path), strict=False)

def prepare_for_model(tokenizer, batch_inputs, batch_labels, max_len_in_batch):
    FLAG_TYPE_IDS = True
    for i in range(len(batch_inputs)):
        # if len(batch_inputs[i]) > max_len_in_batch:
        #     batch_inputs[i] = batch_inputs[i][:max_len_in_batch]
        # else:
        #     batch_inputs[i] += [tokenizer.pad_token_id] * (max_len_in_batch - len(batch_inputs[i]))
        # batch_inputs[i] = [tokenizer.cls_token_id] + batch_inputs[i] + [tokenizer.sep_token_id]
        res = tokenizer.prepare_for_model(ids=batch_inputs[i], max_length=max_len_in_batch, pad_to_max_length=True, truncation=True)
        batch_inputs[i] = dict()
        batch_inputs[i]["input_ids"] = torch.LongTensor(res["input_ids"])
        batch_inputs[i]["attention_mask"] = torch.LongTensor(res["attention_mask"])
        if "token_type_ids" in res:
            batch_inputs[i]["token_type_ids"] = torch.LongTensor(res["token_type_ids"])
        else:
            FLAG_TYPE_IDS = False
    # labels = []
    # for i in batch_labels:
    #     _tmp = torch.zeros(3)
    #     _tmp[i] = 1
    #     labels.append(_tmp)
    # batch_labels = torch.stack(labels).cuda()
    # print_batch_input = [tokenizer.convert_ids_to_tokens(x) for x in batch_inputs]
    # print_batch_labels = [CLASS_NAMES[x] for x in batch_labels]
    # print_tuples = [(print_batch_input[i], print_batch_labels[i]) for i in range(len(print_batch_input))]
    # for item in print_tuples:
    #     print(item)
    # exit()
    batch_labels = torch.LongTensor(batch_labels).cuda()
    new_batch_inputs = dict()
    new_batch_inputs["input_ids"] = torch.stack([x["input_ids"] for x in batch_inputs]).cuda()
    new_batch_inputs["attention_mask"] = torch.stack([x["attention_mask"] for x in batch_inputs]).cuda()
    if FLAG_TYPE_IDS:
        new_batch_inputs["token_type_ids"] = torch.stack([x["token_type_ids"] for x in batch_inputs]).cuda()
    # batch_inputs = torch.stack(batch_inputs).cuda()
    return new_batch_inputs, batch_labels
