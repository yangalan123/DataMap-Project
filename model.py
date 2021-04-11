from torch import nn
from transformers import AutoModel
import torch


class BertClassifier(nn.Module):
    def __init__(self, num_label, arg):
        super(BertClassifier, self).__init__()
        self.base_model = AutoModel.from_pretrained(arg.BASE_MODEL).eval()
        self.hidden_size = self.base_model.config.hidden_size
        self.output = nn.Linear(self.hidden_size, num_label)
        self.criterion = nn.CrossEntropyLoss()
        self.arg = arg

    def forward(self, inputs, **kwargs):
        hiddens = self.base_model(**inputs)[1]
        logits = self.output(hiddens)
        return logits

    def save_pretrained(self, path):
        torch.save(self.state_dict(), path)

    def from_pretrained(self, path):
        self.load_state_dict(torch.load(path), strict=False)

