import pandas as pd
import torch
import torch.nn as nn
from torch.utils.data import Dataset
from transformers import BertTokenizerFast as BertTokenizer, BertForSequenceClassification, BertModel

class TitleDetectionDataset(Dataset):
    def __init__(
        self,
        data: pd.DataFrame,
        tokenizer: BertTokenizer,
        max_token_len: int = 128
    ):
        self.tokenizer = tokenizer
        self.data = data
        self.max_token_len = max_token_len

    def __len__(self):
        return len(self.data)

    def __getitem__(self, index: int):
        data_row = self.data.iloc[index]
        text = data_row["Text"]
        others = list(data_row[1:-1])
        labels = [data_row["Label"]]
        encoding = self.tokenizer.encode_plus(
            text,
            add_special_tokens=True,
            max_length=self.max_token_len,
            return_token_type_ids=False,
            padding="max_length",
            truncation=True,
            return_attention_mask=True,
            return_tensors='pt',
        )
        return dict(
            text=text,
            input_ids=encoding["input_ids"].flatten(),
            attention_mask=encoding["attention_mask"].flatten(),
            others=torch.FloatTensor(others),
            labels=torch.FloatTensor(labels),
        )

class TitleDetection(nn.Module):
    def __init__(self, n_other_features, criterion=nn.BCELoss()):
        super(TitleDetection, self).__init__()
        self.bert = BertModel.from_pretrained(
            "bert-base-cased")
        self.ff1 = nn.Linear(
            self.bert.config.hidden_size + n_other_features, 1)
        self.criterion = criterion

    def forward(self, input_ids, attention_mask, others, labels=None):
        output = self.bert(input_ids=input_ids, attention_mask=attention_mask)
        output = output.pooler_output
        output = torch.cat((output, others), 1)
        output = self.ff1(output)
        output = torch.sigmoid(output)
        loss = 0
        if labels is not None:
            loss = self.criterion(output, labels)
        return loss, output


class TitleDetection_BertSequence(nn.Module):
    def __init__(self, n_other_features, criterion=nn.BCELoss()):
        super(TitleDetection_BertSequence, self).__init__()
        self.bert = BertForSequenceClassification.from_pretrained(
            "bert-base-cased")
        self.ff1 = nn.Linear(
            2 + n_other_features, 1)
        self.criterion = criterion

    def forward(self, input_ids, attention_mask, others, labels=None):
        output = self.bert(input_ids=input_ids, attention_mask=attention_mask)[0]
        output = torch.cat((output, others), 1)
        output = self.ff1(output)
        output = torch.sigmoid(output).to(torch.float)
        loss = 0
        if labels is not None:
            loss = self.criterion(output, labels)
        return loss, output

