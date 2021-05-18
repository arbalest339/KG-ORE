'''
Author: your name
Date: 2021-03-08 08:39:21
LastEditTime: 2021-04-13 09:14:37
LastEditors: Please set LastEditors
Description: In User Settings Edit
FilePath: /code_for_project/models/ore_model.py
'''
import torch
from torch import log_softmax
import torch.nn as nn
from torch.nn.modules.loss import CrossEntropyLoss
from transformers import BertModel
from torchcrf import CRF

from models.Attention import BasicAttention


class OREModel(nn.Module):
    def __init__(self, flags, bertconfig):
        super(OREModel, self).__init__()
        self.label_num = len(flags.rel_map)
        bertconfig.num_labels = self.label_num
        bertconfig.return_dict = True
        bertconfig.output_hidden_states = True

        self.fuse = flags.fuse
        self.knowledges = flags.knowledges
        self.decoder = flags.decoder

        # local bert
        self.bert = BertModel.from_pretrained(flags.pretrained, config=bertconfig)
        self.bn = nn.BatchNorm1d(flags.max_length)
        self.dropout = nn.Dropout(flags.dropout_rate)

        # feature fuse
        if self.fuse == "att":
            self.att = BasicAttention(bertconfig.hidden_size, bertconfig.hidden_size, bertconfig.hidden_size)
            # full connection layers
            self.concat2tag = nn.Linear(bertconfig.hidden_size, self.label_num)
        else:
            self.entEmb = nn.Embedding(len(flags.ent_map), embedding_dim=flags.feature_dim)
            # full connection layers
            self.concat2tag = nn.Linear(bertconfig.hidden_size, self.label_num)

        # decode layer
        if self.decoder == "crf":
            self.crf_layer = CRF(self.label_num, batch_first=True)
        else:
            self.loss = nn.CrossEntropyLoss()

    def forward(self, example):
        token = example["text"]
        mask = example["mask"]
        gold = example["gold"]
        token = self.bert(token, attention_mask=mask)[0]
        token_emb = self.dropout(token)

        if self.fuse == "att":
            query = example["query"]
            query = self.bert(query)[0]
            # batch_size, max_length, bert_hidden
            logits = self.att(query, token_emb, token_emb)
        else:
            ent = example["ent"]
            logits = torch.cat([token_emb, ent], dim=-1)
        # BERT's last hidden layer

        # feature concat, fc layer
        logits = self.concat2tag(logits)

        if self.decoder == "crf":
            # crf loss
            loss = - self.crf_layer(logits, gold, mask=mask, reduction="mean")
            loss += - self.crf_layer(logits, gold, mask=mask, reduction="mean")
            pred = torch.Tensor(self.crf_layer.decode(logits)).cuda()
        else:
            # softmax loss
            loss = self.loss(logits.view(-1, self.label_num), gold.view(-1))
            pred = torch.max(log_softmax(logits, dim=-1), dim=-1).indices

        zero = torch.zeros(*gold.shape, dtype=gold.dtype).cuda()
        eq = torch.eq(pred, gold.float())
        acc = torch.sum(eq * mask.float()) / torch.sum(mask.float())
        zero_acc = torch.sum(torch.eq(zero, gold.float())
                             * mask.float()) / torch.sum(mask.float())

        return loss, acc, zero_acc

    def decode(self, example):
        token = example["text"]
        mask = example["mask"]
        token = self.bert(token, attention_mask=mask)[0]
        token_emb = self.dropout(token)

        if self.fuse == "att":
            query = example["query"]
            query = self.bert(query)[0]
            # batch_size, max_length, bert_hidden
            logits = self.att(query, token_emb, token_emb)
        else:
            ent = example["ent"]
            logits = torch.cat([token_emb, ent], dim=-1)
        # BERT's last hidden layer

        # feature concat, fc layer
        logits = self.concat2tag(logits)

        if self.decoder == "crf":
            # crf decode
            tag_seq = self.crf_layer.decode(logits, mask=mask)
        else:
            # softmax decode
            tag_seq = torch.max(log_softmax(logits, dim=-1), dim=-1).indices
            tag_seq = tag_seq.cpu().detach().numpy().tolist()
        return tag_seq
