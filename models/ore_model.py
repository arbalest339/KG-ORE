'''
Author: your name
Date: 2021-03-08 08:39:21
LastEditTime: 2021-03-10 15:08:55
LastEditors: Please set LastEditors
Description: In User Settings Edit
FilePath: /code_for_project/models/ore_model.py
'''
import torch
from torch import embedding, log_softmax, softmax
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable
import numpy as np
from torch.nn.modules.loss import CrossEntropyLoss
from transformers import BertForTokenClassification
from models.trans_model import TransModel
from torchcrf import CRF


class OREModel(nn.Module):
    def __init__(self, flags, bertconfig):
        super(OREModel, self).__init__()
        self.label_num = len(flags.label_map)
        bertconfig.num_labels = self.label_num
        bertconfig.return_dict = True
        bertconfig.output_hidden_states = True
        # self.bert = BertForTokenClassification.from_pretrained(flags.pretrained, config=bertconfig)

        self.transModel = TransModel(flags, bertconfig)
        self.transModel.load_state_dict(torch.load(flags.pretrained_checkpoint_path))

        # feature emb
        self.nerEmb = nn.Embedding(len(flags.ner_map), flags.feature_dim)
        self.posEmb = nn.Embedding(len(flags.pos_map), flags.feature_dim)

        # full connection layers
        self.concat2tag = nn.Linear(flags.feature_dim*4, self.label_num)    # bertconfig.hidden_size + 

        # CRF layer
        self.crf_layer = CRF(self.label_num, batch_first=True)
        self.loss = nn.CrossEntropyLoss()

    def forward(self, token, pos, ner, arc, gold, mask, acc_mask):
        # BERT's last hidden layer
        # bert_hidden = self.bert(token, labels=gold, attention_mask=mask).hidden_states[-1]
        # bert_hidden = self.bn(bert_hidden)
        # bert_hidden = self.dropout(bert_hidden)     # batch_size, max_length, bert_hidden

        dp_emb, token_emb = self.transModel.encoder(token, arc, mask)
        pos_emb = self.posEmb(pos)
        ner_emb = self.posEmb(ner)

        # feature concat, fc layer
        logits = torch.cat([token_emb, dp_emb, pos_emb, ner_emb], dim=-1)
        logits = self.concat2tag(logits)

        # crf loss
        loss = - self.crf_layer(logits, gold, mask=mask, reduction="mean")
        # crf_loss = self.loss(logits.view(-1, self.label_num), gold.view(-1))
        pred = torch.Tensor(self.crf_layer.decode(logits)).cuda()
        # pred = torch.max(log_softmax(logits, dim=-1), dim=-1).indices
        zero = torch.zeros(*gold.shape, dtype=gold.dtype).cuda()
        eq = torch.eq(pred, gold.float())
        acc = torch.sum(eq * acc_mask.float()) / torch.sum(acc_mask.float())
        zero_acc = torch.sum(torch.eq(zero, gold.float()) * mask.float()) / torch.sum(mask.float())

        return loss, acc, zero_acc

    def decode(self, token, pos, ner, arc, mask):
        # bert_hidden = self.bert(token, attention_mask=mask).hidden_states[-1]
        # bert_hidden = self.bn(bert_hidden)
        # bert_hidden = self.dropout(bert_hidden)     # batch_size, max_length, bert_hidden

        dp_emb, token_emb = self.transModel.encoder(token, arc, mask)
        pos_emb = self.posEmb(pos)
        ner_emb = self.posEmb(ner)

        # feature concat, fc layer
        logits = torch.cat([token_emb, dp_emb, pos_emb, ner_emb], dim=-1)
        logits = self.concat2tag(logits)

        # crf decode
        tag_seq = self.crf_layer.decode(logits, mask=mask)
        # tag_seq = torch.max(log_softmax(logits[mask.bool()], dim=-1), dim=-1).indices
        # tag_seq = tag_seq.cpu().detach().numpy().tolist()
        return tag_seq
