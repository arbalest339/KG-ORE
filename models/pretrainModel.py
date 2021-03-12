'''
Author: your name
Date: 2020-10-21 09:11:24
LastEditTime: 2021-03-12 11:01:06
LastEditors: Please set LastEditors
Description: model defination
FilePath: /code_for_naacl/models/main_model.py
'''
from re import S
import torch
from torch import embedding, log_softmax, sigmoid, softmax, zeros
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable
import numpy as np
from torch.nn.modules.loss import CrossEntropyLoss
from transformers import BertModel, BertForTokenClassification
from models.aggcn import AGGCN
from models.TransModels import TransD, TransE
from torchcrf import CRF


class PretrainModel(nn.Module):
    """Model for TransD training

    Args:
        nn ([type]): [description]
    """

    def __init__(self, flags, bertconfig):
        super(PretrainModel, self).__init__()
        self.dp_num = len(flags.dp_map)
        self.trans_select = flags.trans_select

        # Pre-trained BERT model
        self.bert = BertModel.from_pretrained(
            flags.pretrained, config=bertconfig)
        # Dropout to avoid overfitting
        self.dropout = nn.Dropout(flags.dropout_rate)

        # TransD
        if self.trans_select == "TransD":
            self.transd = TransD(bertconfig.vocab_size, self.dp_num,
                                 dim_e=bertconfig.hidden_size, dim_r=flags.feature_dim, p_norm=1, norm_flag=True, margin=flags.margin)
        elif self.trans_select == "TransE":
            self.transe = TransE(bertconfig.vocab_size, self.dp_num,
                                 dim=flags.feature_dim, margin=flags.margin)

    def forward(self, token, arc, mask):
        # BERT's last hidden layer
        bert_hidden = self.bert(
            token, encoder_attention_mask=mask, attention_mask=mask).last_hidden_state
        # batch_size, max_length, bert_hidden
        bert_emb = self.dropout(bert_hidden)

        bert_flatten = bert_emb.view(-1, bert_emb.shape[-1])

        # TransD
        # entity(word) embedding  想用索引，需要先转化为2维再还原回去
        emb_h = arc[:, :, 0].view(-1)
        emb_h = bert_flatten[emb_h].reshape(bert_emb.shape)
        emb_t = arc[:, :, -1].view(-1)
        emb_t = bert_flatten[emb_t].reshape(bert_emb.shape)
        # entity(word), relation(dp) index
        token_flatten = token.view(-1)
        batch_h = arc[:, :, 0].view(-1)
        batch_h = token_flatten[batch_h].reshape(token.shape)
        batch_r = arc[:, :, 1]
        batch_t = arc[:, :, -1].view(-1)
        batch_t = token_flatten[batch_t].reshape(token.shape)

        # TransD loss
        if self.trans_select == "TransD":
            trans_loss = self.transd(
                emb_h, emb_t, batch_h, batch_t, batch_r, mask)
        elif self.trans_select == "TransE":
            trans_loss = self.transe(emb_h, emb_t, batch_r, mask)

        return trans_loss

    def encoder(self, token, arc, mask):
        # BERT's last hidden layer
        bert_hidden = self.bert(
            token, encoder_attention_mask=mask, attention_mask=mask).last_hidden_state
        # batch_size, max_length, bert_hidden
        bert_emb = self.dropout(bert_hidden)

        bert_flatten = bert_emb.view(-1, bert_emb.shape[-1])

        # TransD
        # entity(word) embedding  想用索引，需要先转化为2维再还原回去
        emb_h = arc[:, :, 0].view(-1)
        emb_h = bert_flatten[emb_h].reshape(bert_emb.shape)
        emb_t = arc[:, :, -1].view(-1)
        emb_t = bert_flatten[emb_t].reshape(bert_emb.shape)
        # entity(word), relation(dp) index
        token_flatten = token.view(-1)
        batch_h = arc[:, :, 0].view(-1)
        batch_h = token_flatten[batch_h].reshape(token.shape)
        batch_r = arc[:, :, 1]
        batch_t = arc[:, :, -1].view(-1)
        batch_t = token_flatten[batch_t].reshape(token.shape)

        h, r, t = self.transd.getEmb(emb_h, emb_t, batch_h, batch_t, batch_r)
        return r, bert_emb  # r是dp embedding t是token embedding

    # def save_embedding(self):
    #     dps = [i for i in range(self.dp_num)]
    #     dps = torch.LongTensor(dps).cuda()
    #     dp_emb = self.encoder(dps).cpu().detach().numpy()
    #     np.save(self.dp_save_path, dp_emb)
