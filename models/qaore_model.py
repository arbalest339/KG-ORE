'''
Author: your name
Date: 2021-03-08 08:39:21
LastEditTime: 2021-05-06 14:04:25
LastEditors: Please set LastEditors
Description: In User Settings Edit
FilePath: /code_for_project/models/ore_model.py
'''
import torch
from torch import log_softmax
import torch.nn as nn
from torch.nn.modules.loss import CrossEntropyLoss
from transformers import BertModel, BertForQuestionAnswering
from torchcrf import CRF
from models.attention import BasicAttention


class OREModel(nn.Module):
    def __init__(self, flags, bertconfig):
        super(OREModel, self).__init__()
        bertconfig.return_dict = True
        # bertconfig.output_hidden_states = True
        self.num_labels = 2
        self.knowledges = flags.knowledges
        self.bert_hidden = bertconfig.hidden_size

        # local bert
        if not self.knowledges or ("kbRel" in self.knowledges and len(self.knowledges) == 1):
            self.bert = BertForQuestionAnswering.from_pretrained(
                flags.pretrained, config=bertconfig)
            self.fuse_hidden = self.bert_hidden
        else:
            self.bert = BertModel.from_pretrained(flags.pretrained, config=bertconfig)
            self.fuse_hidden = self.bert_hidden
            self.queryAtt = BasicAttention(self.bert_hidden, self.bert_hidden, self.bert_hidden)

        # feature
        if "desc" in self.knowledges:
            self.descAtt = BasicAttention(self.bert_hidden, self.bert_hidden, self.bert_hidden)
            # self.fuse_hidden += self.bert_hidden
        if "exrest" in self.knowledges:
            self.exrestAtt = BasicAttention(self.bert_hidden, self.bert_hidden, self.bert_hidden)
            # self.fuse_hidden += self.bert_hidden

        # full connection layer
        self.qa_outputs = nn.Linear(self.fuse_hidden, self.num_labels)

        # predict
        self.sofmax = torch.nn.Softmax(dim=-1)

    def forward(self, datas):
        if not self.knowledges or ("kbRel" in self.knowledges and len(self.knowledges) == 1):
            input_ids, mask, type_ids, start_ids, end_ids = \
                datas["input_ids"], datas["mask"], datas["type_ids"], datas["start_id"], datas["end_id"]
            outputs = self.bert(input_ids=input_ids, attention_mask=mask,
                                token_type_ids=type_ids, start_positions=start_ids, end_positions=end_ids)
            total_loss = outputs.loss
            start_logits = outputs.start_logits
            end_logits = outputs.end_logits
        else:
            text, query, start_ids, end_ids = datas["text"], datas["query"], datas["start_id"], datas["end_id"]
            # BERT's last hidden layer
            text = self.bert(text)
            text = text[0]
            query = self.bert(query)
            query = query[0]
            text = self.queryAtt(query, text, text)

            logits = torch.cat([query, text], dim=1)

            # feature fuse and concat
            if "desc" in self.knowledges:
                desc1, desc2 = datas["desc1"], datas["desc2"]
                desc1 = self.bert(desc1)[0]
                desc2 = self.bert(desc2)[0]
                desc1 = self.descAtt(query, desc1, desc1)
                desc2 = self.descAtt(query, desc2, desc2)
                logits = torch.cat([logits, desc1, desc2], dim=1)
            if "exrest" in self.knowledges:
                exrest1, exrest2 = datas["exrest1"], datas["exrest2"]
                exrest1 = self.bert(exrest1)[0]
                exrest2 = self.bert(exrest2)[0]
                exrest1 = self.exrestAtt(query, exrest1, exrest1)
                exrest2 = self.exrestAtt(query, exrest2, exrest2)
                logits = torch.cat([logits, exrest1, exrest2], dim=1)

            # fc layer
            logits = self.qa_outputs(logits)
            start_logits, end_logits = logits.split(1, dim=-1)
            start_logits = start_logits.squeeze(-1)
            end_logits = end_logits.squeeze(-1)

            total_loss = None
            if start_ids is not None and end_ids is not None:
                # If we are on multi-GPU, split add a dimension
                if len(start_ids.size()) > 1:
                    start_ids = start_ids.squeeze(-1)
                if len(end_ids.size()) > 1:
                    end_ids = end_ids.squeeze(-1)
                # sometimes the start/end positions are outside our model inputs, we ignore these terms
                ignored_index = start_logits.size(1)
                start_ids.clamp_(0, ignored_index)
                end_ids.clamp_(0, ignored_index)

                loss_fct = CrossEntropyLoss(ignore_index=ignored_index)
                start_loss = loss_fct(start_logits, start_ids)
                end_loss = loss_fct(end_logits, end_ids)
                total_loss = (start_loss + end_loss) / 2

        _, pred_s = torch.max(self.sofmax(start_logits), dim=-1)
        _, end_s = torch.max(self.sofmax(end_logits), dim=-1)
        output = (pred_s, end_s)     # + outputs[2:]
        # return ((total_loss,) + output) if total_loss is not None else output
        return (total_loss, pred_s, end_s)
