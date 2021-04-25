'''
Author: your name
Date: 2021-03-08 08:39:21
LastEditTime: 2021-04-25 09:18:54
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


class OREModel(nn.Module):
    def __init__(self, flags, bertconfig):
        super(OREModel, self).__init__()
        bertconfig.return_dict = True
        bertconfig.output_hidden_states = True
        self.num_labels = 2
        self.features = flags.features

        # local bert
        self.bert = BertForQuestionAnswering.from_pretrained(
            flags.pretrained, config=bertconfig)

        # feature emb
        self.qa_hidden = bertconfig.hidden_size
        # if "desc" in self.features:
        #     self.desc = nn.Embedding(len(flags.pos_map), flags.feature_dim)
        # if "exrest" in self.features:
        #     self.exrest = nn.Embedding(len(flags.pos_map), flags.feature_dim)
        # if "kbRel" in self.features:
        #     self.kbRel = nn.Embedding(len(flags.pos_map), flags.feature_dim)

        # full connection layer
        self.qa_outputs = nn.Linear(self.qa_hidden, self.num_labels)

        # predict
        self.sofmax = torch.nn.Softmax(dim=-1)

    def forward(self, input_ids, mask, type_ids, start_ids, end_ids):
        outputs = self.bert(input_ids=input_ids, attention_mask=mask,
                            token_type_ids=type_ids, start_positions=start_ids, end_positions=end_ids)
        # BERT's last hidden layer
        # outputs = self.bert(input_ids, attention_mask=mask, token_type_ids=type_ids)
        # sequence_output = outputs[0]

        # # feature concat
        # # if "desc" in self.features:
        # #     descEmb = self.desc(desc)
        # # if "exrest" in self.features:
        # #     exrestEmb = self.exrest(exrest)
        # # if "kbRel" in self.features:
        # #     kbRelEmb = self.kbRel(kbRel)
        # # sequence_output = torch.concat()

        # # fc layer
        # logits = self.qa_outputs(sequence_output)
        # start_logits, end_logits = logits.split(1, dim=-1)
        # start_logits = start_logits.squeeze(-1)
        # end_logits = end_logits.squeeze(-1)

        # total_loss = None
        # if start_ids is not None and end_ids is not None:
        #     # If we are on multi-GPU, split add a dimension
        #     if len(start_ids.size()) > 1:
        #         start_ids = start_ids.squeeze(-1)
        #     if len(end_ids.size()) > 1:
        #         end_ids = end_ids.squeeze(-1)
        #     # sometimes the start/end positions are outside our model inputs, we ignore these terms
        #     ignored_index = start_logits.size(1)
        #     start_ids.clamp_(0, ignored_index)
        #     end_ids.clamp_(0, ignored_index)

        #     loss_fct = CrossEntropyLoss(ignore_index=ignored_index)
        #     start_loss = loss_fct(start_logits, start_ids)
        #     end_loss = loss_fct(end_logits, end_ids)
        #     total_loss = (start_loss + end_loss) / 2

        total_loss = outputs.loss
        start_logits = outputs.start_logits
        end_logits = outputs.end_logits
        _, pred_s = torch.max(self.sofmax(start_logits), dim=-1)
        _, end_s = torch.max(self.sofmax(end_logits), dim=-1)
        output = (pred_s, end_s)     # + outputs[2:]
        # return ((total_loss,) + output) if total_loss is not None else output
        return (total_loss, pred_s, end_s)

    def decode(self, token, pos, mask):
        # BERT's last hidden layer
        bert_hidden = self.bert(
            token, attention_mask=mask).hidden_states[-1]
        bert_hidden = self.bn(bert_hidden)
        # batch_size, max_length, bert_hidden
        token_emb = self.dropout(bert_hidden)

        if "pos" in self.features:
            pos_emb = self.posEmb(pos)

        # feature concat, fc layer
        logits = token_emb
        if "pos" in self.features:
            logits = torch.cat([logits, pos_emb], dim=-1)
        logits = self.concat2tag(logits)

        if self.decoder == "crf":
            # crf decode
            tag_seq = self.crf_layer.decode(logits, mask=mask)
        else:
            # softmax decode
            tag_seq = torch.max(log_softmax(logits, dim=-1), dim=-1).indices
            tag_seq = tag_seq.cpu().detach().numpy().tolist()
        return tag_seq
