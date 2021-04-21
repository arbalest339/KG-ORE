'''
Author: your name
Date: 2021-04-13 09:07:21
LastEditTime: 2021-04-20 14:34:51
LastEditors: Please set LastEditors
Description: In User Settings Edit
FilePath: /code_for_ore/data_reader.py
'''


from config import FLAGS
import json
import torch
import torch.utils.data as data


class OREDataset(data.Dataset):
    def __init__(self, data_path, tokenizer, max_length, use_cuda=True):
        self.use_cuda = use_cuda
        self.max_length = max_length
        self.tokenizer = tokenizer
        self.features = FLAGS.features

        with open(data_path) as rf:
            self.datas = rf.readlines()
            self.num_example = len(self.datas)

    def loadSample(self, line):
        line = json.loads(line)
        text, query, answer = line["text"], line["query"], line["answer"]
        if "desc" in self.features:
            desc1, desc2 = line["desc"], line["desc"]
        if "exrest" in self.features:
            exrest1, exrest2 = line["exrest1"], line["exrest2"]
        if "kbRel" in self.features:
            kbRel = line["kbRel"]

        # padding
        # question = "[CLS]" + query + "[SEP]" + answer
        # token_length = len(question)
        # pad_length = self.max_length - token_length
        # if pad_length >= 0:

        #     text += ["[PAD]"] * pad_length
        #     # mask pad
        #     mask = [1] * (token_length) + [0] * pad_length
        # else:
        #     text = text[:self.max_length]
        #     mask = [1] * self.max_length

        # 数字化
        inputs = self.tokenizer(query, text, padding='max_length', max_length=self.max_length, return_tensors='pt')
        input_ids, mask, type_ids = \
            inputs["input_ids"].squeeze(), inputs["attention_mask"].squeeze(), inputs["token_type_ids"].squeeze()

        # tensor化
        if self.use_cuda:
            input_ids, mask, type_ids = input_ids.cuda(), mask.cuda(), type_ids.cuda()
            start_id = torch.LongTensor([answer[0]]).cuda()
            end_id = torch.LongTensor([answer[1]]).cuda()
        else:
            start_id, end_id = torch.LongTensor([answer[0]]), torch.LongTensor([answer[1]])
        return [input_ids, mask, type_ids, start_id, end_id]

    def getOrigin(self, idx):
        line = self.datas[idx]
        line = json.loads(line)
        start = line["answer"][0]
        end = line["answer"][1]
        return line["text"], line["query"], line["text"][start:end]

    def __len__(self):
        return self.num_example

    def __getitem__(self, idx):
        line = self.datas[idx]
        # 文件中读取基础数据
        return self.loadSample(line)
