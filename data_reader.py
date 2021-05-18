'''
Author: your name
Date: 2021-04-13 09:07:21
LastEditTime: 2021-05-05 17:20:03
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
        self.ent_map = FLAGS.ent_map
        self.rel_map = FLAGS.rel_map
        self.knowledges = FLAGS.knowledges
        self.fuse = FLAGS.fuse

        with open(data_path, encoding="utf-8") as rf:
            self.datas = rf.readlines()
            self.num_example = len(self.datas)

    def loadSample(self, line):
        example = {}
        line = json.loads(line)
        if "query" in line:
            text, query, answer = line["text"], line["query"], line["answer"]
            e1, e2 = query.split("?")
        else:
            text, e1, e2, answer = line["text"], line["e1"], line["e2"], line["answer"]
        if "kbRel" in self.knowledges:
            kbRel = line["kbRel"]

        if self.fuse == "att":
            query = f"{e1}?{e2}"
            if "kbRel" in self.knowledges and kbRel:
                kbRel = line["kbRel"]
                triples = [query.replace("?", f" {rel} ") for rel in kbRel]
                query = "，".join(triples)
                query = self.tokenizer(query, padding='max_length', truncation=True, max_length=self.max_length // 2, return_tensors='pt')
                query = query["input_ids"].squeeze()
            else:
                query = self.tokenizer(query, padding='max_length', truncation=True, max_length=self.max_length // 2, return_tensors='pt')
                query = query["input_ids"].squeeze()
            query = torch.LongTensor(query).cuda() if self.use_cuda else torch.LongTensor(query)
            example["query"] = query
        else:
            e1_idx = text.index(e1) + 1
            e2_idx = text.index(e2) + 1
            ent = [self.ent_map["O"]] * self.max_length
            for i in range(e1_idx, e1_idx+len(e1)):
                if i == e1_idx and i < len(ent):
                    ent[i] = self.ent_map["B-E1"]
                elif i < len(ent):
                    ent[i] = self.ent_map["I-E1"]

                if i == e2_idx and i < len(ent):
                    ent[i] = self.ent_map["B-E2"]
                elif i < len(ent):
                    ent[i] = self.ent_map["I-E2"]
            example["ent"] = ent
        gold = [self.rel_map["O"]] * self.max_length
        for i in range(answer[0], answer[1]):
            if i == answer[0] and i < len(gold):
                gold[i] = self.rel_map["B-R"]
            elif i < len(gold):
                gold[i] = self.rel_map["I-R"]
        text = self.tokenizer(text, padding='max_length', truncation=True, max_length=self.max_length, return_tensors='pt')
        text = text["input_ids"].squeeze()
        mask = text["attention_mask"].squeeze()
        text = torch.LongTensor(text).cuda() if self.use_cuda else torch.LongTensor(text)
        mask = torch.LongTensor(mask).cuda() if self.use_cuda else torch.LongTensor(mask)

        example["text"] = text
        example["mask"] = mask
        example["gold"] = gold
        return example

    def getOrigin(self, idx):
        line = self.datas[idx]
        line = json.loads(line)
        start = line["answer"][0]
        end = line["answer"][1]
        return [line["text"], line["query"], line["text"][start:end]]

    def __len__(self):
        return self.num_example

    def __getitem__(self, idx):
        line = self.datas[idx]
        # 文件中读取基础数据
        return self.loadSample(line)
