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
        self.knowledges = FLAGS.knowledges

        with open(data_path) as rf:
            self.datas = rf.readlines()
            self.num_example = len(self.datas)

    def loadSample(self, line):
        example = {}
        line = json.loads(line)
        text, query, answer = line["text"], line["query"], line["answer"]
        start_id = answer[0]    # + len(query)+2
        end_id = answer[1]  # + len(query)+2
        if "desc" in self.knowledges:
            desc1, desc2 = line["desc1"], line["desc2"]
            if not desc1:
                desc1 = ""
            if not desc2:
                desc2 = ""
        if "exrest" in self.knowledges:
            exrest1, exrest2 = line["exrest1"], line["exrest2"]
            if not exrest1:
                exrest1 = ""
            if not exrest2:
                exrest2 = ""
        if "kbRel" in self.knowledges:
            kbRel = line["kbRel"]

        # 数字化
        if not self.knowledges or ("kbRel" in self.knowledges and len(self.knowledges) == 1):  # 无辅助信息
            if "kbRel" in self.knowledges and kbRel:
                kbRel = line["kbRel"]
                triples = [query.replace("?", f" {rel} ") for rel in kbRel]
                query = "或者".join(triples)
            inputs = self.tokenizer(query, text, padding='max_length', truncation=True, max_length=self.max_length, return_tensors='pt')
            input_ids, mask, type_ids = \
                inputs["input_ids"].squeeze(), inputs["attention_mask"].squeeze(), inputs["token_type_ids"].squeeze()

            # tensor化
            if self.use_cuda:
                input_ids, mask, type_ids = input_ids.cuda(), mask.cuda(), type_ids.cuda()
                start_id = torch.LongTensor([start_id]).cuda()
                end_id = torch.LongTensor([end_id]).cuda()
            else:
                start_id, end_id = torch.LongTensor([start_id]), torch.LongTensor([end_id])
            example["input_ids"] = input_ids
            example["mask"] = mask
            example["type_ids"] = type_ids
            example["start_id"] = start_id
            example["end_id"] = end_id
            return example
        else:
            text = self.tokenizer(text, padding='max_length', truncation=True, max_length=self.max_length, return_tensors='pt')
            text = text["input_ids"].squeeze()
            text = torch.LongTensor(text).cuda() if self.use_cuda else torch.LongTensor(text)
            start_id = torch.LongTensor([start_id]).cuda() if self.use_cuda else torch.LongTensor([start_id])
            end_id = torch.LongTensor([end_id]).cuda() if self.use_cuda else torch.LongTensor([end_id])

            if "kbRel" in self.knowledges and kbRel:
                kbRel = line["kbRel"]
                triples = [query.replace("?", f" {rel} ") for rel in kbRel]
                query = "或者".join(triples)
                query = self.tokenizer(query, padding='max_length', truncation=True, max_length=self.max_length, return_tensors='pt')
                query = query["input_ids"].squeeze()
            else:
                query = self.tokenizer(query, padding='max_length', truncation=True, max_length=self.max_length, return_tensors='pt')
                query = query["input_ids"].squeeze()
            query = torch.LongTensor(query).cuda() if self.use_cuda else torch.LongTensor(query)

            example["text"] = text
            example["query"] = query
            example["start_id"] = start_id
            example["end_id"] = end_id

            if "desc" in self.knowledges:
                desc1 = self.tokenizer(desc1, padding='max_length', truncation=True, max_length=self.max_length, return_tensors='pt')
                desc2 = self.tokenizer(desc2, padding='max_length', truncation=True, max_length=self.max_length, return_tensors='pt')
                desc1 = desc1["input_ids"].squeeze()
                desc2 = desc2["input_ids"].squeeze()
                desc1 = torch.LongTensor(desc1).cuda() if self.use_cuda else torch.LongTensor(desc1)
                desc2 = torch.LongTensor(desc2).cuda() if self.use_cuda else torch.LongTensor(desc2)
                example["desc1"] = desc1
                example["desc2"] = desc2
            if "exrest" in self.knowledges:
                exrest1 = self.tokenizer(exrest1, padding='max_length', truncation=True, max_length=self.max_length, return_tensors='pt')
                exrest2 = self.tokenizer(exrest2, padding='max_length', truncation=True, max_length=self.max_length, return_tensors='pt')
                exrest1 = exrest1["input_ids"].squeeze()
                exrest2 = exrest2["input_ids"].squeeze()
                exrest1 = torch.LongTensor(exrest1).cuda() if self.use_cuda else torch.LongTensor(exrest1)
                exrest2 = torch.LongTensor(exrest2).cuda() if self.use_cuda else torch.LongTensor(exrest2)
                example["exrest1"] = exrest1
                example["exrest2"] = exrest2
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
