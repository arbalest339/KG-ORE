"""
read training/test/predict data
"""
from torch import randint
from config import FLAGS
import json
import torch
import random
import torch.utils.data as data
import numpy as np


class OREDataset(data.Dataset):
    def __init__(self, data_path, tokenizer, max_length, mode, use_cuda=True):
        self.use_cuda = use_cuda
        self.max_length = max_length - 2
        self.tokenizer = tokenizer
        self.mode = mode

        self.label_map = FLAGS.label_map
        self.ner_map = FLAGS.ner_map
        with open(data_path) as rf:
            self.datas = rf.readlines()
            self.num_example = len(self.datas)

    def load_train(self, line):
        line = json.loads(line)
        token, pos, dp, head, gold = \
            list(line["token"]), line["pos"], line["dp"], line["head"], line["gold"]
        ner = [self.ner_map["O"] if tag ==
               "O" or "REL" in tag else self.ner_map[tag] for tag in gold]
        gold = [tag if "REL" in tag else "O" for tag in gold]
        arc = [[hr[0]+1, hr[1], i+1] for i, hr in enumerate(zip(head, dp))]

        # padding
        token_length = len(token)
        pad_length = self.max_length - token_length
        if pad_length >= 0:
            # BERT special token
            token = ["[CLS]"] + token + ["[SEP]"]
            pos = [0] + pos + [0]
            ner = [0] + ner + [0]
            gold = ["O"] + gold + ["O"]
            # padding
            token += ["[PAD]"] * pad_length
            pos += [0] * pad_length
            ner += [0] * pad_length
            gold += ["O"] * pad_length
            # dp arcs pad
            arc = [[0,0,0]] + arc + [[0,0,0]]
            for p in range(pad_length):     # 为arc pad的是随机产生的负样本
                neg = arc[-1]
                while neg in arc:
                    neg = [int(random.uniform(1, token_length+1)), int(random.uniform(0, len(FLAGS.dp_map))), int(random.uniform(1, token_length+1))]
                arc += [neg]
            #     dp += [0]
            #     head += [0]
            # mask pad
            mask = [1] * (token_length + 2) + [0] * pad_length
        else:
            token = ["[CLS]"] + token[:self.max_length] + ["[SEP]"]
            pos = [0] + pos[:self.max_length] + [0]
            ner = [0] + ner[:self.max_length] + [0]
            gold = ["O"] + gold[:self.max_length] + ["O"]
            mask = [1] * (self.max_length + 2)
            arc = [[0,0,0]] + arc[:self.max_length] + [[0,0,0]]
            # if pad_length == -1:    # 127
            #     dp += [0]
            #     head += [0]
            # else:
            #     dp = dp[:(self.max_length + 2)]
            #     head = head[:(self.max_length + 2)]

        acc_mask = [1 if g != "O" else 0 for g in gold]
        acc_mask[0] = 1

        # 数字化
        token = self.tokenizer.convert_tokens_to_ids(token)
        # head = [token[h] if h > 0 else 0 for h in head]
        gold = [self.label_map[g] for g in gold]

        # tensor化
        token = torch.LongTensor(token).cuda(
        ) if self.use_cuda else torch.LongTensor(token)
        pos = torch.LongTensor(pos).cuda(
        ) if self.use_cuda else torch.LongTensor(pos)
        ner = torch.LongTensor(ner).cuda(
        ) if self.use_cuda else torch.LongTensor(ner)
        gold = torch.LongTensor(gold).cuda(
        ) if self.use_cuda else torch.LongTensor(gold)
        arc = torch.LongTensor(arc).cuda(
        ) if self.use_cuda else torch.LongTensor(arc)
        # dp = torch.LongTensor(dp).cuda(
        # ) if self.use_cuda else torch.LongTensor(dp)
        # head = torch.LongTensor(head).cuda(
        # ) if self.use_cuda else torch.LongTensor(head)
        mask = torch.BoolTensor(mask).cuda(
        ) if self.use_cuda else torch.BoolTensor(mask)
        acc_mask = torch.BoolTensor(acc_mask).cuda(
        ) if self.use_cuda else torch.BoolTensor(acc_mask)
        return [token, pos, ner, arc, gold, mask, acc_mask]

    def load_test(self, line):
        line = json.loads(line)
        # 文件中读取基础数据
        token, pos, dp, head, gold = \
            list(line["token"]), line["pos"], line["dp"], line["head"], line["gold"]
        ner = [self.ner_map["O"] if tag ==
               "O" or "REL" in tag else self.ner_map[tag] for tag in gold]
        gold = [tag if "REL" in tag else "O" for tag in gold]
        arc = [[hr[0]+1, hr[1], i+1] for i, hr in enumerate(zip(head, dp))]

        # padding
        token_length = len(token)
        pad_length = self.max_length - token_length
        if pad_length >= 0:
            # BERT special token
            token = ["[CLS]"] + token + ["[SEP]"]
            pos = [0] + pos + [0]
            ner = [0] + ner + [0]
            gold = ["O"] + gold + ["O"]
            # padding
            token += ["[PAD]"] * pad_length
            pos += [0] * pad_length
            ner += [0] * pad_length
            # dp arcs pad
            arc = [[0,0,0]] + arc + [[0,0,0]]
            for p in range(pad_length):     # 为arc pad的是随机产生的负样本
                neg = arc[-1]
                while neg in arc:
                    neg = [int(random.uniform(1, token_length+1)), int(random.uniform(0, len(FLAGS.dp_map))), int(random.uniform(1, token_length+1))]
                arc += [neg]
            gold += ["O"] * pad_length
            # mask pad
            mask = [1] * (token_length + 2) + [0] * pad_length
        else:
            token = ["[CLS]"] + token[:self.max_length] + ["[SEP]"]
            pos = [0] + pos[:self.max_length] + [0]
            ner = [0] + ner[:self.max_length] + [0]
            gold = ["O"] + gold[:self.max_length] + ["O"]
            mask = [1] * (self.max_length + 2)
            arc = [[0,0,0]] + arc[:self.max_length] + [[0,0,0]]
            # if pad_length == -1:    # 127
            #     dp += [0]
            #     head += [0]
            # else:
            #     dp = dp[:(self.max_length + 2)]
            #     head = head[:(self.max_length + 2)]

        # 数字化
        token = self.tokenizer.convert_tokens_to_ids(token)
        # head = [token[h] if h > 0 else 0 for h in head]
        # head = self.tokenizer.convert_tokens_to_ids(head)
        gold = [self.label_map[g] for g in gold]

        # tensor化
        token = torch.LongTensor(token).cuda(
        ) if self.use_cuda else torch.LongTensor(token)
        pos = torch.LongTensor(pos).cuda(
        ) if self.use_cuda else torch.LongTensor(pos)
        ner = torch.LongTensor(ner).cuda(
        ) if self.use_cuda else torch.LongTensor(ner)
        arc = torch.LongTensor(arc).cuda(
        ) if self.use_cuda else torch.LongTensor(arc)
        # dp = torch.LongTensor(dp).cuda(
        # ) if self.use_cuda else torch.LongTensor(dp)
        # head = torch.LongTensor(head).cuda(
        # ) if self.use_cuda else torch.LongTensor(head)
        gold = torch.LongTensor(gold).cuda(
        ) if self.use_cuda else torch.LongTensor(gold)
        mask = torch.ByteTensor(mask).cuda(
        ) if self.use_cuda else torch.ByteTensor(mask)

        return [token, pos, ner, arc, gold, mask]

    def __len__(self):
        return self.num_example

    def __getitem__(self, idx):
        line = self.datas[idx]
        # 文件中读取基础数据
        if self.mode == "train":
            return self.load_train(line)
        else:
            return self.load_test(line)


class AnalyseDataset(data.Dataset):
    def __init__(self, data_path, matrix_path, tokenizer, max_length, task, use_cuda=True):
        self.task = task
        self.use_cuda = use_cuda
        self.max_length = max_length - 2
        self.label_map = FLAGS.label_map
        self.load_test(data_path, matrix_path, tokenizer)

    def load_test(self, data_path, matrix_path, tokenizer):
        self.data = list()
        datas = json.load(open(data_path))
        matrixs = np.load(matrix_path, allow_pickle=True)

        for line, matrix in zip(datas, matrixs):
            # 文件中读取基础数据
            token, pos, ner, gold = \
                line["token"], line["pos"], line["ner"], line["gold"]

            if self.task == "ore":
                e1, e2, r = line["e1"], line["e2"], line["pred"]
            else:
                e1, e2, r = line["sub"], line["obj"], line["pred"]

            token_str = [t for t in token]

            arc = []     # for TransD
            for i, row in enumerate(matrix):
                for j, t in enumerate(row):
                    if int(t) > 0:
                        arc.append([i, int(t), j])
                        matrix[i][j] = 1.0
                    else:
                        matrix[i][j] = 0.0
            arc.sort(key=lambda x: x[2])
            dp = [t[1] for t in arc]
            head = [token[t[0]-1] if t[0] != 0 else "[CLS]" for t in arc]

            # padding
            token_length = len(token)
            pad_length = self.max_length - token_length
            if pad_length >= 0:
                # BERT special token
                token = ["[CLS]"] + token + ["[SEP]"]
                pos = [0] + pos + [0]
                ner = [0] + ner + [0]
                # padding
                token += ["[PAD]"] * pad_length
                pos += [0] * pad_length
                ner += [0] * pad_length
                # dp arcs pad
                for p in range(pad_length + 2):     # 为arc pad的是随机产生的负样本
                    dp += [0]
                    head += ["[PAD]"]
                # mask pad
                mask = [1] * (token_length + 2) + [0] * pad_length
                # matrix pad
                for i, row in enumerate(matrix):
                    matrix[i] = row + [0.0] * (pad_length + 1)
                for i in range(pad_length + 1):
                    matrix.append([0.0] * (self.max_length + 2))
            else:
                token = ["[CLS]"] + token[:self.max_length] + ["[SEP]"]
                pos = [0] + pos[:self.max_length] + [0]
                ner = [0] + ner[:self.max_length] + [0]
                mask = [1] * (self.max_length + 2)
                if pad_length == -1:    # 127
                    dp += [0]
                    head += ["[PAD]"]
                else:
                    dp = dp[:(self.max_length + 2)]
                    head = head[:(self.max_length + 2)]
                    matrix = [mr[:self.max_length+2]
                              for mr in matrix[:self.max_length+2]]

            # 数字化
            token = tokenizer.convert_tokens_to_ids(token)
            head = tokenizer.convert_tokens_to_ids(head)

            # tensor化
            token = torch.LongTensor(token).cuda(
            ) if self.use_cuda else torch.LongTensor(token)
            pos = torch.LongTensor(pos).cuda(
            ) if self.use_cuda else torch.LongTensor(pos)
            ner = torch.LongTensor(ner).cuda(
            ) if self.use_cuda else torch.LongTensor(ner)
            dp = torch.LongTensor(dp).cuda(
            ) if self.use_cuda else torch.LongTensor(dp)
            head = torch.LongTensor(head).cuda(
            ) if self.use_cuda else torch.LongTensor(head)
            mask = torch.BoolTensor(mask).cuda(
            ) if self.use_cuda else torch.BoolTensor(mask)
            matrix = torch.Tensor(matrix).cuda(
            ) if self.use_cuda else torch.Tensor(matrix)

            self.data.append([token, pos, ner, dp, head, matrix,
                              gold, mask, token_str, e1, e2, r])
