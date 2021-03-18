"""
read training/test/predict data
"""
from config import FLAGS
import json
import torch
import torch.utils.data as data


class NERDataset(data.Dataset):
    def __init__(self, data_path, tokenizer, max_length, mode, use_cuda=True):
        self.use_cuda = use_cuda
        self.max_length = max_length - 2
        self.tokenizer = tokenizer
        self.mode = mode

        self.label_map = FLAGS.label_map
        with open(data_path) as rf:
            self.datas = rf.readlines()
            self.num_example = len(self.datas)

    def load_train(self, line):
        line = json.loads(line)
        token, pos, gold = \
            list(line["token"]), line["pos"], line["gold"]
        gold = ["O" if tag == "O" or "REL" in tag else tag for tag in gold]

        # padding
        token_length = len(token)
        pad_length = self.max_length - token_length
        if pad_length >= 0:
            # BERT special token
            token = ["[CLS]"] + token + ["[SEP]"]
            pos = [0] + pos + [0]
            gold = ["O"] + gold + ["O"]
            # padding
            token += ["[PAD]"] * pad_length
            pos += [0] * pad_length
            gold += ["O"] * pad_length
            # mask pad
            mask = [1] * (token_length + 2) + [0] * pad_length
        else:
            token = ["[CLS]"] + token[:self.max_length] + ["[SEP]"]
            pos = [0] + pos[:self.max_length] + [0]
            gold = ["O"] + gold[:self.max_length] + ["O"]
            mask = [1] * (self.max_length + 2)

        acc_mask = [1 if g != "O" else 0 for g in gold]
        acc_mask[0] = 1

        # 数字化
        token = self.tokenizer.convert_tokens_to_ids(token)
        gold = [self.label_map[g] for g in gold]

        # tensor化
        token = torch.LongTensor(token).cuda(
        ) if self.use_cuda else torch.LongTensor(token)
        pos = torch.LongTensor(pos).cuda(
        ) if self.use_cuda else torch.LongTensor(pos)
        gold = torch.LongTensor(gold).cuda(
        ) if self.use_cuda else torch.LongTensor(gold)
        # dp = torch.LongTensor(dp).cuda(
        # ) if self.use_cuda else torch.LongTensor(dp)
        # head = torch.LongTensor(head).cuda(
        # ) if self.use_cuda else torch.LongTensor(head)
        mask = torch.BoolTensor(mask).cuda(
        ) if self.use_cuda else torch.BoolTensor(mask)
        acc_mask = torch.BoolTensor(acc_mask).cuda(
        ) if self.use_cuda else torch.BoolTensor(acc_mask)
        return [token, pos, gold, mask, acc_mask]

    def load_test(self, line):
        line = json.loads(line)
        # 文件中读取基础数据
        token, pos, gold = \
            list(line["token"]), line["pos"], line["gold"]
        gold = ["O" if tag == "O" or "REL" in tag else tag for tag in gold]

        # padding
        token_length = len(token)
        pad_length = self.max_length - token_length
        if pad_length >= 0:
            # BERT special token
            token = ["[CLS]"] + token + ["[SEP]"]
            gold = ["O"] + gold + ["O"]
            pos = [0] + pos + [0]
            # padding
            token += ["[PAD]"] * pad_length
            pos += [0] * pad_length
            gold += ["O"] * pad_length
            # dp arcs pad
            # mask pad
            mask = [1] * (token_length + 2) + [0] * pad_length
        else:
            token = ["[CLS]"] + token[:self.max_length] + ["[SEP]"]
            gold = ["O"] + gold[:self.max_length] + ["O"]
            mask = [1] * (self.max_length + 2)
        # 数字化
        token = self.tokenizer.convert_tokens_to_ids(token)
        gold = [self.label_map[g] for g in gold]

        # tensor化
        token = torch.LongTensor(token).cuda(
        ) if self.use_cuda else torch.LongTensor(token)
        pos = torch.LongTensor(pos).cuda(
        ) if self.use_cuda else torch.LongTensor(pos)
        gold = torch.LongTensor(gold).cuda(
        ) if self.use_cuda else torch.LongTensor(gold)
        mask = torch.ByteTensor(mask).cuda(
        ) if self.use_cuda else torch.ByteTensor(mask)

        return [token, pos, gold, mask]

    def __len__(self):
        return self.num_example

    def __getitem__(self, idx):
        line = self.datas[idx]
        # 文件中读取基础数据
        if self.mode == "train":
            return self.load_train(line)
        else:
            return self.load_test(line)
