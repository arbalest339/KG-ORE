"""
test process entry
"""

import time
import json
import torch

from transformers import BertTokenizer, BertConfig
from models.ore_model import OREModel
from data_reader import OREDataset
from config import FLAGS


def en_metrics(e1, e2, r, tag_seq):
    positive_true = 0
    positive_false = 0
    negative_false = 0

    for e1p in e1:
        if tag_seq[e1p] in (1, 2):
            positive_true += 1
        else:
            negative_false += 1

    for e2p in e2:
        if tag_seq[e2p] in (3, 4):
            positive_true += 1
        else:
            negative_false += 1

    for rp in r:
        if tag_seq[rp] in (5, 6):
            positive_true += 1
        else:
            negative_false += 1

    for i, t in enumerate(tag_seq):
        if i not in e1 and t in (1, 2):
            positive_false += 1
        if i not in e2 and t in (3, 4):
            positive_false += 1
        if i not in r and t in (5, 6):
            positive_false += 1

    return positive_true, positive_false, negative_false


# def zh_metrics(e1, e2, r, tag_seq):
#     positive_true = 0
#     positive_false = 0
#     negative_false = 0

#     for e1p in e1:
#         if tag_seq[e1p + 1] in (1, 2):
#             positive_true += 1
#         else:
#             negative_false += 1

#     for e2p in e2:
#         if tag_seq[e2p + 1] in (3, 4):
#             positive_true += 1
#         else:
#             negative_false += 1

#     for rp in r:
#         if tag_seq[rp + 1] in (5, 6):
#             positive_true += 1
#         else:
#             negative_false += 1

#     for i, t in enumerate(tag_seq):
#         if i - 1 not in e1 and t in (1, 2):
#             positive_false += 1
#         if i - 1 not in e2 and t in (3, 4):
#             positive_false += 1
#         if i - 1 not in r and t in (5, 6):
#             positive_false += 1

#     return positive_true, positive_false, negative_false


def zh_metrics(gold, tag_seq):
    positive_true = 0
    positive_false = 0
    negative_false = 0

    for g, t in zip(gold, tag_seq):
        if g == t and g != 0:
            positive_true += 1
        elif g != 0 and g != t:
            positive_false += 1
        elif t != 0:
            negative_false += 1

    return positive_true, positive_false, negative_false


def test():
    # load from pretrained config file
    # bertconfig = json.load(open(FLAGS.pretrained_config))
    bertconfig = BertConfig.from_pretrained(FLAGS.pretrained)

    # Initiate model
    print("Initiating model.")
    model = OREModel(FLAGS, bertconfig)
    if torch.cuda.is_available():
        model.cuda()

    model.load_state_dict(torch.load(FLAGS.test_checkpoint))
    print('Loading from previous model.')
    print("Model initialized.")

    # load data
    print("Loading test data")
    tokenizer = BertTokenizer.from_pretrained(FLAGS.pretrained)
    test_set = OREDataset(FLAGS.test_path, tokenizer, FLAGS.max_length)
    testset_loader = torch.utils.data.DataLoader(test_set, 1, shuffle=False, drop_last=False)
    print("Start testing", time.strftime('%Y-%m-%d %H:%M:%S', time.localtime(time.time())))

    model.eval()
    for i, example in enumerate(testset_loader):
        model.zero_grad()
        tag_seq = model.decode(example)
        # tag_seq = tag_seq.cpu().detach().numpy().tolist()
        # en_metrics(e1, e2, r, tag_seq) if FLAGS.language == "en" else
        for j, tag in enumerate(tag_seq):
            org = test_set.getOrigin(i)
            text = org[0]
            try:
                rel = [text[k] if tag[k] != 0 else "" for k in range(len(text))]
            except:
                continue
            rel = "".join(rel)
            org.append(rel)
            with open("out/sl_error.txt", "a") as wf:
                wf.write(str(org)+"\n")


if __name__ == "__main__":
    test()
