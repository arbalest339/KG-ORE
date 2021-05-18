"""
test process entry
"""

import os
import time
import json
import torch

from transformers import BertTokenizer, BertConfig
from models.qaore_model import OREModel
from data_reader import OREDataset
from config import FLAGS

os.environ["CUDA_VISIBLE_DEVICES"] = '2'


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
        elif t != 0 and g != t:
            positive_false += 1
        elif g != 0:
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
    testset_loader = torch.utils.data.DataLoader(test_set, FLAGS.test_batch_size, num_workers=0, drop_last=False)
    wf = open("out/super.txt", "a")
    wf.write("Start testing " + str(time.strftime('%Y-%m-%d %H:%M:%S', time.localtime(time.time()))) + "\n")
    print("Start testing", time.strftime('%Y-%m-%d %H:%M:%S', time.localtime(time.time())))

    positive_true = 0
    positive_false = 0
    negative_false = 0
    errorsExps = []
    model.eval()
    for data in testset_loader:
        model.zero_grad()
        _, slogits, elogits = model(data)
        start_id, end_id = data["start_id"], data["end_id"]
        start_id = start_id.squeeze(dim=-1).cpu().numpy().tolist()
        end_id = end_id.squeeze(dim=-1).cpu().numpy().tolist()
        pred_s = slogits.cpu().detach().numpy().tolist()
        pred_e = elogits.cpu().detach().numpy().tolist()
        for gs, ge, ps, pe in zip(start_id, end_id, pred_s, pred_e):
            pf = max(ps-gs, 0) + max(ge-pe, 0)
            nf = max(gs-ps, 0) + max(pe-ge, 0)
            pt = max(pe-ps, pe-gs, ge-gs, ge-ps, 0) - pf - nf
            # tag_seq = tag_seq.cpu().detach().numpy().tolist()
            positive_true += pt
            positive_false += pf
            negative_false += nf

    precision = positive_true / max(positive_false + positive_true, 1)
    recall = positive_true / max(positive_true + negative_false, 1)
    f1 = 2 * precision * recall / max(precision + recall, 1e-9)

    print(f"Precision: {precision: .4f}, Recall: {recall: .4f}, F1: {f1: .4f}")
    wf.write(f"Precision: {precision: .4f}, Recall: {recall: .4f}, F1: {f1: .4f}\n")
    print('Testing finished.  ', time.strftime('%Y-%m-%d %H:%M:%S', time.localtime(time.time())))
    wf.write("Testing finished " + str(time.strftime('%Y-%m-%d %H:%M:%S', time.localtime(time.time()))) + "\n")

    with open(FLAGS.train_path, "a") as tf:
        for exp in errorsExps:
            tf.write(json.dumps(exp, ensure_ascii=False))
    # for exp in pfExps:
    #     exp = [t+"/"+g+"/"+str(p) for t, g, p in zip(exp[0], exp[1], exp[2])]
    #     wf.write(str(exp)+"\n"+"\n")
    # wf.write("Start of nf")
    # for exp in nfExps:
    #     exp = [t+"/"+g+"/"+str(p) for t, g, p in zip(exp[0], exp[1], exp[2])]
    #     wf.write(str(exp)+"\n"+"\n")
    # wf.close()
    return f1


if __name__ == "__main__":
    test()
