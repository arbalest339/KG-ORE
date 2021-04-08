"""
training process entry
"""

import time
import torch
import numpy as np


from transformers import BertTokenizer, BertConfig
from torch.optim.lr_scheduler import ReduceLROnPlateau, CosineAnnealingLR
from tqdm import tqdm
from models.ner_model import NERModel
from test import zh_metrics
from data_reader import NERDataset
from config import FLAGS


def select_model(flags, bertconfig):
    model = NERModel(flags, bertconfig)
    return model


def select_optim(model):
    # optimizer = torch.optim.Adadelta([{"params": model.aggcn.parameters(), "lr": aggcnargs.lr},
    #                                   {"params": model.bert.parameters()},
    #                                   {"params": model.transd.parameters()},
    #                                   {"params": model.crf_layer.parameters()}], lr=FLAGS.learning_rate)
    optimizer = torch.optim.Adam(model.parameters(), lr=FLAGS.learning_rate, weight_decay=FLAGS.weight_decay)
    return optimizer


def test(model, validset_loader):
    positive_true = 0
    positive_false = 0
    negative_false = 0
    model.eval()
    for token, pos, golds, mask in validset_loader:
        model.zero_grad()
        tag_seq = model.decode(token, pos, mask)
        golds = golds.cpu().numpy().tolist()
        # tag_seq = tag_seq.cpu().detach().numpy().tolist()
        # en_metrics(e1, e2, r, tag_seq) if FLAGS.language == "en" else
        for gold, seq in zip(golds, tag_seq):
            pt, pf, nf = zh_metrics(gold, seq)
            positive_true += pt
            positive_false += pf
            negative_false += nf

    precision = positive_true / (positive_false + positive_true)
    recall = positive_true / (positive_true + negative_false)
    f1 = 2 * precision * recall / (precision + recall)

    print(f"Precision: {precision: .4f}, Recall: {recall: .4f}, F1: {f1: .4f}")
    with open(FLAGS.record_path, "a") as rf:
        rf.write(f"Precision: {precision: .4f}, Recall: {recall: .4f}, F1: {f1: .4f}\n")
    return f1


def main():
    # load from pretrained config file
    # bertconfig = json.load(open(FLAGS.pretrained_config))
    bertconfig = BertConfig.from_pretrained(FLAGS.pretrained)

    # Initiate model
    print("Initiating model.")
    model = select_model(FLAGS, bertconfig)
    if torch.cuda.is_available():
        model.cuda()
    if FLAGS.is_continue:
        model.load_state_dict(torch.load(FLAGS.pretrain_checkpoint))
        print('Training from previous model.')
    print("Model initialized.")

    # load data
    print("Loading traning and valid data")
    tokenizer = BertTokenizer.from_pretrained(FLAGS.pretrained)
    train_set = NERDataset(FLAGS.train_path, tokenizer, FLAGS.max_length, mode="train")
    dev_set = NERDataset(FLAGS.dev_path, tokenizer, FLAGS.max_length, mode="test")
    trainset_loader = torch.utils.data.DataLoader(train_set, FLAGS.batch_size, num_workers=0, drop_last=True, shuffle=True)
    validset_loader = torch.utils.data.DataLoader(dev_set, FLAGS.test_batch_size, num_workers=0, drop_last=True, shuffle=True)

    optimizer = select_optim(model)
    scheduler = ReduceLROnPlateau(optimizer, 'min', verbose=1, patience=2, factor=0.5, min_lr=1.e-8)
    # scheduler = CosineAnnealingLR(optimizer, T_max=(FLAGS.epoch // 9) + 1)
    # best_loss = 100
    best_acc = 0.0
    # patience = FLAGS.patient

    print("Start training", time.strftime('%Y-%m-%d %H:%M:%S', time.localtime(time.time())))
    for epoch in range(FLAGS.epoch):
        model.train()
        losses = []
        accs = []
        with tqdm(total=len(train_set)//FLAGS.batch_size, desc=f'Epoch {epoch+1}/{FLAGS.epoch}', unit='it') as pbar:
            for step, data in enumerate(trainset_loader):
                token, pos, gold, mask, acc_mask = data
                model.zero_grad()
                loss, acc, zero_acc = model(token, pos, gold, mask, acc_mask)
                losses.append(loss.data.item())
                accs.append(acc.data.item())
                # backward
                loss.backward()
                optimizer.step()
                # tqdm
                pbar.set_postfix({'batch_loss': loss.data.item(), "acc": acc.data.item(), "zero": zero_acc.data.item()})   # 在进度条后显示当前batch的损失
                pbar.update(1)
        train_loss = np.mean(losses)
        train_acc = np.mean(accs)
        print(f"[{epoch + 1}/{FLAGS.epoch}] trainset mean_loss: {train_loss: 0.4f} trainset mean_acc: {train_acc: 0.4f}")

        f1 = test(model, validset_loader)
        if f1 > best_acc:
            best_acc = f1
            print('Saving model...  ', time.strftime('%Y-%m-%d %H:%M:%S', time.localtime(time.time())))
            torch.save(model.state_dict(), FLAGS.checkpoint_path)
            print('Saving model finished.  ', time.strftime('%Y-%m-%d %H:%M:%S', time.localtime(time.time())))
        #     patience = FLAGS.patient
        # else:
        #     patience -= 1

        # if patience == 0:
        #     break
        # print('Saving model...  ', time.strftime('%Y-%m-%d %H:%M:%S', time.localtime(time.time())))
        # torch.save(model.state_dict(), FLAGS.checkpoint_path)
        # print('Saving model finished.  ', time.strftime('%Y-%m-%d %H:%M:%S', time.localtime(time.time())))
    print('Training finished.  ', time.strftime('%Y-%m-%d %H:%M:%S', time.localtime(time.time())))


if __name__ == "__main__":
    main()
