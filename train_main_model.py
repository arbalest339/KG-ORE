"""
training process entry
"""
import os
import time
import torch
import numpy as np


from transformers import BertTokenizer, BertConfig
from torch.optim.lr_scheduler import ReduceLROnPlateau, CosineAnnealingLR
from tqdm import tqdm
from models.qaore_model import OREModel
from test import zh_metrics
from data_reader import OREDataset
from config import FLAGS

os.environ["CUDA_VISIBLE_DEVICES"] = '2'


def select_model(flags, bertconfig):
    model = OREModel(flags, bertconfig)
    return model


def select_optim(model):
    optimizer = torch.optim.Adam(model.parameters(), lr=FLAGS.learning_rate)
    return optimizer


def eval(model, validset_loader):
    positive_true = 0
    positive_false = 0
    negative_false = 0
    model.eval()
    for data in validset_loader:
        model.zero_grad()
        start_id, end_id = data["start_id"], data["end_id"]
        _, slogits, elogits = model(data)
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
    train_set = OREDataset(FLAGS.train_path, tokenizer, FLAGS.max_length)
    dev_set = OREDataset(FLAGS.dev_path, tokenizer, FLAGS.max_length)
    trainset_loader = torch.utils.data.DataLoader(train_set, FLAGS.batch_size, num_workers=0, drop_last=False, shuffle=True)
    validset_loader = torch.utils.data.DataLoader(dev_set, FLAGS.test_batch_size, num_workers=0, drop_last=False, shuffle=True)

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
        with tqdm(total=len(train_set)//FLAGS.batch_size, desc=f'Epoch {epoch+1}/{FLAGS.epoch}', unit='it') as pbar:
            for step, data in enumerate(trainset_loader):
                # input_ids, mask, type_ids, start_id, end_id = data
                model.zero_grad()
                loss, slogits, elogits = model(data)
                losses.append(loss.data.item())
                # backward
                loss.backward()
                optimizer.step()
                # tqdm
                pbar.set_postfix({'batch_loss': loss.data.item()})   # 在进度条后显示当前batch的损失
                pbar.update(1)
        train_loss = np.mean(losses)
        print(f"[{epoch + 1}/{FLAGS.epoch}] trainset mean_loss: {train_loss: 0.4f}")

        f1 = eval(model, validset_loader)
        if f1 > best_acc:
            best_acc = f1
            print('Saving model...  ', time.strftime('%Y-%m-%d %H:%M:%S', time.localtime(time.time())))
            torch.save(model.state_dict(), FLAGS.checkpoint_path)
            print('Saving model finished.  ', time.strftime('%Y-%m-%d %H:%M:%S', time.localtime(time.time())))
            patience = FLAGS.patient
        else:
            patience -= 1

        if patience == 0:
            break
        # print('Saving model...  ', time.strftime('%Y-%m-%d %H:%M:%S', time.localtime(time.time())))
        # torch.save(model.state_dict(), FLAGS.checkpoint_path)
        # print('Saving model finished.  ', time.strftime('%Y-%m-%d %H:%M:%S', time.localtime(time.time())))
    print('Training finished.  ', time.strftime('%Y-%m-%d %H:%M:%S', time.localtime(time.time())))


if __name__ == "__main__":
    main()
