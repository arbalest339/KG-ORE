'''
Author: your name
Date: 2020-09-23 09:23:31
LastEditTime: 2021-04-23 17:37:49
LastEditors: Please set LastEditors
Description: code and model configs
FilePath: /entity_disambiguation/config.py
'''
import os
import json
import time


class Flags(object):
    def __init__(self):
        # task info
        self.data_set = "chinatimes"  # "corekb" "chinatimes"
        self.is_continue = False
        self.is_test = False
        self.Canonicalizing = False

        # data dirs
        curpath = os.path.abspath(os.path.dirname(__file__))
        self.pretrained = "bert-base-chinese"
        self.checkpoint_dir = os.path.join(
            curpath, "checkpoints")  # Path of model checkpoints
        self.checkpoint_path = os.path.join(
            self.checkpoint_dir, f"{time.strftime('%m-%d-%H', time.localtime(time.time()))}.pkl")
        self.pretrain_checkpoint = os.path.join(
            self.checkpoint_dir, "03-19-10.pkl")
        self.test_checkpoint = os.path.join(
            self.checkpoint_dir, "03-24-16.pkl")

        self.data_dir = os.path.join(
            curpath, "zh_data")  # Path of input data dir
        self.train_path = os.path.join(
            self.data_dir, "train.txt")     # !!!!!!!!!!!!!!!!!!!!!!
        self.dev_path = os.path.join(
            self.data_dir, "dev.txt")
        self.test_path = os.path.join(
            self.data_dir, "test.txt")
        # self.train_mat = os.path.join(self.data_dir, f"{self.data_set}_train_matrixs.npy")
        # self.dev_mat = os.path.join(self.data_dir, f"{self.data_set}_dev_matrixs.npy")
        # self.test_mat = os.path.join(self.data_dir, f"{self.data_set}_test_matrixs.npy")

        # Path of output results dir
        self.out_dir = os.path.join(curpath, "out")
        self.record_path = os.path.join(self.out_dir, "ner_record.txt")

        # train hyper parameters
        self.learning_rate = 3.e-5
        self.epoch = 100
        self.batch_size = 30
        self.test_batch_size = 1
        self.max_length = 128
        self.dropout_rate = 0.5
        self.weight_decay = 1.e-3
        self.patient = 3
        self.use_cuda = True

        # TransD config
        self.trans_select = "TransD"  # TransE TransH
        self.feature_dim = 100
        self.margin = 4.0

        # model choice
        self.features = []   # ["dp", "pos", "ner"]
        self.decoder = "crf"  # crf, softmax

        # lstm
        self.lstm_hidden = 300
        self.n_layers = 2

        # global datas
        self.label_map = {"O": 0, "B-PER": 1, "I-PER": 2, "B-ORG": 3, "I-ORG": 4,
                          "B-LOC": 5, "I-LOC": 6, "B-REG": 7, "I-REG": 8, "B-OTH": 9, "I-OTH": 10}
        # self.dp_map = json.load(
        #     open(os.path.join(self.data_dir, "dp_map.json")))
        self.pos_map = json.load(
            open(os.path.join(self.data_dir, "pos_map.json")))


FLAGS = Flags()
