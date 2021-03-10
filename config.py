'''
Author: your name
Date: 2020-09-23 09:23:31
LastEditTime: 2021-03-10 10:31:31
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
        self.data_set = "mini_corekb"  # "corekb" "chinatimes"
        self.is_continue = False
        self.is_test = False
        self.Canonicalizing = False

        # data dirs
        curpath = os.path.abspath(os.path.dirname(__file__))
        self.pretrained = "bert-base-chinese"
        self.checkpoint_dir = os.path.join(
            curpath, "checkpoints")  # Path of model checkpoints
        self.checkpoint_path = os.path.join(
            self.checkpoint_dir, f"{time.time()}.pkl")

        self.data_dir = os.path.join(
            curpath, f"mini_ore_data")  # Path of input data dir
        self.train_path = os.path.join(
            self.data_dir, f"{self.data_set}_train.txt")     # !!!!!!!!!!!!!!!!!!!!!!
        self.dev_path = os.path.join(
            self.data_dir, f"{self.data_set}_dev.txt")
        self.test_path = os.path.join(
            self.data_dir, f"{self.data_set}_test.txt")
        # self.train_mat = os.path.join(self.data_dir, f"{self.data_set}_train_matrixs.npy")
        # self.dev_mat = os.path.join(self.data_dir, f"{self.data_set}_dev_matrixs.npy")
        # self.test_mat = os.path.join(self.data_dir, f"{self.data_set}_test_matrixs.npy")

        # Path of output results dir
        self.out_dir = os.path.join(curpath, "out")

        # train hyper parameters
        self.learning_rate = 3.e-5
        self.epoch = 30
        self.batch_size = 20
        self.test_batch_size = 8
        self.max_length = 128
        self.dropout_rate = 0.5
        self.weight_decay = 1.e-3
        self.patient = 3
        self.use_cuda = True

        # TransD config
        self.feature_dim = 100
        self.margin = 4.0

        # lstm
        self.lstm_hidden = 300
        self.n_layers = 2

        # global datas
        self.ner_map = {"O": 0, "B-PER": 1, "I-PER": 2, "B-ORG": 3, "I-ORG": 4,
                        "B-LOC": 5, "I-LOC": 6, "B-REG": 7, "I-REG": 8, "B-OTH": 9, "I-OTH": 10}
        self.label_map = {"O": 0, "B-REL": 1, "I-REL": 2}
        self.dp_map = json.load(
            open(os.path.join(self.data_dir, "dp_map.json")))
        self.pos_map = json.load(
            open(os.path.join(self.data_dir, "pos_map.json")))


FLAGS = Flags()
