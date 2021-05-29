'''
Author: your name
Date: 2020-09-23 09:23:31
LastEditTime: 2021-05-10 08:12:27
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
        # self.data_set = "chinatimes"  # "corekb" "chinatimes"
        self.is_continue = False
        self.is_test = False
        self.Canonicalizing = False

        # data dirs
        curpath = os.path.abspath(os.path.dirname(__file__))
        self.pretrained = "bert-base-chinese"
        self.checkpoint_dir = os.path.join(
            curpath, "checkpoints")  # Path of model checkpoints
        # self.checkpoint_path = os.path.join(
        #     self.checkpoint_dir, f"{time.strftime('%m-%d-%H', time.localtime(time.time()))}.pkl")
        self.checkpoint_path = "checkpoints/no_best.pkl"
        self.pretrain_checkpoint = os.path.join(
            self.checkpoint_dir, "03-19-10.pkl")
        # self.test_checkpoint = self.checkpoint_path
        self.test_checkpoint = "checkpoints/no_best.pkl"

        self.data_dir = os.path.join(curpath, "coerkb")  # Path of input data dir
        self.train_path = os.path.join(self.data_dir, "train.txt")     # !!!!!!!!!!!!!!!!!!!!!!
        self.dev_path = os.path.join(self.data_dir, "dev.txt")
        self.test_path = os.path.join(self.data_dir, "case_study")    # testWithRel

        # Path of output results dir
        self.out_dir = os.path.join(curpath, "out")
        self.record_path = os.path.join(self.out_dir, "ner_record.txt")
        self.error_path = os.path.join(self.out_dir, "rel_error.txt")

        # train hyper parameters
        self.learning_rate = 3.e-5
        self.epoch = 100
        self.batch_size = 50
        self.test_batch_size = 8
        self.max_length = 128
        self.dropout_rate = 0.7
        self.weight_decay = 1.e-3
        self.patient = 3
        self.use_cuda = True

        # TransD config
        self.trans_select = "TransD"  # TransE TransH
        self.feature_dim = 100
        self.margin = 4.0

        # model choice
        self.use_knowledge = False
        self.knowledges = []   # ["desc", "exrest", "kbRel"]

        # QA full connection
        self.qa_hidden = 300


FLAGS = Flags()
