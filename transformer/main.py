# -*- coding: UTF-8 -*-
'''
 # @Project     : self-attentive
 # @File        : main.py.py
 # #@Author     : mount_potato
 # @Date        : 2021/11/6 9:20 下午
 # @Description :
'''

import numpy as np
import os
from transformer.bgl_preprocessor import *
from transformer.Learning.self_atten_train import train_model, test_model

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

origin_log_path = "data/BGL_2k.log"
wordvec_path = "crawl-300d-2M.vec"
pattern_vec_out_path = "output/pattern_vec"
index_vec_out_path = "output/index_vec"
out_dic_path = "./"
train_out_file_name = "train_seq.log"
validation_out_file_name = "validate_seq.log"
test_out_file_name = "test_seq.log"
train_file_maxsize = 50
validation_file_maxsize = 50

N_HEADES = 4
INPUT_DIM = 300  # 就是INPUT_DIM
HID_DIM = 512
OUTPUT_DIM = 300

N_ENCODERS = 3
FEEDFORWARD_DIM = 2048
DROPOUT_RATE = 0.5
LEARNING_RATE = 0.005

N_EPOCHS = 20
CLIP = 1
MODEL_OUTPUT_PATH = "output/model.pt"
BATCH_SIZE = 64
PAD_IDX = 1
EMM_IDX = 0

train_iterator = []
validation_iterator = []
test_iterator = []
epsilon = 0
index_to_tensor = []
pattern_to_indexList = {}
nn_embedding = []

if not os.path.exists("./data"):
    os.makedirs("./data")
if not os.path.exists("./output"):
    os.makedirs("./output")


def extract_feature():
    """
    生成train_seq.log,test_seq.log,validate_seq.log三个集合，分别代表用于
    训练、测试、验证的日志序列的集合(格式包含Sequence id, Label)
    :return: None
    """
    generate_train_and_test_file(origin_log_path, out_dic_path,
                                 train_out_file_name, validation_out_file_name,
                                 test_out_file_name, train_file_maxsize,
                                 validation_file_maxsize)


def pattern_to_vec():
    """
    生成index_to_tensor和pattern_to_index两个列表，前者是"索引-词向量"列表
    （注：未纳入[EMBEDDING]和PAD，即词向量从0开始索引)
    后者是"日志序列id-词语表索引"列表，
    :return: index_to_tensor和pattern_to_index两个列表
    """
    global index_to_tensor, pattern_to_indexList
    index_to_tensor, pattern_to_indexList, _ = pattern_to_vec_bgl(
        origin_log_path, wordvec_path, pattern_vec_out_path,
        index_vec_out_path, PAD_IDX, EMM_IDX)


def get_iterator():
    # 获取index_to_tensor和pattern_to_indexList
    # 获取iterator
    with open(out_dic_path + train_out_file_name, mode="r") as train_file_obj, \
            open(out_dic_path + validation_out_file_name, mode="r") as validation_file_obj, \
            open(out_dic_path + test_out_file_name, mode="r") as test_file_obj:

        train_file_lines = train_file_obj.readlines()
        validation_file_lines = validation_file_obj.readlines()
        test_file_lines = test_file_obj.readlines()

        cnt = 0
        batch_info = [[], []]
        batch = torch.tensor([])
        for line in train_file_lines:
            info = [item.strip() for item in line.split(',')]

            batch_info[0].append(pattern_to_indexList[info[0]])
            batch_info[1].append(int(info[1]))
            cnt += 1
            if cnt >= BATCH_SIZE:
                train_iterator.append(batch_info)
                cnt = 0
                batch_info = [[], []]
                continue

        cnt = 0
        batch_info = [[], []]
        for line in validation_file_lines:
            info = [item.strip() for item in line.split(',')]
            batch_info[0].append(pattern_to_indexList[info[0]])
            batch_info[1].append(int(info[1]))
            cnt += 1
            if cnt >= BATCH_SIZE:
                validation_iterator.append(batch_info)
                cnt = 0
                batch_info = [[], []]

        cnt = 0
        batch_info = [[], []]
        for line in test_file_lines:
            info = [item.strip() for item in line.split(',')]
            batch_info[0].append(pattern_to_indexList[info[0]])
            batch_info[1].append(int(info[1]))
            cnt += 1
            if cnt >= BATCH_SIZE:
                test_iterator.append(batch_info)
                cnt = 0
                batch_info = [[], []]

        # print(train_iterator)


def train():
    global epsilon
    epsilon = train_model(N_HEADES=N_HEADES,
                          INPUT_DIM=INPUT_DIM,
                          HID_DIM=HID_DIM,
                          OUTPUT_DIM=OUTPUT_DIM,
                          N_ENCODERS=N_ENCODERS,
                          FEEDFORWARD_DIM=FEEDFORWARD_DIM,
                          DROPOUT_RATE=DROPOUT_RATE,
                          LEARNING_RATE=LEARNING_RATE,
                          N_EPOCHS=N_EPOCHS,
                          CLIP=CLIP,
                          TRAIN_ITERATOR=train_iterator,
                          VALID_ITERATOR=validation_iterator,
                          MODEL_OUTPUT_PATH=MODEL_OUTPUT_PATH,
                          PAD_IDX=PAD_IDX,
                          INDEX_TO_TENSOR=index_to_tensor,
                          NN_EMBEDDING=nn_embedding,
                          INDEX_VEC_PATH=index_vec_out_path,
                          DEVICE=device)


def test():
    test_model(N_HEADES=N_HEADES,
               INPUT_DIM=INPUT_DIM,
               HID_DIM=HID_DIM,
               OUTPUT_DIM=OUTPUT_DIM,
               N_ENCODERS=N_ENCODERS,
               FEEDFORWARD_DIM=FEEDFORWARD_DIM,
               DROPOUT_RATE=DROPOUT_RATE,
               MODEL_OUTPUT_PATH=MODEL_OUTPUT_PATH,
               TEST_ITERATOR=test_iterator,
               EPSILON=epsilon,
               PAD_IDX=PAD_IDX,
               INDEX_TO_TENSOR=index_to_tensor,
               NN_EMBEDDING=nn_embedding,
               INDEX_VEC_PATH=index_vec_out_path,
               DEVICE=device)


extract_feature()
pattern_to_vec()

# danny
get_iterator()

# train model
train()
# # test model
test()
