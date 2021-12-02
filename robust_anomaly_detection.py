# -*- coding: UTF-8 -*-

import os
from extractfeature import hdfs_robust_preprocessor
from anomalydetection.logrobust import bi_lstm_att_train
from anomalydetection.logrobust import bi_lstm_att_predict
import numpy as np
import random
import torch


# parameters for early prepare
logparser_structed_file = './Data/logparser_result/Drain/HDFS.log_structured.csv'
logparser_event_file = './Data/logparser_result/Drain/HDFS.log_templates.csv'
anomaly_label_file = './Data/log/hdfs/anomaly_label.csv'

# train and test file
sequential_directory = './Data/DrainResult-HDFS/robust_att_bi_model_train/sequential_files/'
train_file_name = 'robust_train_file'
test_file_name = 'robust_test_file'
valid_file_name = 'robust_valid_file'


wordvec_file_path = './Data/pretrainedwordvec/crawl-300d-2M.vec(0.1M)'
pattern_vec_out_path = './Data/DrainResult-HDFS/robust_att_bi_model_train/pattern_vec'
variable_symbol = '<*> '


def set_seed(seed):
    np.random.seed(seed)
    random.seed(seed)
    torch.manual_seed(seed) # cpu
    torch.cuda.manual_seed_all(seed)  # gpu
    torch.backends.cudnn.deterministic = True

# log anomaly sequential model parameters
sequence_length = 50
input_size = 300
hidden_size = 128
num_of_layers = 3
# 1 using sigmoid, 2 using softmax
num_of_classes = 1
num_epochs = 100
batch_size = 512
# for robust attention bi
train_root_path = './Data/DrainResult-HDFS/robust_att_bi_model_train/'
model_out_path = train_root_path + 'model_out/'
train_file = sequential_directory + train_file_name
pattern_vec_json = pattern_vec_out_path


# predict parameters
# log anomaly sequential model parameters
if not os.path.exists(sequential_directory):
    os.makedirs(sequential_directory)
if not os.path.exists(train_root_path):
    os.makedirs(train_root_path)


def extract_feature():
    hdfs_robust_preprocessor.generate_train_and_test_file(logparser_structed_file, logparser_event_file, anomaly_label_file, sequential_directory, train_file_name, valid_file_name, test_file_name, wordvec_file_path, pattern_vec_out_path, variable_symbol)


def train_model():
    bi_lstm_att_train.train_model(sequence_length, input_size, hidden_size, num_of_layers, num_of_classes, num_epochs, batch_size, train_root_path, model_out_path, train_file, pattern_vec_json)

def test_model():
    # do something
    bi_lstm_att_predict.do_predict(input_size, hidden_size, num_of_layers, num_of_classes, sequence_length, model_out_path + 'Adam_batch_size=' + str(batch_size) + ';epoch=' + str(num_epochs) + ';sequence=' + str(sequence_length) + '.pt', sequential_directory + test_file_name, batch_size, pattern_vec_json)

set_seed(5)
#eventid2number.add_numberid(logparser_event_file)
#pattern_extract()
#extract_feature()
train_model()
test_model()


