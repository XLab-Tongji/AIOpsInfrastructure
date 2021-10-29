# -*- coding: UTF-8 -*-
import os
from extractfeature import template2Vec_preprocessor
from loganomaly.Depricated import log_anomaly_quantitive_train, log_anomaly_sequential_predict, \
    log_anomaly_quantitive_predict, log_anomaly_sequential_train, Predict
import numpy as np
import random
import torch


# parameters for early prepare √
logparser_structed_file = '../../Data/logparser_result/Drain/HDFS_split_40w.log_structured.csv'
logparser_event_file = '../../Data/logparser_result/Drain/HDFS_split_40w.log_templates.csv'
anomaly_label_file = '../../Data/log/hdfs/anomaly_label.csv'

# Generate train/test/valid data √
sequential_directory = './Data/DrainResult-HDFS/loganomaly/sequential_files/'
train_file_name = 'loganomaly_train_file'
test_file_name = 'loganomaly_test_file'
valid_file_name = 'loganomaly_valid_file'

wordvec_file_path = 'G:\\crawl-300d-2M.vec'
pattern_vec_out_path = '../../Data/DrainResult-HDFS/loganomaly_model_train/pattern_vec'
variable_symbol = '<*> '

def set_seed(seed):
    np.random.seed(seed)
    random.seed(seed)
    torch.manual_seed(seed) # cpu
    torch.cuda.manual_seed_all(seed)  # gpu
    torch.backends.cudnn.deterministic = True

# log anomaly sequential model parameters some parameter maybe changed to train similar models
hidden_size = 128
num_of_layers = 2
num_of_classes = 31
num_epochs = 3

sequence_length_sequential = 5
sequence_length_quantitive = num_of_classes
input_size_sequential = 300
input_size_quantitive =1
batch_size_sequential = 100
batch_size_quantitive =100


# for logAnomaly
train_root_path = '../../Data/DrainResult-HDFS/loganomaly_model_train/'
model_out_path_sequential = train_root_path + 'model_out_sequential/'
model_out_path_quantitive = train_root_path + 'model_out_quantitive/'
train_file = sequential_directory + train_file_name
pattern_vec_json = pattern_vec_out_path

# predict parameters
# log anomaly sequential model parameters
if not os.path.exists(sequential_directory):
    os.makedirs(sequential_directory)
if not os.path.exists(train_root_path):
    os.makedirs(train_root_path)


def extract_feature():
    template2Vec_preprocessor.generate_train_and_test_file(logparser_structed_file, logparser_event_file,
                                                           anomaly_label_file, sequential_directory, train_file_name,
                                                           valid_file_name, test_file_name, wordvec_file_path,
                                                           pattern_vec_out_path, variable_symbol)
    template2Vec_preprocessor.pattern_to_vec_tf_idf_from_log(logparser_event_file, wordvec_file_path,
                                                             pattern_vec_out_path, variable_symbol)
def train_sequential_model():
    log_anomaly_sequential_train.train_model(sequence_length_sequential, input_size_sequential, hidden_size, num_of_layers, num_of_classes,
                                             num_epochs, batch_size_sequential, train_root_path, model_out_path_sequential, train_file, pattern_vec_json)

def train_quantitive_model():
   log_anomaly_quantitive_train.train_model(sequence_length_quantitive, input_size_quantitive, hidden_size,
                                            num_of_layers, num_of_classes,
                                            num_epochs, batch_size_quantitive, train_root_path, model_out_path_quantitive,
                                            train_file, pattern_vec_json)

def test_sequential_model():
    log_anomaly_sequential_predict.do_predict(input_size_sequential, hidden_size, num_of_layers, num_of_classes, sequence_length_sequential,
                                              model_out_path_sequential + 'Adam_batch_size=' + str(batch_size_sequential) + ';epoch=' + str(
                                                  num_epochs) + '.pt', anomaly_label_file,
                                              sequential_directory + test_file_name, 10, pattern_vec_json)

def test_quantitive_model():
    log_anomaly_quantitive_predict.do_predict(input_size_quantitive, hidden_size, num_of_layers, num_of_classes, sequence_length_quantitive,
                                              model_out_path_quantitive + 'Adam_batch_size=' + str(batch_size_quantitive) + ';epoch=' + str(
                                                  num_epochs) + '.pt', anomaly_label_file,
                                              sequential_directory + test_file_name, 10, pattern_vec_json)

def predict():
    Predict.do_predict(input_size_sequential, input_size_quantitive, hidden_size, num_of_layers, num_of_classes, sequence_length_sequential,
                       model_out_path_sequential + 'Adam_batch_size=' + str(batch_size_sequential) + ';epoch=' + str(
                                                  num_epochs) + '.pt', model_out_path_quantitive + 'Adam_batch_size=' + str(batch_size_quantitive) + ';epoch=' + str(
                                                  num_epochs) + '.pt', anomaly_label_file,
                       sequential_directory + test_file_name, 10, pattern_vec_json)


set_seed(5) #19 13 9 10 0(93.538# )


#extract_feature()
#train_sequential_model()
#train_quantitive_model()
#test_sequential_model()
#test_quantitive_model()
predict()



