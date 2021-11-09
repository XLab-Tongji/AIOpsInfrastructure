# -*- coding: UTF-8 -*-
import os
import io
import re
import random
import math
import json
import pandas as pd
import numpy as np
import torch
import unicodedata

from extractfeature.eventid2number import add_numberid_new
# from extractfeature.get_synonym_and_antonym import get_synonym_and_antonym

block_id_regex = r'blk_(|-)[0-9]+'
special_patterns = {'dfs.FSNamesystem:': ['dfs', 'FS', 'Name', 'system'], 'dfs.FSDataset:': ['dfs', 'FS', 'dataset']}


def get_anomaly_block_id_set(anomaly_label_file):
    datafile = open(anomaly_label_file, 'r', encoding='UTF-8')
    data = pd.read_csv(datafile)

    data = data[data['Label'].isin(['Anomaly'])]
    # 16838 anomaly block right with the log anomaly paper
    anomaly_block_set = set(data['BlockId'])
    return anomaly_block_set


def get_log_template_dic(logparser_event_file):
    dic = {}
    datafile = open(logparser_event_file, 'r', encoding='UTF-8')
    data = pd.read_csv(datafile)
    for _, row in data.iterrows():
        # 更改为Index
        # dic[row['EventId']] = row['EventId']
        dic[row['EventId']] = row['numberID']
    return dic


# 更改1：训练集，pattern_vec的十六进制改成0-31的数字
# log parser_file should be structed.csv
def generate_train_and_test_file(logparser_structed_file, logparser_event_file, anomaly_label_file, out_dic,
                                 train_out_file_name, validation_out_file_name, test_out_file_name, wordvec_path,
                                 pattern_vec_out_path, variable_symbol):
    add_numberid_new(logparser_event_file)
    # anomaly_label.csv，即标签
    anomaly_block_set = get_anomaly_block_id_set(anomaly_label_file)
    # HDFS_split_40w.log_templates.csv，即事件，共31个模板
    log_template_dic = get_log_template_dic(logparser_event_file)
    # 转换为结构{'blk_id':[某一个模板，某一个模板，xxx]}
    session_dic = {}
    # HDFS_split_40w.log_structured.csv，即日志结构文件
    logparser_result = pd.read_csv(logparser_structed_file, header=0)
    # 正常的block
    normal_block_ids = set()
    # 异常的block
    abnormal_block_ids = set()
    for _, row in logparser_result.iterrows():
        key = row['EventTemplate']
        content = row['Content']
        block_id = re.search(block_id_regex, content).group()
        # 加入的是序号
        session_dic.setdefault(block_id, []).append(log_template_dic[row['EventId']])
        if block_id in anomaly_block_set:
            abnormal_block_ids.add(block_id)
        else:
            normal_block_ids.add(block_id)
    abnormal_block_ids = list(abnormal_block_ids)
    normal_block_ids = list(normal_block_ids)
    # 按照id进行排序
    abnormal_block_ids.sort()
    normal_block_ids.sort()
    # 随机打乱blocs_id
    random.shuffle(abnormal_block_ids)
    random.shuffle(normal_block_ids)
    # 生成test，train以及file文件
    with open(out_dic + train_out_file_name, 'w+') as train_file_obj, \
            open(out_dic + test_out_file_name, 'w+') as test_file_obj, \
            open(out_dic + validation_out_file_name, 'w+') as validation_file_obj:
        train_file_obj.write('BlockId,Sequence,label\n')
        test_file_obj.write('BlockId,Sequence,label\n')
        validation_file_obj.write('BlockId,Sequence,label\n')
        # 正常数据集8：1：1分配在训练集，测试集，验证集
        for i in range(len(normal_block_ids)):
            if i < len(normal_block_ids) * 0.8:
                # blockid
                train_file_obj.write(str(normal_block_ids[i]) + ', ')
                # 序列
                train_file_obj.write(' '.join([str(num_id) for num_id in session_dic[normal_block_ids[i]]]))
                # 表示是否异常
                train_file_obj.write(', 0\n')
            elif i < len(normal_block_ids) * 0.9:
                validation_file_obj.write(str(normal_block_ids[i]) + ', ')
                validation_file_obj.write(' '.join([str(num_id) for num_id in session_dic[normal_block_ids[i]]]))
                validation_file_obj.write(', 0\n')
            else:
                test_file_obj.write(str(normal_block_ids[i]) + ', ')
                test_file_obj.write(' '.join([str(num_id) for num_id in session_dic[normal_block_ids[i]]]))
                test_file_obj.write(', 0\n')

        # 异常数据均分分在测试集与验证集
        for i in range(len(abnormal_block_ids)):
            if i < 0:
                train_file_obj.write(str(abnormal_block_ids[i]) + ', ')
                train_file_obj.write(' '.join([str(num_id) for num_id in session_dic[abnormal_block_ids[i]]]))
                train_file_obj.write(', 1\n')
            elif i < len(abnormal_block_ids) / 2:
                validation_file_obj.write(str(abnormal_block_ids[i]) + ', ')
                validation_file_obj.write(' '.join([str(num_id) for num_id in session_dic[abnormal_block_ids[i]]]))
                validation_file_obj.write(', 1\n')
            else:
                test_file_obj.write(str(abnormal_block_ids[i]) + ', ')
                test_file_obj.write(' '.join([str(num_id) for num_id in session_dic[abnormal_block_ids[i]]]))
                test_file_obj.write(', 1\n')
    print('训练集，测试集，验证集生成完成')


def build_pretrained_embeddings(pretrained_file, embedding_dim, id2word, word2id):
    print('embedding matrix loading...')
    vocab_size = len(id2word)
    # 95*300的矩阵，词向量
    nn_embeddings = torch.nn.Embedding(vocab_size, embedding_dim)
    wv_file_path = pretrained_file
    count = 0
    #
    pretrain_id_list = []
    with open(wv_file_path, encoding="utf8") as f:
        for line in f:
            elems = line.rstrip().split(' ')
            token = unicodedata.normalize('NFD', elems[0])
            with torch.no_grad():
                if token in word2id:
                    count += 1
                    word_id = word2id[token]
                    nn_embeddings.weight[word_id] = torch.Tensor([float(v) for v in elems[1:]])
                    pretrain_id_list.append(word_id)
    embeddings = nn_embeddings.weight.data
    print('embedding matrix loaded.')
    print("#" * 40)
    print("total words in dataset: ", vocab_size)
    print("words in embedding matrix: ", count)
    print("Proportion: ", count / vocab_size * 100, "%")
    print("#" * 40)
    return embeddings, pretrain_id_list


def get_lower_case_name(text):
    word_list = []
    if text in special_patterns:
        return
    for index, char in enumerate(text):
        if not char.isupper():
            break
        else:
            if index == len(text) - 1:
                return [text]
    lst = []
    for index, char in enumerate(text):
        if char.isupper() and index != 0:
            word_list.append("".join(lst))
            lst = []
        lst.append(char)
    word_list.append("".join(lst))
    return word_list


def preprocess_pattern(log_pattern):
    special_list = []
    if log_pattern.split(' ')[0] in special_patterns.keys():
        special_list = special_patterns[log_pattern.split(' ')[0]]
        log_pattern = log_pattern[len(log_pattern.split(' ')[0]):]
    pattern = r'\*|,|\.|/|;|\'|`|\[|\]|<|>|\?|:|"|\{|\}|\~|!|@|#|\$|%|\^|&|\(|\)|-|=|\_|\+|，|。|、|；|‘|’|【|】|·|！| |…|（|）'
    result_list = [x for x in re.split(pattern, log_pattern) if len(x) > 0]
    final_list = list(map(get_lower_case_name, result_list))
    final_list.append(special_list)
    return [x for x in re.split(pattern, final_list.__str__()) if len(x) > 0]


def load_vectors(fname):
    fin = io.open(fname, 'r', encoding='utf-8', newline='\n', errors='ignore')
    data = {}
    x = 0
    for line in fin:
        x += 1
        print(str(x))
        tokens = line.rstrip().split(' ')
        data[tokens[0]] = list(map(float, tokens[1:]))
    return data


def pattern_to_vec_average(logparser_event_file, wordvec_path, pattern_vec_out_path, variable_symbol):
    data = load_vectors(wordvec_path)
    pattern_to_words = {}
    pattern_to_vectors = {}
    datafile = open(logparser_event_file, 'r', encoding='UTF-8')
    df = pd.read_csv(datafile)
    for _, row in df.iterrows():
        wd_list = preprocess_pattern(row['EventTemplate'].replace(variable_symbol, '').strip())
        pattern_to_words[row['EventTemplate'].replace(variable_symbol, '').strip()] = wd_list
    print(pattern_to_words)
    for key in pattern_to_words.keys():
        wd_list = pattern_to_words[key]
        pattern_vector = np.array([0.0 for _ in range(300)])
        word_used = 0
        for word in wd_list:
            if not word in data.keys():
                print('out of 0.1m words', ' ', word)
            else:
                word_used = word_used + 1
                pattern_vector = pattern_vector + np.array(data[word])
        pattern_to_vectors[key] = pattern_vector / word_used
    numberid2vec = {}
    for _, row in df.iterrows():
        numberid2vec[row['EventId']] = pattern_to_vectors[
            row['EventTemplate'].replace(variable_symbol, '').strip()].tolist()
    json_str = json.dumps(numberid2vec)
    with open(pattern_vec_out_path, 'w+') as file_obj:
        file_obj.write(json_str)
    return pattern_to_vectors


def pattern_to_vec_tf_idf_from_log(logparser_event_file, wordvec_path, pattern_vec_out_path, variable_symbol):
    """
    logparser_event_file = HDFS_split_40w.log_templates.csv
    wordvec_file_path = '../crawl-300d-2M.vec'
    pattern_vec_out_path = '../Data/DrainResult-HDFS/loganomaly_model_train/pattern_vec'
    variable_symbol = '<*> '
    """
    # 记录某模板（字符串）对应的单词列表
    pattern_to_words = {}
    pattern_to_vectors = {}
    # 记录某模板（字符串）出现的次数
    pattern_to_occurrences = {}
    datafile = open(logparser_event_file, 'r', encoding='UTF-8')
    df = pd.read_csv(datafile)
    # pattern_num = len(df)
    log_num = 11175629
    for _, row in df.iterrows():
        # 把模板分隔开，得出词语
        wd_list = preprocess_pattern(row['EventTemplate'].replace(variable_symbol, '').strip())
        # 转换为以下形式{'word1 word2 word3':[word1,word2,word3]}
        pattern_to_words[row['EventTemplate'].replace(variable_symbol, '').strip()] = wd_list
        pattern_to_occurrences[row['EventTemplate'].replace(variable_symbol, '').strip()] = row['Occurrences']
    print(pattern_to_words)
    # 模板中有的文字的集合
    words_set = set()
    for key in pattern_to_words.keys():
        words_set.update(pattern_to_words[key])
    words_list = list(words_set)
    words_list.sort()
    # 记录每个单词对应的id
    word2id = {}
    # 记录每个id对应的单词
    id2word = {}
    for i in range(len(words_list)):
        word2id[words_list[i]] = i
        id2word[i] = words_list[i]
    # 词向量，给定单词与词数据集的交集,95*300维度
    word_embedding, pretrain_id_list = build_pretrained_embeddings(wordvec_path, 300, id2word, word2id)
    IDF = {}
    for key in pattern_to_words.keys():
        wd_list = pattern_to_words[key]
        pattern_vector = np.array([0.0 for _ in range(300)])
        word_used = 0
        for word in wd_list:
            if word2id[word] in pretrain_id_list:
                word_used = word_used + 1
                weight = wd_list.count(word) / 1.0 / len(pattern_to_words[key])
                if word in IDF.keys():
                    pattern_vector = pattern_vector + weight * IDF[word] * np.array(word_embedding[word2id[word]])
                else:
                    pattern_occur_num = 0
                    for k in pattern_to_words.keys():
                        if word in pattern_to_words[k]:
                            pattern_occur_num = pattern_occur_num + pattern_to_occurrences[key]
                    IDF[word] = math.log10(log_num / 1.0 / pattern_occur_num)
                    # print('tf', weight, 'idf', IDF[word], word)
                    # print(data[word])
                    pattern_vector = pattern_vector + weight * IDF[word] * np.array(word_embedding[word2id[word]])
            else:
                pattern_vector = pattern_vector + np.array(word_embedding[word2id[word]])
                word_used = word_used + 1
        pattern_to_vectors[key] = pattern_vector / word_used
    numberid2vec = {}
    for _, row in df.iterrows():
        numberid2vec[row['EventId']] = pattern_to_vectors[
            row['EventTemplate'].replace(variable_symbol, '').strip()].tolist()
    json_str = json.dumps(numberid2vec)
    with open(pattern_vec_out_path, 'w+') as file_obj:
        file_obj.write(json_str)
    return pattern_to_vectors


def pattern_to_vec_template_from_log(logparser_event_file, wordvec_path, pattern_vec_out_path, variable_symbol):
    """
    logparser_event_file = HDFS_split_40w.log_templates.csv
    wordvec_file_path = '../crawl-300d-2M.vec'
    pattern_vec_out_path = '../Data/DrainResult-HDFS/loganomaly_model_train/pattern_vec'
    variable_symbol = '<*> '
    """
    # 记录某模板（字符串）对应的单词列表
    pattern_to_words = {}
    pattern_to_vectors = {}
    # 记录某模板（字符串）出现的次数
    pattern_to_occurrences = {}
    datafile = open(logparser_event_file, 'r', encoding='UTF-8')
    df = pd.read_csv(datafile)
    for _, row in df.iterrows():
        # 把模板分隔开，得出词语
        wd_list = preprocess_pattern(row['EventTemplate'].replace(variable_symbol, '').strip())
        # 转换为以下形式{'word1 word2 word3':[word1,word2,word3]}
        pattern_to_words[row['EventTemplate'].replace(variable_symbol, '').strip()] = wd_list
        pattern_to_occurrences[row['EventTemplate'].replace(variable_symbol, '').strip()] = row['Occurrences']
    print(pattern_to_words)
    # 模板中有的文字的集合
    words_set = set()
    for key in pattern_to_words.keys():
        words_set.update(pattern_to_words[key])
    words_list = list(words_set)
    words_list.sort()
    # 获取近义词与反义词集合
    # synonym, antonym, synonym_pair, antonym_pair = get_synonym_and_antonym(words_list)
    # print('近义词：' + str(synonym_pair))
    # print('反义词：' + str(antonym_pair))
    # 单词权重，目前找到的解决方案是所有词向量取平均，因此权重系数全部设置为1
    weight = np.array([1] * len(words_list))
    # 记录每个单词对应的id
    word2id = {}
    # 记录每个id对应的单词
    id2word = {}
    for i in range(len(words_list)):
        word2id[words_list[i]] = i
        id2word[i] = words_list[i]
    # 词向量，给定单词与词数据集的交集,95(单词数量)*300维度
    word_embedding, pretrain_id_list = build_pretrained_embeddings(wordvec_path, 300, id2word, word2id)
    print('word_embedding:' + str(word_embedding.shape))
    # 加权平均
    for key in pattern_to_words.keys():
        wd_list = pattern_to_words[key]
        pattern_vector = np.array([0.0 for _ in range(300)])
        word_used = 0
        for word in wd_list:
            pattern_vector = pattern_vector + weight[word2id[word]] * np.array(word_embedding[word2id[word]])
            word_used = word_used + 1
        pattern_to_vectors[key] = pattern_vector / word_used

    numberid2vec = {}
    for _, row in df.iterrows():
        numberid2vec[int(row['numberID'])] = pattern_to_vectors[
            row['EventTemplate'].replace(variable_symbol, '').strip()].tolist()
    json_str = json.dumps(numberid2vec)
    with open(pattern_vec_out_path, 'w+') as file_obj:
        file_obj.write(json_str)
    return pattern_to_vectors


def pattern_to_vec_tf_idf_advanced_from_log(logparser_event_file, wordvec_path, pattern_vec_out_path, variable_symbol):
    pattern_to_words = {}
    pattern_to_vectors = {}
    pattern_to_occurrences = {}
    datafile = open(logparser_event_file, 'r', encoding='UTF-8')
    df = pd.read_csv(datafile)
    # pattern_num = len(df)
    log_num = 11175629
    for _, row in df.iterrows():
        wd_list = preprocess_pattern(row['EventTemplate'].replace(variable_symbol, '').strip())
        pattern_to_words[row['EventTemplate'].replace(variable_symbol, '').strip()] = wd_list
        pattern_to_occurrences[row['EventTemplate'].replace(variable_symbol, '').strip()] = row['Occurrences']
    print(pattern_to_words)
    words_set = set()
    max_length_pattern = 0
    for key in pattern_to_words.keys():
        words_set.update(pattern_to_words[key])
        if len(pattern_to_words[key]) > max_length_pattern:
            max_length_pattern = len(pattern_to_words[key])
    words_list = list(words_set)
    words_list.sort()
    word2id = {}
    id2word = {}
    for i in range(len(words_list)):
        word2id[words_list[i]] = i
        id2word[i] = words_list[i]
    word_embedding, pretrain_id_list = build_pretrained_embeddings(wordvec_path, 300, id2word, word2id)
    IDF = {}
    for key in pattern_to_words.keys():
        wd_list = pattern_to_words[key]
        pattern_vector = np.array([0.0 for _ in range(300)])
        word_used = 0
        for word in wd_list:
            if word2id[word] in pretrain_id_list:
                word_used = word_used + 1
                TF = math.log10(wd_list.count(word) / 1.0 / len(pattern_to_words[key]) * max_length_pattern)
                if word in IDF.keys():
                    pattern_vector = pattern_vector + TF * IDF[word] * np.array(word_embedding[word2id[word]])
                else:
                    pattern_occur_num = 0
                    for k in pattern_to_words.keys():
                        if word in pattern_to_words[k]:
                            pattern_occur_num = pattern_occur_num + pattern_to_occurrences[key]
                    IDF[word] = math.log10(log_num / 1.0 / pattern_occur_num)
                    # print('tf', weight, 'idf', IDF[word], word)
                    # print(data[word])
                    pattern_vector = pattern_vector + TF * IDF[word] * np.array(word_embedding[word2id[word]])
            else:
                pattern_vector = pattern_vector + np.array(word_embedding[word2id[word]])
                word_used = word_used + 1
        pattern_to_vectors[key] = pattern_vector / word_used
    numberid2vec = {}
    for _, row in df.iterrows():
        numberid2vec[row['EventId']] = pattern_to_vectors[
            row['EventTemplate'].replace(variable_symbol, '').strip()].tolist()
    json_str = json.dumps(numberid2vec)
    with open(pattern_vec_out_path, 'w+') as file_obj:
        file_obj.write(json_str)
    return pattern_to_vectors
