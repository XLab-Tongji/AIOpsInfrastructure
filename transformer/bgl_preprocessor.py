# -*- coding: UTF-8 -*-
"""
 # @Project     : self-attentive
 # @File        : bgl_preprocessor.py
 # #@Author     : mount_potato
 # @Date        : 2021/11/2 8:01 下午
 # @Description :
"""
import json

import random
import torch
import re
import unicodedata


def generate_train_and_test_file(
        origin_log_path,
        out_dic,
        train_out_file_name,
        validation_out_file_name,
        test_out_file_name,
        train_file_maxsize=50,
        validation_file_maxsize=50,
):
    """
    生成train,validation和test三个集合，
    生成的file包括两列:包含的日志消息id，该日志的正常与否(0正常，1不正常)
    Args:
        origin_log_path: bgl_2k.log的位置
        out_dic: 生成三个log文件的地址
        train_out_file_name: 生成train file的名字
        validation_out_file_name: 生成validation file的名字
        test_out_file_name: 生成test file的名字
        train_file_maxsize: 规定train file包含最多的日志消息数
        validation_file_maxsize: 规定test file包含最多的日志消息数（剩下的都是test file的）

    """

    assert train_file_maxsize > 0
    assert validation_file_maxsize > 0

    file = open(origin_log_path, mode='r', encoding="utf-8")

    lines = file.read().splitlines()
    file.close()

    # 正常的日志消息1857条
    # 错误的日志消息143条
    sequence_normal = []
    sequence_abnormal = []

    for line in lines:
        if not line:
            continue
        columns = [col.strip() for col in line.split(" ") if col]

        # 带'-'的为正常 否则为异常
        label = 1 if columns[0] != '-' else 0

        # 将对应日志的id放入两个列表中
        if label == 0:
            sequence_normal.append(columns[1])
        else:
            sequence_abnormal.append(columns[1])

    seq_normal_id_list = sequence_normal
    seq_abnormal_id_list = sequence_abnormal

    # 随机打乱
    random.shuffle(sequence_normal)
    random.shuffle(seq_abnormal_id_list)

    with open(out_dic + train_out_file_name, mode="w+") as train_file_obj, \
            open(out_dic + validation_out_file_name, mode="w+") as validation_file_obj, \
            open(out_dic + test_out_file_name, mode="w+") as test_file_obj:

        for i in range(len(sequence_normal)):
            if i < train_file_maxsize:
                train_file_obj.write(
                    ''.join([str(num_id)
                             for num_id in seq_normal_id_list[i]]) + ', 0\n')
            elif i < train_file_maxsize + validation_file_maxsize:
                validation_file_obj.write(
                    ''.join([str(num_id)
                             for num_id in seq_normal_id_list[i]]) + ', 0\n')
            else:
                test_file_obj.write(
                    ''.join([str(num_id)
                             for num_id in seq_normal_id_list[i]]) + ', 0\n')

        for i in range(len(sequence_abnormal)):
            if i < train_file_maxsize:
                train_file_obj.write(''.join(
                    [str(num_id)
                     for num_id in seq_abnormal_id_list[i]]) + ', 1\n')
            elif i < train_file_maxsize + validation_file_maxsize:
                validation_file_obj.write(''.join(
                    [str(num_id)
                     for num_id in seq_abnormal_id_list[i]]) + ', 1\n')
            else:
                test_file_obj.write(''.join(
                    [str(num_id)
                     for num_id in seq_abnormal_id_list[i]]) + ', 1\n')

        train_file_obj.close()
        validation_file_obj.close()
        test_file_obj.close()


def build_pretrained_embeddings(pretrained_file, id2word, word2id, PAD_IDX):
    """
    修改的建立embedding
    Args:
        pretrained_file: crawl-300d.vec文件
        id2word: 字典：key-词语表中词语索引，value，词语表中词语
        word2id: 字典：key-词词语表中词语，value，词语表中词语索引

    Returns:
        embeddings 词向量embedding，300维向量
        pretrain_id_list 暂时没有使用到
    """
    print('embedding matrix loading...')
    vocab_size = len(id2word)
    nn_embeddings = torch.nn.Embedding(
        vocab_size + 2, embedding_dim=300, padding_idx=PAD_IDX)
    wv_file_path = pretrained_file
    count = 0
    pretrain_id_list = []

    # 似乎需要关掉torch的梯度，否则nn_embeddings.weight[word_id]赋值时报错(需要梯度的值进行原地计算)
    with torch.no_grad():
        with open(wv_file_path, encoding="utf8") as f:
            for line in f:
                elems = line.rstrip().split(' ')
                vector_content = elems[1:]
                token = unicodedata.normalize('NFD', elems[0])
                if token in word2id:
                    count += 1
                    word_id = word2id[token]
                    nn_embeddings.weight[word_id] = torch.Tensor(
                        [float(v) for v in vector_content])
                    pretrain_id_list.append(word_id)

    embeddings = nn_embeddings.weight.data
    print('embedding matrix loaded.')
    print("#" * 40)
    print("total words in dataset: ", vocab_size)
    print("words in embedding matrix: ", count)
    print("Proportion: ", count / vocab_size * 100, "%")
    print("#" * 40)
    return nn_embeddings, embeddings, pretrain_id_list


def get_lower_case_name(text):
    """
    修改的分词机制
    Args:
        text: 词语字符串
    Returns:
        列表，用于map操作
    """

    # 特化处理特殊情况
    if text == "ALERTs":
        return [text]
    word_list = []
    # 处理全部大写的情况，发现日志中的一些全大写单词写法带小写覆盖关键信息
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


def pattern_to_vec_bgl(origin_log_path, wordvec_path, pattern_vec_out_path,
                       index_vec_out_path, PAD_IDX, EMM_IDX):
    """
    :param origin_log_path: bgl_2k.log的位置
    :param wordvec_path: crawl-300d.vec的位置
    :param pattern_vec_out_path: 要输出词向量json的位置
    :param index_vec_out_path: 索引-词向量json的输出位置
    :param PAD_IDX: 词语表中PAD_IDX的值，目前只能支持1
    :param EMM_IDX: 词语表中[EMBEDDING]的索引值，按论文意思是0，目前只支持0
    :return: 一个字典，key为日志id，value为一个(词向量长度，embedding维度(300) )的torch.tensor (在这个数据集中是(72,300))
             一个
    """

    file = open(origin_log_path, mode='r', encoding="utf-8")
    lines = file.read().splitlines()
    file.close()

    # 正则表达式处理:判断符号与数字
    ascii_pattern = r'\*|,|\.|/|;|\'|`|\[|\]|<|>|\?|:|"|\{|\}|\~|!|@|#|\$|%|\^|&|\(|\)|-|=|\_|\+|，|。|、|；|‘|’|【|】|·|！| |…|（|）'
    non_num_pattern = re.compile('[0-9]+')

    final_list = []
    seq_id_list = []

    seq_length = 0

    words_set = set()

    for line in lines:
        seq_id_list.append(line.split(" ")[1])
        line = " ".join(
            [x for x in line.split(" ") if not non_num_pattern.findall(x)])
        result_list = [x for x in re.split(ascii_pattern, line) if len(x) > 0]

        mid_list = list(map(get_lower_case_name, result_list))
        mid_list.append([])

        # TODO: 去除常用词如介词等，由于时间关系暂时没有实现论文里说的此功能
        remove_ascii_list = [
            x.lower() for x in re.split(ascii_pattern, mid_list.__str__())
            if len(x) > 0
        ]

        # 获取当前最长序列的长度作为输出矩阵的第二维度
        seq_length = seq_length if seq_length >= len(
            remove_ascii_list) else len(remove_ascii_list)

        final_seq = " ".join(remove_ascii_list)

        final_list.append(final_seq)
        words_set.update(final_seq.split(" "))

    # 建立词语表，根据词语索引建立词语表映射
    words_list = list(words_set)

    words_list.sort()

    word2id = {}
    id2word = {}
    for i in range(len(words_list)):
        word2id[words_list[i]] = i + 2
        id2word[i + 2] = words_list[i]

    # 生成embedding
    # index2vector [word number, 300]
    embedding, index_to_tensor, pretrain_id_list = build_pretrained_embeddings(
        wordvec_path, id2word, word2id, PAD_IDX)

    index_to_vector = index_to_tensor.tolist()



    pattern_to_indexList = {}
    for i, sentence in enumerate(final_list):
        # 添加emm_idx和pad_idx
        seq_list = [PAD_IDX for _ in range(seq_length + 1)]
        seq_list[0] = EMM_IDX

        wd_list = sentence.split(" ")

        for index, word in enumerate(wd_list):
            # 因为第0个是emm_idx,所以这里需要加一
            seq_list[index + 1] = word2id[word]
        pattern_to_indexList[seq_id_list[i]] = seq_list

    # 存储index list
    json_str = json.dumps(pattern_to_indexList)
    with open(pattern_vec_out_path, 'w+') as file_obj:
        file_obj.write(json_str)

    # 储存index_to_vector
    json_str = json.dumps(index_to_vector)
    with open(index_vec_out_path, 'w+') as file_obj:
        file_obj.write(json_str)

    return index_to_tensor, pattern_to_indexList, embedding


def log_index_sequence_to_vec(src, index_tensor_path, EMB_DIM=0, PAD_DIM=1):
    """
    输入一个日志索引序列src[batch,seq_len]，每一个batch点的数据是类似[0,idx,idx,..,1,1,1]
    的形式[0为[EMBEDDING],idx为实词索引，1为PAD的索引。对每个词找到对应的词向量
    :param src: 日志索引序列
    :param index_tensor_path: fasttext词向量集合的存储位置
    :param EMB_DIM: [EMBEDDING]的值，默认且目前只能0
    :param PAD_DIM: PAD的索引值，默认且目前只能1
    :return: 即将输入Encoder的向量即[batch,seq_len,embedding_dim]的矩阵
    """
    assert EMB_DIM == 0 and PAD_DIM == 1

    with open(index_tensor_path, 'r') as index_tensor:
        index_vocab = json.load(index_tensor)

    batch_size = src.shape[0]
    seq_len = src.shape[1]
    embedding_dim = len(index_vocab[0])

    if not hasattr(log_index_sequence_to_vec, 'embedding_origin'):
        log_index_sequence_to_vec.embedding_origin = torch.sigmoid(
            torch.randn(embedding_dim))

    embedding_matrix = torch.zeros(batch_size, seq_len, embedding_dim)

    for i, log in enumerate(src):
        for j, word_index in enumerate(log):
            if word_index == EMB_DIM:
                embedding_matrix[i][j] = log_index_sequence_to_vec.embedding_origin
            elif word_index == PAD_DIM:
                continue
            else:
                embedding_matrix[i][j] = torch.Tensor(index_vocab[int(word_index)])

    return embedding_matrix
