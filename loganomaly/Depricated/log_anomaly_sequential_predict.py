# -*- coding: UTF-8 -*-
import json
import time

import pandas
import torch

from loganomaly.Depricated.log_anomaly_sequential_train import Model

# use cuda if available  otherwise use cpu
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# len(line) < window_length

def generate(name, window_length):
    """
    log_keys_sequences = list()
    with open(name, 'r') as f:
        for line in f.readlines():
            line = tuple(map(lambda n: tuple(map(float, n.strip().split())), [x for x in line.strip().split(',') if len(x) > 0]))
            # for i in range(len(line) - window_size):
            #     inputs.add(tuple(line[i:i+window_size]))
            log_keys_sequences.append(tuple(line))
    return log_keys_sequences
    """
    log_keys_sequences = list()
    file = pandas.read_csv(name)
    for i in range(len(file)):
        line = [int(id, 16) for id in file["Sequence"][i].strip().split(' ')]
        #print(line)
        #print(line[i:i + window_length])
        log_keys_sequences.append(tuple(line))
    #print(log_keys_sequences)
    return log_keys_sequences

def getAbnormalLabel(test_file_path):
    abnormal_label = list()
    file = pandas.read_csv(test_file_path)
    for i in range(len(file)):
        label = file["label"][i]
        #print(label)
        if label ==1:
            abnormal_label.append(i)
    #print(abnormal_label)
    return abnormal_label


def load_sequential_model(input_size, hidden_size, num_layers, num_classes, model_path):
    model1 = Model(input_size, hidden_size, num_layers, num_classes).to(device)
    model1.load_state_dict(torch.load(model_path, map_location='cpu'))
    model1.eval()
    print('model_path: {}'.format(model_path))
    return model1


def filter_small_top_k(predicted, output):
    filter = []
    for p in predicted:
        if output[0][p] > 0.001:
            filter.append(p)
    return filter


def do_predict(input_size, hidden_size, num_layers, num_classes, window_length, model_path, anomaly_test_line_path, test_file_path, num_candidates, pattern_vec_file):
    with open(pattern_vec_file, 'r') as pattern_file:
        PF = json.load(pattern_file)
    vec_to_class_type = {}
    pattern_vec = {}
    with open(pattern_vec_file, 'r') as pattern_file:
        i = 0
        for key, value in PF.items():
            pattern, vec = key, value
        #for line in pattern_file.readlines():
        #    print(type(line))
        #    pattern, vec = line.split('[:]')
        #    print(pattern, vec)
            pattern_vector = "0x"+pattern
            vec_to_class_type[int(pattern_vector,16)] = i
            pattern_vec[int(pattern_vector, 16)] = vec
            i = i + 1

    sequential_model = load_sequential_model(input_size, hidden_size, num_layers, num_classes, model_path)

    start_time = time.time()
    TP = 0
    FP = 0
    TN = 0
    FN = 0
    ALL = 0
    skip_count = 0
    Test_File_loader = generate(test_file_path, window_length)
    abnormal_label = getAbnormalLabel(test_file_path)
    print('predict start')
    with torch.no_grad():
        count_num = 0
        #current_file_line = 0
        lineNum = 0
        for line in Test_File_loader:
            i = 0
            # first traverse [0, window_size)
            #print(lineNum, line)
            while i < len(line) - window_length:
                window_input = []
                seq = []
                for j in range(i, i + window_length):
                    window_input.append(pattern_vec[line[i]])
                # print(line[i:i + window_length])
                # print(len(window_input))
                seq.append(window_input)
                #lineNum = current_file_line * 200 + i + window_length + 1
                #count_num += 1

                #seq = line[i:i + window_length]

                label = line[i + window_length]
                #for n in range(len(seq)):
                #    if current_file_line * 200 + i + n + 1 in abnormal_label:
                #        i = i + n + 1
                #        continue
                seq = torch.tensor(seq, dtype=torch.float).view(-1, window_length, input_size).to(device)
                #label = torch.tensor(label).view(-1).to(device)
                output = sequential_model(seq)
                predicted = torch.argsort(output, 1)[0][-num_candidates:]
                predicted = filter_small_top_k(predicted, output)
                print('{} - predict result: {}, true label: {}'.format(count_num, predicted, vec_to_class_type[label]))
                if lineNum in abnormal_label:  ## 若出现异常日志，则接下来的预测跳过异常日志，保证进行预测的日志均为正常日志
                    i += window_length + 1
                    skip_count += 1
                else:
                    i += 1
                ALL += 1
                if vec_to_class_type[label] not in predicted:
                    if lineNum in abnormal_label:
                        TP += 1
                    else:
                        FP += 1
                else:
                    if lineNum in abnormal_label:
                        FN += 1
                    else:
                        TN += 1
            #current_file_line += 1
            lineNum += 1
    # Compute precision, recall and F1-measure
    if TP + FP == 0:
        P = 0
    else:
        P = 100 * TP / (TP + FP)

    if TP + FN == 0:
        R = 0
    else:
        R = 100 * TP / (TP + FN)

    if P + R == 0:
        F1 = 0
    else:
        F1 = 2 * P * R / (P + R)

    Acc = (TP + TN) * 100 / ALL

    print('FP: {}, FN: {}, TP: {}, TN: {}'.format(FP, FN, TP, TN))
    print('Acc: {:.3f}, Precision: {:.3f}%, Recall: {:.3f}%, F1-measure: {:.3f}%'.format(Acc, P, R, F1))
    print('Finished Predicting')
    elapsed_time = time.time() - start_time
    print('elapsed_time: {}'.format(elapsed_time))
    print('skip_count: {}'.format(skip_count))
    #draw_evaluation("Evaluations", ['Acc', 'Precision', 'Recall', 'F1-measure'], [Acc, P, R, F1], 'evaluations', '%')