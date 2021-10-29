import json
import time

import pandas
import torch
from torch import nn

from loganomaly.Depricated.log_anomaly_quantitive_predict import load_quantitive_model
from loganomaly.Depricated.log_anomaly_sequential_predict import load_sequential_model

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

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

def filter_small_top_k(predicted, output):
    filter = []
    for p in predicted:
        if output[0][p] > 0.001:
            filter.append(p)
    return filter


class Model(nn.Module):
    def __init__(self, input_size_0,input_size_1, hidden_size, num_of_layers, out_size):
        super(Model, self).__init__()
        self.hidden_size = hidden_size
        self.num_of_layers = num_of_layers
        self.lstm0 = nn.LSTM(input_size_0, hidden_size, num_of_layers, batch_first=True)
        self.lstm1 = nn.LSTM(input_size_1, hidden_size, num_of_layers, batch_first=True)
        self.fc = nn.Linear(2*hidden_size, out_size)


    def forward(self, input_0,input_1):
        h0_0 = torch.zeros(self.num_of_layers, input_0.size(0), self.hidden_size).to(device)
        c0_0 = torch.zeros(self.num_of_layers, input_0.size(0), self.hidden_size).to(device)
        out_0, _ = self.lstm0(input_0, (h0_0, c0_0))
        h0_1 = torch.zeros(self.num_of_layers, input_1.size(0), self.hidden_size).to(device)
        c0_1 = torch.zeros(self.num_of_layers, input_1.size(0), self.hidden_size).to(device)
        out_1, _ = self.lstm1(input_1, (h0_1, c0_1))
        multi_out = torch.cat((out_0[:, -1, :], out_1[:, -1, :]), -1)
        out = self.fc(multi_out)
        return out

def load_model(seq_input_size, quan_input_size, hidden_size, num_layers, num_classes, model_path):
    model = Model(seq_input_size, quan_input_size,hidden_size, num_layers, num_classes).to(device)
    model.load_state_dict(torch.load(model_path, map_location='cpu'))
    model.eval()
    print('model_path: {}'.format(model_path))
    return model

def do_predict(seq_input_size, quan_input_size, hidden_size, num_layers, num_classes, window_length, seq_model_path, quan_model_path, anomaly_test_line_path, test_file_path, num_candidates, pattern_vec_file):
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

    window_length = 5
    with open(pattern_vec_file, 'r') as pattern_file:
        PF = json.load(pattern_file)
    keys = {}
    with open(pattern_vec_file, 'r') as pattern_file:
        i = 0
        for key, value in PF.items():
            pattern, vec = key, value
            pattern_vector = "0x"+pattern
            keys[int(pattern_vector,16)] = i
            i = i + 1

    sequential_model = load_sequential_model(seq_input_size, hidden_size, num_layers, num_classes, seq_model_path)
    quantitive_model = load_quantitive_model(quan_input_size, hidden_size, num_layers, num_classes, quan_model_path)

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
        label = []
        for line in Test_File_loader:
            #print(line)
            i = 0
            # first traverse [0, window_size)
            #print(lineNum, line)
            while i < len(line) - window_length:
                window_input_sequential = []
                window_input_quantitive = [0] * 31
                seq = []
                quan = []
                print(line[i:i + window_length])
                for j in range(i, i + window_length):
                    window_input_sequential.append(pattern_vec[line[i]])
                    for key, value in keys.items():
                        pattern, num = key, value
                        if line[j] == pattern:
                            window_input_quantitive[keys[pattern]] += 1

                # print(len(window_input))
                quan = window_input_quantitive
                seq.append(window_input_sequential)
                print(quan)
                #lineNum = current_file_line * 200 + i + window_length + 1
                #count_num += 1

                #seq = line[i:i + window_length]

                #label = line[i + window_length]
                label = keys[line[i + window_length]]
                #for n in range(len(seq)):
                #    if current_file_line * 200 + i + n + 1 in abnormal_label:
                #        i = i + n + 1
                #        continue
                seq = torch.tensor(seq, dtype=torch.float).view(-1, window_length, seq_input_size).to(device)
                quan = torch.tensor(quan, dtype=torch.float).view(-1, num_classes, 1).to(device)

                seq = torch.tensor(seq, dtype=torch.float).view(-1, window_length, input_size_1).to(device)
                quan = torch.tensor(quan, dtype=torch.float).view(-1, window_length, input_size_2).to(device)
                test_output = model(seq, quan)
                predicted = torch.argsort(test_output, 1)[0][-num_candidates:]
                #label = torch.tensor(label).view(-1).to(device)
                output_sequential = sequential_model(seq)
                output_quantitive = quantitive_model(quan)

                predicted_seq = torch.argsort(output_sequential, 1)[0][-num_candidates:]
                predicted_quan = torch.argsort(output_quantitive, 1)[0][-num_candidates:]

                print(predicted_seq, predicted_quan)
                #print(output_sequential)
                #print(torch.argsort(output_sequential, 1)[0])
                #print(torch.argsort(output_quantitive, 1)[0])
                #output = torch.cat((output_sequential, output_quantitive), dim=1)
                #predicted = torch.argsort(output, 1)[0][-num_candidates:]
                #print(predicted)
                #predicted = torch.argsort(output, 1)[0][-num_candidates:]
                output = predicted_seq+predicted_quan
                print(output)
                #predicted = filter_small_top_k(predicted, output)
                print('{} - predict result: {}, true label: {}'.format(count_num, predicted, label))
                if lineNum in abnormal_label:  ## 若出现异常日志，则接下来的预测跳过异常日志，保证进行预测的日志均为正常日志
                    i += window_length + 1
                    skip_count += 1
                else:
                    i += 1
                ALL += 1
                if label not in output:
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
