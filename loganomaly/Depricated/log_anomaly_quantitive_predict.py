import json

import pandas
import pandas as pd
import torch
import time
from loganomaly.Depricated.log_anomaly_quantitive_train import Model

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
    print(log_keys_sequences)
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

def generate_quantitive_label_new(file_path, window_length, num_of_classes, pattern_vec_file):
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
    print(keys)

    input_data = []
    output_data = []
    print(num_of_classes)
    train_file = pd.read_csv(file_path)
    for i in range(len(train_file)):
        line = [int(id, 16) for id in train_file["Sequence"][i].strip().split(' ')]
        #print(line)
        if len(line) < window_length:
            continue
        for i in range(len(line) - window_length):
            window_input = [0] * 31
            for j in range(i, i + window_length):
                for key, value in keys.items():
                    pattern, num = key, value
                    if line[j] == pattern:
                        window_input[keys[pattern]] += 1
            input_data.append(window_input)
            output_data.append(keys[line[i + window_length]])

    return len(input_data), input_data, output_data


def load_quantitive_model(input_size, hidden_size, num_layers, num_classes, model_path):
    model2 = Model(input_size, hidden_size, num_layers, num_classes).to(device)
    model2.load_state_dict(torch.load(model_path, map_location='cpu'))
    model2.eval()
    print('model_path: {}'.format(model_path))
    return model2

def filter_small_top_k(predicted, output):
    filter = []
    for p in predicted:
        if output[0][p] > 0.001:
            filter.append(p)
    return filter


def do_predict(input_size, hidden_size, num_layers, num_classes, seq_length, model_path, anomaly_test_line_path, test_file_path, num_candidates, pattern_vec_file):
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

    #length, input, output = generate_quantitive_label_new(test_file_path, window_length, num_classes, pattern_vec_file)
    quantitive_model = load_quantitive_model(input_size, hidden_size, num_layers, num_classes, model_path)
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
        lineNum = 0
        #current_file_line = 0
        input_data = []
        label = []
        for line in Test_File_loader:
            #print(line)
            i = 0
            # first traverse [0, window_size)
            #print(i, len(line))
            while i < len(line) - window_length:
                #lineNum = current_file_line * 200 + i + window_length + 1
                count_num += 1
                window_input = [0] * 31
                for j in range(i, i + window_length):
                    for key, value in keys.items():
                        pattern, num = key, value
                        if line[j] == pattern:
                            window_input[keys[pattern]] += 1

                input_data = window_input
                label = keys[line[i + window_length]]

                print(input_data,label)
                #for n in range(len(seq)):
                #    if current_file_line * 200 + i + n + 1 in abnormal_label:
                #        i = i + n + 1
                #        continue
                quan = input_data
                quan = torch.tensor(quan, dtype=torch.float).view(-1, seq_length, 1).to(device)
                #label = torch.tensor(label).view(-1).to(device)
                output = quantitive_model(quan)
                predicted = torch.argsort(output, 1)[0][-num_candidates:]
                predicted = filter_small_top_k(predicted, output)
                print('{} - predict result: {}, true label: {}'.format(count_num, predicted, label))
                if lineNum in abnormal_label:  ## 若出现异常日志，则接下来的预测跳过异常日志，保证进行预测的日志均为正常日志
                    i += window_length + 1
                    skip_count += 1
                else:
                    i += 1
                ALL += 1
                if label not in predicted:
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
"""
def do_predict(input_size, hidden_size, num_layers, num_classes, window_length, model_path, anomaly_test_line_path, logkey_path, num_candidates, pattern_vec_file):
    quantitive_model = load_quantitive_model(input_size, hidden_size, num_layers, num_classes, model_path)
    start_time = time.time()
    TP = 0
    FP = 0
    TN = 0
    FN = 0
    ALL = 0
    length,input,output = generate_quantitive_label_new(logkey_path, window_length, num_classes, pattern_vec_file)
    abnormal_label = []
    #with open(anomaly_test_line_path) as f:
    #    abnormal_label = [int(x) for x in f.readline().strip().split()]
    print('predict start')
    with torch.no_grad():
        count_num = 0
        current_file_line = 0
        for i in range(0,length-2*window_length+1):
            lineNum = i + 2*window_length
            quan = input[i]
            label = output[i]
            quan = torch.tensor(quan, dtype=torch.float).view(-1, window_length, input_size).to(device)
            test_output = quantitive_model(quan)
            predicted = torch.argsort(test_output , 1)[0][-num_candidates:]
            print('{} - predict result: {}, true label: {}'.format(lineNum, predicted,label))
            if lineNum in abnormal_label:  ## 若出现异常日志，则接下来的预测跳过异常日志，保证进行预测的日志均为正常日志
                i += 2*window_length + 1
            else:
                i += 1
            ALL += 1
            if label not in predicted:
                if lineNum in abnormal_label:
                    TN += 1
                else:
                    FN += 1
            else:
                if lineNum in abnormal_label:
                    FP += 1
                else:
                    TP += 1
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


input_size = 61
hidden_size = 30
num_of_layers = 2
num_of_classes = 61
num_epochs = 100
batch_size = 200
window_length = 5
train_logkey_path = '../../Data/FTTreeResult-HDFS/deeplog_files/logkey/logkey_train'
test_logkey_path = '../../Data/FTTreeResult-HDFS/deeplog_files/logkey/logkey_test'
train_root_path = '../../Data/FTTreeResult-HDFS/model_train/'
label_file_name = '../../Data/FTTreeResult-HDFS/deeplog_files/HDFS_abnormal_label.txt'
model_out_path = train_root_path + 'quantitive_model_out/'
"""
# train_model(window_length, input_size, hidden_size,
#             num_of_layers, num_of_classes, num_epochs, batch_size, train_root_path,
#             model_out_path,train_logkey_path)

#do_predict(input_size, hidden_size, num_of_layers, num_of_classes, window_length,
#           model_out_path + 'Adam_batch_size=200;epoch=100.pt', label_file_name, 3, test_logkey_path)
"""
def generate_quantitive_test_label(logkey_path, window_length, num_of_classes, pattern_vec_file):
    keys = {}
    with open(pattern_vec_file, 'r') as pattern_file:
        PF = json.load(pattern_file)
    with open(pattern_vec_file, 'r') as pattern_file:
        i = 0
        for key, value in PF.items():
            pattern, vec = key, value
            pattern_vector = "0x"+pattern
            keys[int(pattern_vector,16)] = i
            i = i + 1
    #print(keys)
    log = list()
    train_file = pd.read_csv(logkey_path)
    for i in range(len(train_file)):
        line = [int(id, 16) for id in train_file["Sequence"][i].strip().split(' ')]
        log.append(line)
    log = [b for a in log for b in a]
    length = len(log)
    for i in range(length):
        for key, value in keys.items():
            pattern, num = key, value
            if log[i] == pattern:
                log[i] = num

    input = np.zeros((length - window_length, num_of_classes))
    output = np.zeros(length - window_length, dtype=np.int)
    for i in range(0, length - window_length):
        for j in range(i, i + window_length):
            input[i][log[j]] += 1
        output[i] = log[i + window_length]

    new_input = np.zeros((length - 2 * window_length + 1, window_length, num_of_classes))
    for i in range(0, length - 2 * window_length + 1):
        for j in range(i, i + window_length):
            new_input[i][j - i] = input[j]
    new_output = output[window_length - 1:]
    return length, new_input, new_output


def generate_test_label(logkey_path, window_length):
    f = open(logkey_path,'r')
    keys = f.readline().split()
    keys = list(map(int, keys))
    print(keys)
    length = len(keys)
    input = np.zeros((length -window_length,num_of_classes))
    output = np.zeros(length -window_length,dtype=np.int)
    for i in range(0,length -window_length):
        for j in range(i,i+window_length):
            input[i][keys[j]-1] += 1
        output[i] = keys[i+window_length]-1
    new_input = np.zeros((length -2*window_length+1,window_length,num_of_classes))
    for i in range(0,length -2*window_length+1):
        for j in range(i,i+window_length):
            new_input[i][j-i] = input[j]
    new_output = output[window_length-1:]
    print(new_input.shape)
    print(new_output.shape)
    print(new_input[0])
    print(new_output[0])
    return length,new_input,new_output
    """