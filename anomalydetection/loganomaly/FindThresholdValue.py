# coding:utf-8
import json
import time

import numpy as np
import LogAnomaly_Test
import torch
from sklearn.metrics import precision_recall_curve

from LogAnomaly_Train import Model

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def linePrediction_Threshold(predicted, label, threshold):
    dim0, dim1 = predicted.shape  # predicted is the output of all the windows in a log block
    abnormal_flag = 0
    for i in range(dim0):
        #print(label[i], predicted[i][label[i]])
        if predicted[i][label[i]] < threshold:
            abnormal_flag = 1
    return abnormal_flag


def generate_predict_and_label(predicted, label, ground_truth):
    dim0, dim1 = predicted.shape  # predicted is the output of all the windows in a log block
    if ground_truth == 0:
        maxPre = 0
        for i in range(dim0):
            predict.append(predicted[i][label[i]])
            label_.append(1)
            maxPre = max(maxPre, predicted[i][label[i]])
        # predict.append(maxPre)
        # label_.append(1)
    else:
        minPre = 100000
        for i in range(dim0):
            minPre = min(predicted[i][label[i]], minPre)
        label_.append(0)
        predict.append(minPre)


"""The general idea is that since the label is attached to each block (each line), the window(length=5) is 
moved down each line. As the window moves down each line, if the predicted result doesn't match the ground 
truth in any of these windows, this block (this line) is flagged as abnormal. """
def get_threshold_value(window_length, input_size_sequential, input_size_quantitive, hidden_size, num_of_layers, num_of_classes,
                        model_output_directory, valid_file, pattern_vec_file):
    model = LogAnomaly_Test.load_model(input_size_sequential, input_size_quantitive, hidden_size, num_of_layers, num_of_classes,
                       model_output_directory)
    with open(pattern_vec_file, 'r') as pattern_file:
        PF = json.load(pattern_file)
        pattern_vec = {}

        # Cast each log event to its pattern vector
        i = 0
        for key, value in PF.items():
            pattern, vec = key, value
            pattern_vec[int(pattern)] = vec
            i = i + 1

    test_file_loader, abnormal_label = LogAnomaly_Test.generateTestFile(valid_file, window_length)  # Load test file

    start_time = time.time()
    print('Start Prediction')
    with torch.no_grad():
        batch_num = 0
        lineNum = 0
        n = 0
        # Batch with full length
        while n < (len(test_file_loader) - len(test_file_loader) % test_batch_size):
            batch_input_sequential = []
            batch_input_quantitative = []
            batch_label = []
            line_windowNum = []
            #  Each batch has (test_batch_size) log blocks
            for x in range(n, n + test_batch_size):
                line = test_file_loader[x]
                # Each line represents a block of log event
                i = 0
                # Skip the lines that are too short
                if len(line) < window_length:
                    continue
                # Slide the window in each line (window_length=5)
                while i < len(line) - window_length:
                    window_input_sequential = []
                    window_input_quantitative = []
                    for j in range(i, i + window_length):
                        # For sequential pattern, each log event in the window is cast to its pattern vector which eventually leads to a shape of 5*300
                        window_input_sequential.append(pattern_vec[line[i]])

                        quantitative_subwindow = [0] * num_of_classes  # Initiate quantitative input window
                        # For quantitative pattern, the quantitative window is used to generate the count vector (shape=5*31)
                        if j >= window_length:  # Full length windows
                            for k in range(j - window_length + 1, j + 1):
                                quantitative_subwindow[line[k]] += 1
                            window_input_quantitative.append(quantitative_subwindow)
                        else:  # Partial length windows
                            for m in range(0, j + 1):
                                quantitative_subwindow[line[m]] += 1
                            window_input_quantitative.append(quantitative_subwindow)

                    # The label is the index of the next log event
                    batch_label.append(line[i + window_length])
                    batch_input_sequential.append(window_input_sequential)
                    batch_input_quantitative.append(window_input_quantitative)
                    i += 1
                line_windowNum.append(i)

            seq = batch_input_sequential
            quan = batch_input_quantitative

            seq = torch.tensor(seq, dtype=torch.float).view(-1, window_length, input_size_sequential).to(device)
            quan = torch.tensor(quan, dtype=torch.float).view(-1, window_length, input_size_quantitive).to(device)
            # print(seq.shape, quan.shape)
            test_output = model(seq, quan)
            # print(test_output.shape)
            #  Reconstruct the output to the original log blocks
            current_window_num = 0
            for k in range(
                    len(line_windowNum)):  # Reconstruct every line in a batch, each line has line_windowNum[k] windows
                line_label = []
                num_of_windows = line_windowNum[k]
                line_output = torch.empty(num_of_windows, num_of_classes)

                for i in range(current_window_num, current_window_num + num_of_windows):
                    line_output[i - current_window_num] = test_output[i]
                    line_label.append(batch_label[i])

                if lineNum in abnormal_label:
                    ground_truth = 1
                else:
                    ground_truth = 0

                generate_predict_and_label(line_output, line_label, ground_truth)
                lineNum += 1
                current_window_num += num_of_windows
                # End of for loop. Move on to the next line (Next block of log events)
            batch_num += 1
            n += test_batch_size

            # End of while loop. Deal with the remaining part.
        if n >= (len(test_file_loader) - len(test_file_loader) % test_batch_size):
            batch_input_sequential = []
            batch_input_quantitative = []
            batch_label = []
            line_windowNum = []
            # Deal with the remaining part
            for y in range(n, len(test_file_loader)):
                line = test_file_loader[y]
                # Each line represents a block of log event
                i = 0
                # Skip the lines that are too short
                if len(line) < window_length:
                    continue
                # Slide the window in each line (window_length=5)
                while i < len(line) - window_length:
                    window_input_sequential = []
                    window_input_quantitative = []
                    for j in range(i, i + window_length):
                        # For sequential pattern, each log event in the window is cast to its pattern vector which eventually leads to a shape of 5*300
                        window_input_sequential.append(pattern_vec[line[i]])

                        quantitative_subwindow = [0] * num_of_classes  # Initiate quantitative input window
                        # For quantitative pattern, the quantitative window is used to generate the count vector (shape=5*31)
                        if j >= window_length:  # Full length windows
                            for k in range(j - window_length + 1, j + 1):
                                quantitative_subwindow[line[k]] += 1
                            window_input_quantitative.append(quantitative_subwindow)
                        else:  # Partial length windows
                            for m in range(0, j + 1):
                                quantitative_subwindow[line[m]] += 1
                            window_input_quantitative.append(quantitative_subwindow)

                    # The label is the index of the next log event
                    batch_label.append(line[i + window_length])
                    batch_input_sequential.append(window_input_sequential)
                    batch_input_quantitative.append(window_input_quantitative)
                    i += 1
                line_windowNum.append(i)

            seq = batch_input_sequential
            quan = batch_input_quantitative

            seq = torch.tensor(seq, dtype=torch.float).view(-1, window_length, input_size_sequential).to(device)
            quan = torch.tensor(quan, dtype=torch.float).view(-1, window_length, input_size_quantitive).to(device)
            # print(seq.shape, quan.shape)
            test_output = model(seq, quan)
            # print(test_output.shape)

            current_window_num = 0
            for k in range(len(line_windowNum)):
                line_label = []
                num_of_windows = line_windowNum[k]
                line_output = torch.empty(num_of_windows, num_of_classes)

                for i in range(current_window_num, current_window_num + num_of_windows):
                    line_output[i - current_window_num] = test_output[i]
                    line_label.append(batch_label[i])

                if lineNum in abnormal_label:
                    ground_truth = 1
                else:
                    ground_truth = 0

                generate_predict_and_label(line_output, line_label, ground_truth)

                # When this line(block) is flagged as abnormal
                lineNum += 1
                current_window_num += num_of_windows
                # End of for loop. Move on to the next line (Next block of log events)

    # Compute precision, recall and F1-measure
    elapsed_time = time.time() - start_time
    print('elapsed_time: {}'.format(elapsed_time))
    precisions, recalls, thresholds = precision_recall_curve(label_, predict)

    # 拿到最优结果以及索引
    f1_scores = (2 * precisions * recalls) / (precisions + recalls)
    best_f1_score = np.max(f1_scores[np.isfinite(f1_scores)])
    best_f1_score_index = np.argmax(f1_scores[np.isfinite(f1_scores)])

    # 阈值
    print('best_f1_score: {}, threshold: {}'.format(best_f1_score, thresholds[best_f1_score_index]))
    return thresholds[best_f1_score_index]


if __name__ == '__main__':
    predict = []

    label_ = []
    hidden_size = 128
    num_of_layers = 2
    num_of_classes = 31
    num_epochs = 15

    window_length = 5
    input_size_sequential = 300
    input_size_quantitive = 31
    batch_size = 512
    test_batch_size = 64

    num_candidates = 5
    threshold = 3.714398e-07

    logparser_structed_file = '../../Data/logparser_result/Drain/HDFS_split_40w.log_structured.csv'
    logparser_event_file = '../../Data/logparser_result/Drain/HDFS_split_40w.log_templates.csv'
    anomaly_label_file = '../../Data/log/hdfs/anomaly_label.csv'

    sequential_directory = '../../Data/DrainResult-HDFS/loganomaly/sequential_files/'
    train_file_name = 'loganomaly_train_file'
    test_file_name = 'loganomaly_test_file'
    valid_file_name = 'loganomaly_valid_file'

    train_file = sequential_directory + train_file_name
    test_file = sequential_directory + test_file_name
    valid_file = sequential_directory + valid_file_name
    train_root_path = '../../Data/DrainResult-HDFS/loganomaly_model_train/'
    model_out_path = train_root_path + 'model_out/'

    wordvec_file_path = '../crawl-300d-2M.vec'
    pattern_vec_out_path = '../../Data/DrainResult-HDFS/loganomaly_model_train/pattern_vec'

    get_threshold_value(window_length, input_size_sequential, input_size_quantitive, hidden_size, num_of_layers, num_of_classes,
               model_out_path + 'Adam_batch_size=' + str(batch_size) + ';epoch=' + str(num_epochs) + '.pt',
               valid_file, pattern_vec_out_path)
