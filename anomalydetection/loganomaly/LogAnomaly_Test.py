import json
import time
import pandas
import torch
from LogAnomaly_Train import Model

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def load_model(input_size_1, input_size_2, hidden_size, num_layers, num_classes, model_path):
    model = Model(input_size_1, input_size_2, hidden_size, num_layers, num_classes).to(device)
    model.load_state_dict(torch.load(model_path, map_location='cpu'))
    model.eval()
    print('model_path: {}'.format(model_path))
    return model


def generateTestFile(name):
    log_keys_sequences = list()
    file = pandas.read_csv(name)
    for i in range(len(file)):
        line = [int(id, 16) for id in file["Sequence"][i].strip().split(' ')]
        log_keys_sequences.append(tuple(line))
    return log_keys_sequences


def getAbnormalLineNumber(test_file_path):
    abnormal_label = list()
    file = pandas.read_csv(test_file_path)
    for i in range(len(file)):
        label = file["label"][i]
        # print(label)
        if label == 1:
            abnormal_label.append(i)
    # print(abnormal_label)
    return abnormal_label

def do_predict(window_length, input_size_sequential, input_size_quantitive, hidden_size, num_of_layers, num_of_classes,
               model_output_directory, test_file, pattern_vec_file, num_candidates):
    model = load_model(input_size_sequential, input_size_quantitive, hidden_size, num_of_layers, num_of_classes,
                       model_output_directory)
    TP = 0
    FP = 0
    TN = 0
    FN = 0
    ALL = 0

    with open(pattern_vec_file, 'r') as pattern_file:
        PF = json.load(pattern_file)
        pattern_vec = {}

        # Cast each log event to its pattern vector
        i = 0
        for key, value in PF.items():
            pattern, vec = key, value
            pattern_vec[int(pattern)] = vec
            i = i + 1

    test_file_loader = generateTestFile(test_file)  # Load test file
    abnormal_label = getAbnormalLineNumber(test_file)  # Get abnormal line number

    start_time = time.time()
    print('Start Prediction')
    with torch.no_grad():

        lineNum = 0
        abnormal_flag = 0

        """The general idea is that since the label is attached to each block (each line), the window(length=5) is 
        moved down each line. As the window moves down each line, if the predicted result doesn't match the ground 
        truth in any of these windows, this block (this line) is flagged as abnormal. """

        for line in test_file_loader:
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
                label = line[i + window_length]
                quan = window_input_quantitative
                seq = window_input_sequential
                # Feed the input into the model
                seq = torch.tensor(seq, dtype=torch.float).view(-1, window_length, input_size_sequential).to(device)
                quan = torch.tensor(quan, dtype=torch.float).view(-1, window_length, input_size_quantitive).to(device)
                #print(seq.shape, quan.shape)
                test_output = model(seq, quan)
                # print(test_output)
                # Sort the top 10 predicted results
                predicted = torch.argsort(test_output, 1)[0][-num_candidates:]
                # print(predicted)
                #print('{} - predict result: {}, true label: {}'.format(lineNum, predicted, label))
                # If the predicted result doesn't match the ground truth, this block is flagged as abnormal
                if label not in predicted:
                    abnormal_flag = 1

                i += 1  # Window slides down one line ->

            # When window is down to the end of the line, print the prediction result
            if lineNum in abnormal_label:
                ground_truth = 1
            else:
                ground_truth = 0

            print("line:", lineNum,"Predicted Label:", abnormal_flag, "Ground Truth:", ground_truth)

            # When this line(block) is flagged as abnormal
            if abnormal_flag == 1:
                if lineNum in abnormal_label:
                    TP += 1
                else:
                    FP += 1

            # When this line(block) is not flagged as abnormal
            else:
                if lineNum in abnormal_label:
                    FN += 1
                else:
                    TN += 1
            lineNum += 1
            ALL += 1
            abnormal_flag = 0

        # End of for loop. Move on to the next line (Next block of log events)


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


if __name__ == '__main__':
    hidden_size = 128
    num_of_layers = 2
    num_of_classes = 31
    num_epochs = 10

    window_length = 5
    input_size_sequential = 300
    input_size_quantitive = 31
    batch_size = 100

    num_candidates = 5

    logparser_structed_file = '../../Data/logparser_result/Drain/HDFS_split_40w.log_structured.csv'
    logparser_event_file = '../../Data/logparser_result/Drain/HDFS_split_40w.log_templates.csv'
    anomaly_label_file = '../../Data/log/hdfs/anomaly_label.csv'

    sequential_directory = '../../Data/DrainResult-HDFS/loganomaly/sequential_files/'
    train_file_name = 'loganomaly_train_file'
    test_file_name = 'loganomaly_test_file'
    valid_file_name = 'loganomaly_valid_file'

    train_file = sequential_directory + train_file_name
    test_file = sequential_directory + test_file_name
    train_root_path = '../../Data/DrainResult-HDFS/loganomaly_model_train/'
    model_out_path = train_root_path + 'model_out/'

    wordvec_file_path = 'G:\\crawl-300d-2M.vec'
    pattern_vec_out_path = '../../Data/DrainResult-HDFS/loganomaly_model_train/pattern_vec'

    do_predict(window_length, input_size_sequential, input_size_quantitive, hidden_size, num_of_layers, num_of_classes,
               model_out_path + 'Adam_batch_size=' + str(batch_size) + ';epoch=' + str(num_epochs) + '.pt',
               test_file, pattern_vec_out_path, num_candidates)
