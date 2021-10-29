import json
import os

import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
from tensorboardX import SummaryWriter
from torch.utils.data import TensorDataset, DataLoader

from extractfeature import template2Vec_preprocessor

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def generate_label(file_path, window_length, num_of_classes, pattern_vec_file):
    with open(pattern_vec_file, 'r') as pattern_file:
        PF = json.load(pattern_file)  # Load pattern_vec

        pattern_vec = {}

        # Cast each log event to its pattern vector
        i = 0
        for key, value in PF.items():
            pattern, vec = key, value
            pattern_vec[int(pattern)] = vec
            i = i + 1

        # print(pattern_vec, keys)
    seq_input_data, quan_input_data, output_data = [], [], []
    train_file = pd.read_csv(file_path)  # Load train file

    for i in range(len(train_file)):
        # Each line represents a block of log events
        line = [int(id) for id in train_file["Sequence"][i].strip().split(' ')]
        # Skip the lines that are too short
        if len(line) < window_length:
            continue
        # Slide the window in each line (window_length=5)
        for i in range(len(line) - window_length):
            window_input_sequential = []  # Initiate sequential input list
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

            seq_input_data.append(window_input_sequential)
            quan_input_data.append(window_input_quantitative)

            # The output is the index of the next log event
            output_data.append(line[i + window_length])

    seq = torch.tensor(seq_input_data, dtype=torch.float)
    quan = torch.tensor(quan_input_data, dtype=torch.float)
    in_tensor = torch.cat((seq, quan), dim=-1)     #Put sequential and quantitative data together to avoid problems caused by shuffling
    print("Input tensor shape:", in_tensor.shape)
    torch.tensor(output_data, dtype=torch.float)

    data_set = TensorDataset(in_tensor, torch.tensor(output_data, dtype=torch.float))
    return data_set


class Model(nn.Module):
    def __init__(self, input_size_0, input_size_1, hidden_size, num_of_layers, out_size):
        super(Model, self).__init__()
        self.hidden_size = hidden_size
        self.num_of_layers = num_of_layers
        self.lstm0 = nn.LSTM(input_size_0, hidden_size, num_of_layers, batch_first=True)
        self.lstm1 = nn.LSTM(input_size_1, hidden_size, num_of_layers, batch_first=True)
        self.fc = nn.Linear(2 * hidden_size, out_size)

    def forward(self, input_0, input_1):
        h0_0 = torch.zeros(self.num_of_layers, input_0.size(0), self.hidden_size).to(device)
        c0_0 = torch.zeros(self.num_of_layers, input_0.size(0), self.hidden_size).to(device)
        out_0, _ = self.lstm0(input_0, (h0_0, c0_0))
        h0_1 = torch.zeros(self.num_of_layers, input_1.size(0), self.hidden_size).to(device)
        c0_1 = torch.zeros(self.num_of_layers, input_1.size(0), self.hidden_size).to(device)
        out_1, _ = self.lstm1(input_1, (h0_1, c0_1))
        multi_out = torch.cat((out_0[:, -1, :], out_1[:, -1, :]), -1)
        out = self.fc(multi_out)
        return out


def train_model(window_length, input_size_sequential, input_size_quantitive, hidden_size, num_of_layers, num_of_classes,
                num_epochs,
                batch_size, root_path, model_output_directory, train_file, pattern_vec_file):
    # log setting
    log_directory = root_path + 'log_out/'
    log_template = 'Adam_batch_size=' + str(batch_size) + ';epoch=' + str(num_epochs)

    print("Train num_classes: ", num_of_classes)
    model = Model(input_size_sequential, input_size_quantitive, hidden_size, num_of_layers, num_of_classes).to(device)
    # create data set
    data_set = generate_label(train_file, window_length, num_of_classes, pattern_vec_file)

    # create data_loader
    data_loader = DataLoader(dataset=data_set, batch_size=batch_size, shuffle=True, pin_memory=False)
    writer = SummaryWriter(logdir=log_directory + log_template)

    # Loss and optimizer  classify job
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters())

    # Training
    for epoch in range(num_epochs):
        train_loss = 0
        for step, (in_tensor, label) in enumerate(data_loader):

            seq = in_tensor.index_select(2, torch.arange(300))
            quan = in_tensor.index_select(2, torch.arange(300, 300+num_of_classes))

            #print(seq.shape, quan.shape, label.shape)
            #Seperate the sequential and quantitative features
            seq = seq.clone().detach().view(-1, window_length, input_size_sequential).to(device)
            quan = quan.clone().detach().view(-1, window_length, input_size_quantitive).to(device)
            #print("Sequential shape:", seq.shape, "Quantitative shape:", quan.shape, "Label shape:", label.shape)
            output = model(seq, quan)
            # print(output.shape)
            # print(output, label)
            loss = criterion(output, label.to(device).long())

            # Backward and optimize
            optimizer.zero_grad()
            loss.backward()
            train_loss += loss.item()
            optimizer.step()
        print(
            'Epoch [{}/{}], training_loss: {:.6f}'.format(epoch + 1, num_epochs, train_loss / len(data_loader.dataset)))
        if (epoch + 1) % num_epochs == 0:
            if not os.path.isdir(model_output_directory):
                os.makedirs(model_output_directory)
            e_log = 'Adam_batch_size=' + str(batch_size) + ';epoch=' + str(epoch + 1)
            torch.save(model.state_dict(), model_output_directory + '/' + e_log + '.pt')
    writer.close()
    print('Training finished')


def extract_feature():
    template2Vec_preprocessor.generate_train_and_test_file(logparser_structed_file, logparser_event_file,
                                                           anomaly_label_file, sequential_directory,
                                                           train_file_name,
                                                           valid_file_name, test_file_name, wordvec_file_path,
                                                           pattern_vec_out_path, variable_symbol)

    template2Vec_preprocessor.pattern_to_vec_template_from_log(logparser_event_file, wordvec_file_path,
                                                               pattern_vec_out_path, variable_symbol)


if __name__ == '__main__':
    hidden_size = 128
    num_of_layers = 2
    num_of_classes = 31
    num_epochs = 10

    window_length = 5
    input_size_sequential = 300
    input_size_quantitive = 31
    batch_size = 500

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

    wordvec_file_path = '../crawl-300d-2M.vec'
    pattern_vec_out_path = '../../Data/DrainResult-HDFS/loganomaly_model_train/pattern_vec'
    variable_symbol = '<*> '

    # extract_feature()

    train_model(window_length, input_size_sequential, input_size_quantitive, hidden_size, num_of_layers, num_of_classes,
                num_epochs, batch_size, train_root_path, model_out_path, train_file, pattern_vec_out_path)
