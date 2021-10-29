import json

import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
from tensorboardX import SummaryWriter
from torch.utils.data import TensorDataset, DataLoader
import numpy as np
import argparse
import os

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


class Model(nn.Module):
    def __init__(self, input_size, hidden_size, num_of_layers, out_size):
        super(Model, self).__init__()
        self.hidden_size = hidden_size
        self.num_of_layers = num_of_layers
        self.lstm = nn.LSTM(input_size, hidden_size, num_of_layers, batch_first=True, dropout=0.5)
        self.fc = nn.Linear(hidden_size, out_size)

    def init_hidden(self, size):
        h0 = torch.zeros(self.num_of_layers, size, self.hidden_size).to(device)
        c0 = torch.zeros(self.num_of_layers, size, self.hidden_size).to(device)
        return (h0, c0)

    def forward(self, input):
        # h_n: hidden state h of last time step
        # c_n: hidden state c of last time step
        out, _ = self.lstm(input, self.init_hidden(input.size(0)))
        # the output of final time step
        out = self.fc(out[:, -1, :])
        # print('out[:, -1, :]:')
        # print(out)
        return out

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
    #print(keys)

    input_data = []
    output_data = []
    #print(num_of_classes)
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
            #print((input_data), (output_data))
    #print((input_data), (output_data))
    data_set = TensorDataset(torch.tensor(input_data, dtype=torch.float), torch.tensor(output_data))
    return data_set

def train_model(window_length, input_size, hidden_size, num_of_layers, num_of_classes, num_epochs, batch_size,
                root_path, model_output_directory, logkey_path, pattern_vec_file):
    # log setting
    print("Train num_classes: ", num_of_classes)
    log_directory = root_path + 'quantitive_log_out/'
    log_template = 'Adam_batch_size=' + str(batch_size) + ';epoch=' + str(num_epochs)

    window_length = 5
    model = Model(input_size, hidden_size, num_of_layers, num_of_classes).to(device)
    # create data set
    quantitive_data_set = generate_quantitive_label_new(logkey_path, window_length, num_of_classes, pattern_vec_file)
    # create data_loader
    data_loader = DataLoader(dataset=quantitive_data_set, batch_size=batch_size, shuffle=True, pin_memory=False)
    writer = SummaryWriter(logdir=log_directory + log_template)

    # Loss and optimizer  classify job
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters())

    # Training
    for epoch in range(num_epochs):
        train_loss = 0
        for step, (quan, label) in enumerate(data_loader):
            quan = quan.clone().detach().view(-1, num_of_classes, input_size).to(device)

            output = model(quan)
            #output = output.view([5, 31])
            #print(output, label)
            #print(output.shape, label.shape)

            loss = criterion(output, label.to(device))
            # Backward and optimize
            optimizer.zero_grad()
            loss.backward()
            train_loss += loss.item()
            optimizer.step()
        print(
            'Epoch [{}/{}], training_loss: {:.10f}'.format(epoch + 1, num_epochs, train_loss / len(data_loader.dataset)))
        if (epoch + 1) % num_epochs == 0:
            if not os.path.isdir(model_output_directory):
                os.makedirs(model_output_directory)
            e_log = 'Adam_batch_size=' + str(batch_size) + ';epoch=' + str(epoch + 1)
            torch.save(model.state_dict(), model_output_directory + '/' + e_log + '.pt')
    writer.close()
    print('Training finished')


"""
def generate_quantitive_label(logkey_path, window_length, num_of_classes, pattern_vec_file):

    #f = open(logkey_path, 'r')
    #keys = f.readline().split()
    #keys = list(map(int, keys))
    #print(keys)
    #length = len(keys)

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
    print(keys)
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

    print(input.shape)
    print(output.shape)
    print(input[0])
    print(output[0])

    dataset = TensorDataset(torch.tensor(new_input, dtype=torch.float), torch.tensor(new_output, dtype=torch.long))
    return dataset
"""