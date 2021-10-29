import pandas as pd
import json
import torch
import torch.nn as nn
import torch.optim as optim
from tensorboardX import SummaryWriter
from torch.utils.data import TensorDataset, DataLoader
import numpy as np
import argparse
import os
from loganomaly import *

# use cuda if available  otherwise use cpu
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

def generate_seq_label_new(file_path, window_length, pattern_vec_file):
    with open(pattern_vec_file, 'r') as pattern_file:
        PF = json.load(pattern_file)
    #print(type(PF))
    vec_to_class_type = {}
    with open(pattern_vec_file, 'r') as pattern_file:
        i = 0
        for key, value in PF.items():
            pattern, vec = key, value
        #for line in pattern_file.readlines():
        #    print(type(line))
        #    pattern, vec = line.split('[:]')
            pattern_vector = "0x"+pattern
            vec_to_class_type[int(pattern_vector,16)] = i
            i = i + 1
    #print(vec_to_class_type)
    input_data, output_data = [], []
    train_file = pd.read_csv(file_path)
    for i in range(len(train_file)):
        line = [int(id, 16) for id in train_file["Sequence"][i].strip().split(' ')]
        #print(line)
        if len(line) < window_length:
            continue
        for i in range(len(line) - window_length):
            #print(line[i:i + window_length])
            input_data.append(line[i:i + window_length])
            # line[i] is a list need to read file form a dic{vec:log_key} to get log key
            #print(vec_to_class_type[line[i + window_length]])
            output_data.append(vec_to_class_type[line[i + window_length]])
    #print(len(input_data), len(output_data))
    #print(torch.tensor(input_data, dtype=torch.float), torch.tensor(output_data))
    data_set = TensorDataset(torch.tensor(input_data, dtype=torch.float), torch.tensor(output_data))
    return data_set

def generate_seq_label_new_2(file_path, window_length, pattern_vec_file):
    with open(pattern_vec_file, 'r') as pattern_file:
        PF = json.load(pattern_file)
    #print(type(PF))
    vec_to_class_type = {}
    pattern_vec = {}
    with open(pattern_vec_file, 'r') as pattern_file:
        i = 0
        for key, value in PF.items():
            pattern, vec = key, value
        #for line in pattern_file.readlines():
        #    print(type(line))
        #    pattern, vec = line.split('[:]')
            pattern_vector = "0x"+pattern
            vec_to_class_type[int(pattern_vector,16)] = i
            pattern_vec[int(pattern_vector,16)] = vec
            i = i + 1
    #print(vec_to_class_type)
    input_data, output_data = [], []
    train_file = pd.read_csv(file_path)
    for i in range(len(train_file)):
        line = [int(id, 16) for id in train_file["Sequence"][i].strip().split(' ')]
        #print(line)
        if len(line) < window_length:
            continue
        for i in range(len(line) - window_length):
            window_input = []
            for j in range(i, i + window_length):
                window_input.append(pattern_vec[line[i]])
            #print(line[i:i + window_length])
            #print(len(window_input))
            input_data.append(window_input)
            # line[i] is a list need to read file form a dic{vec:log_key} to get log key
            #print(vec_to_class_type[line[i + window_length]])
            output_data.append(vec_to_class_type[line[i + window_length]])
    #print(len(input_data), len(output_data))
    #print(torch.tensor(input_data, dtype=torch.float), torch.tensor(output_data))
    data_set = TensorDataset(torch.tensor(input_data, dtype=torch.float), torch.tensor(output_data))
    return data_set

def train_model(window_length, input_size, hidden_size, num_of_layers, num_of_classes, num_epochs, batch_size, root_path, model_output_directory, data_file, pattern_vec_file):
    # log setting
    log_directory = root_path + 'sequential_log_out/'
    log_template = 'Adam_batch_size=' + str(batch_size) + ';epoch=' + str(num_epochs)

    print("Train num_classes: ", num_of_classes)
    model = Model(input_size, hidden_size, num_of_layers, num_of_classes).to(device)
    # create data set
    sequence_data_set = generate_seq_label_new_2(data_file, window_length, pattern_vec_file)
    # create data_loader
    data_loader = DataLoader(dataset=sequence_data_set, batch_size=batch_size, shuffle=True, pin_memory=False)
    writer = SummaryWriter(logdir=log_directory + log_template)

    # Loss and optimizer  classify job
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters())

    # Training
    for epoch in range(num_epochs):
        train_loss = 0
        for step, (seq, label) in enumerate(data_loader):
            seq = seq.clone().detach().view(-1, window_length, input_size).to(device)
            #print(seq.shape)
            output = model(seq)
            #print(output.shape, label.shape)
            loss = criterion(output, label.to(device))

            # Backward and optimize
            optimizer.zero_grad()
            loss.backward()
            train_loss += loss.item()
            optimizer.step()
        print('Epoch [{}/{}], training_loss: {:.10f}'.format(epoch + 1, num_epochs, train_loss / len(data_loader.dataset)))
        if (epoch + 1) % num_epochs == 0:
            if not os.path.isdir(model_output_directory):
                os.makedirs(model_output_directory)
            e_log = 'Adam_batch_size=' + str(batch_size) + ';epoch=' + str(epoch+1)
            torch.save(model.state_dict(), model_output_directory + '/' + e_log + '.pt')
    writer.close()
    print('Training finished')

class Model(nn.Module):
    def __init__(self, input_size, hidden_size, num_of_layers, out_size):
        super(Model, self).__init__()
        self.hidden_size = hidden_size
        self.num_of_layers = num_of_layers
        self.lstm = nn.LSTM(input_size, hidden_size, num_of_layers, batch_first=True, dropout=0.5)
        self.fc = nn.Linear(hidden_size, out_size)
        # self.out = nn.Linear(in_features=in_features, out_features=out_features)

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
"""
def make_src_mask(src, src_pad_idx):
    # src = [batch, src len]

    src_mask = (src != src_pad_idx).unsqueeze(1).unsqueeze(2).to(device)
    # print(src_mask.shape, "make")
    # src_mask = [batch, 1, 1, src len]

    return src_mask

def generate_robust_seq_label(file_path, sequence_length):
    input_data, output_data, mask_data = [], [], []
    train_file = pd.read_csv(file_path)
    for i in range(len(train_file)):
        line = [int(id,16) for id in train_file["Sequence"][i].strip().split(' ')]
        line = line[0:sequence_length]
        if len(line) < sequence_length:
            line.extend(list([0]) * (sequence_length - len(line)))
        input_data.append(line)
        output_data.append(int(train_file["label"][i]))
    data_set = TensorDataset(torch.tensor(input_data), torch.tensor(output_data))
    return data_set


def get_batch_semantic_with_mask(seq, pattern_vec_file):
    with open(pattern_vec_file, 'r') as pattern_file:
        class_type_to_vec = json.load(pattern_file)
    print(seq)
    batch_data = []
    for s in seq:
        # print(s)
        semantic_line = []
        for event in s.numpy().tolist():
            if event == 0:
                semantic_line.append([-1] * 300)
            else:
                event = eval(hex(event))
                event = format(event, 'x')
                event = event.rjust(8, '0')
                
                semantic_line.append(class_type_to_vec[str(event)])
        batch_data.append(semantic_line)
    mask_data = make_src_mask(seq, 0)
    return batch_data, mask_data
def train_model(sequence_length, input_size, hidden_size, num_of_layers, num_of_classes, num_epochs, batch_size, root_path, model_output_directory, data_file, pattern_vec_file):
    print("Train num_classes: ", num_of_classes)
    with open(pattern_vec_file, 'r') as pattern_file:
        class_type_to_vec = json.load(pattern_file)
    print(class_type_to_vec)
    model = Model(input_size, hidden_size, num_of_layers, num_of_classes).to(device)
    # create data set
    sequence_data_set = generate_robust_seq_label(data_file, sequence_length)
    # create data_loader
    data_loader = DataLoader(dataset=sequence_data_set, batch_size=batch_size, shuffle=True, pin_memory=False)

    # Loss and optimizer  classify job
    criterion = nn.BCELoss()
    optimizer = optim.Adam(model.parameters())

    # Training
    for epoch in range(num_epochs):
        train_loss = 0
        for step, (seq, label) in enumerate(data_loader):
            batch_data, mask_data = get_batch_semantic_with_mask(seq, pattern_vec_file)
            # print(mask_data.shape)
            seq = torch.tensor(batch_data)
            #print(seq.shape)
            seq = seq.clone().detach().view(-1, sequence_length, input_size).to(device)
            #print(seq.shape)
            output = model(seq, mask_data)

            loss = criterion(output.squeeze(-1), label.float().to(device))

            # Backward and optimize
            optimizer.zero_grad()
            loss.backward()
            train_loss += loss.item()
            optimizer.step()
        print('Epoch [{}/{}], training_loss: {:.4f}'.format(epoch + 1, num_epochs, train_loss / len(data_loader.dataset)))
        if (epoch + 1) % num_epochs == 0:
            if not os.path.isdir(model_output_directory):
                os.makedirs(model_output_directory)
            e_log = 'Adam_batch_size=' + str(batch_size) + ';epoch=' + str(epoch+1) + ';sequence=' + str(sequence_length)
            torch.save(model.state_dict(), model_output_directory + '/' + e_log + '.pt')
    print('Training finished')

def generate_seq_label(file_path, window_length, pattern_vec_file):
    vec_to_class_type = {}
    with open(pattern_vec_file, 'r') as pattern_file:
        i = 0
        for line in pattern_file.readlines():
            print(line)
            pattern, vec = line.split(':')
            pattern_vector = tuple(map(float, vec.strip().split(' ')))
            vec_to_class_type[pattern_vector] = i
            i = i + 1
    num_of_sessions = 0
    input_data, output_data = [], []
    with open(file_path, 'r') as file:
        for line in file.readlines():
            num_of_sessions += 1
            line = tuple(map(lambda n: tuple(map(float, n.strip().split())), [x for x in line.strip().split(',') if len(x) > 0]))
            if len(line) < window_length:
                #print(line)
                continue
            for i in range(len(line) - window_length):
                input_data.append(line[i:i + window_length])
                # line[i] is a list need to read file form a dic{vec:log_key} to get log key
                output_data.append(vec_to_class_type[line[i + window_length]])
    data_set = TensorDataset(torch.tensor(input_data, dtype=torch.float), torch.tensor(output_data))
    return data_set
"""