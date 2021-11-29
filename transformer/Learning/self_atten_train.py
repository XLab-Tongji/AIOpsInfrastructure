import torch
import torch.nn as nn
import time
import math
import numpy as np
from torch.optim import optimizer


from transformer.bgl_preprocessor import log_index_sequence_to_vec

from transformer.Encoder.Encoder import *

PADDING_INDEX = 1


class SelfAttentive(nn.Module):
    def __init__(self, encoder, src_pad_idx):
        super().__init__()
        self.encoder = encoder
        self.src_pad_idx = src_pad_idx

    def make_src_mask(self, src):
        # src = [batch size, src len]

        src_mask = (src != self.src_pad_idx).unsqueeze(1).unsqueeze(2)

        # src_mask = [batch size, 1, 1, src len]

        return src_mask

    def forward(self, src, src_mask):
        enc_src = self.encoder(src, src_mask)

        # enc_src:[batch,seq len,emb_dim]

        # enc_src = self.encoder(src)

        # Rd->Rp
        # enc_src = torch.sigmoid(enc_src[:, -1, :])
        # print("enc_src :", enc_src)

        # enc_src = enc_src / torch.sum(enc_src)

        enc_src = enc_src[:, 0, :]

        # print("size",enc_src.size())
        # print("slice",enc_src)

        # self.enc_src.retain_grad()

        # print("enc_result:", enc_src)

        # enc_src = [batch size, src len, hid dim]
        # result = torch.linalg.norm(enc_src, dim=1, ord=2)**2
        result = torch.norm(enc_src, dim=1, p=2)

        # print("最后输出: ", result)
        # print("res:", result)

        return result
        # return 1 - torch.exp(-result)


class My_loss(nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, x, y):
        # x = [batch size]
        # y = [batch size]
        # print("src : ", x)
        # print("trg :", y)
        # print("loss:",
        #       torch.mean((1 - y) * x - y * torch.log(1 - torch.exp(-x))))
        return torch.mean((1 - y) * x - y * torch.log(1 - torch.exp(-x)))


def make_src_mask(src, PAD_INDEX):
    # src = [batch size, src len]

    src_mask = (src != PAD_INDEX).unsqueeze(1).unsqueeze(2)

    return src_mask


'''
INDEX_TO_TENSOR,NN_EMBEDDING这两个参数是无效参数，需要去掉
'''


def train(model, INDEX_TO_TENSOR, NN_EMBEDDING, INDEX_VEC_PATH, iterator,
          optimizer, criterion, clip):
    model.train()

    epoch_loss = 0

    for i, batch in enumerate(iterator):
        src = torch.tensor(batch[0], dtype=torch.float, requires_grad=True)
        trg = torch.tensor(batch[1], dtype=torch.float, requires_grad=False)

        optimizer.zero_grad()
        # print("train src1:", src[6])
        src_mask = make_src_mask(src, PADDING_INDEX)

        src = log_index_sequence_to_vec(src, INDEX_VEC_PATH)
        # print("train src2:", src[6])
        output = model(src, src_mask)
        # print("train trg : ", trg)
        # print("train output : ", output)

        # output.retain_grad()
        # print("train output :", output)
        # print("train trg :", trg)

        # MYEDIT
        # loss = criterion(1 - torch.exp(-output), trg)
        loss = criterion(output, trg)
        print("train loss : ", loss)
        # print("train loss", loss)
        # print(1 - torch.exp(-output))
        # loss.retain_grad()
        # loss = criterion(output, trg)

        loss.backward()
        # print(loss.grad)
        # print("grad: ", output.grad)

        torch.nn.utils.clip_grad_norm_(model.parameters(), clip)

        optimizer.step()

        epoch_loss += loss.item()

    return epoch_loss / len(iterator)


'''
INDEX_TO_TENSOR,NN_EMBEDDING这两个参数是无效参数，需要去掉
'''

def evaluate(model, INDEX_TO_TENSOR, NN_EMBEDDING, INDEX_VEC_PATH, iterator,
             criterion):
    max_output = 0.

    model.eval()
    # model.train()

    epoch_loss = 0

    with torch.no_grad():
        for i, batch in enumerate(iterator):
            src = torch.tensor(batch[0], dtype=torch.float, requires_grad=True)
            trg = torch.tensor(batch[1],
                               dtype=torch.float,
                               requires_grad=False)

            # print("eval src1:", src[6])

            # output = model(src, INDEX_TO_TENSOR)
            src_mask = make_src_mask(src, PADDING_INDEX)
            src = log_index_sequence_to_vec(src, INDEX_VEC_PATH)
            # print("eval src2:", src[6])
            output = model(src, src_mask)
            # print("eval output :", output)
            # print("eval trg :", trg)

            max_output = max(max_output, torch.max(output).item())

            # MYEDIT
            # loss = criterion(1 - torch.exp(-output), trg)
            loss = criterion(output, trg)
            print("eval loss: ", loss)

            epoch_loss += loss.item()

    return epoch_loss / len(iterator), max_output

'''
INDEX_TO_TENSOR,NN_EMBEDDING这两个参数是无效参数，需要去掉
'''

def evaluateEpsilon(model, INDEX_TO_TENSOR, NN_EMBEDDING, INDEX_VEC_PATH,
                    iterator, epsilon):
    model.eval()

    epoch_loss = 0

    with torch.no_grad():
        for i, batch in enumerate(iterator):
            src = torch.tensor(batch[0], dtype=torch.float, requires_grad=True)
            trg = torch.tensor(batch[1],
                               dtype=torch.float,
                               requires_grad=False)

            # output = model(src, INDEX_TO_TENSOR)
            src_mask = make_src_mask(src, PADDING_INDEX)
            src = log_index_sequence_to_vec(src, INDEX_VEC_PATH)
            output = model(src, src_mask)
            print("output", output)
            print(trg)
            # output = [batch size,1]
            zero = torch.zeros(output.shape)
            one = torch.ones(output.shape)

            loss = torch.sum(torch.where((output >= epsilon) == trg, one,
                                         zero))

            epoch_loss += loss

    return epoch_loss / len(iterator)


def epoch_time(start_time, end_time):
    elapsed_time = end_time - start_time
    elapsed_mins = int(elapsed_time / 60)
    elapsed_secs = int(elapsed_time - (elapsed_mins * 60))
    return elapsed_mins, elapsed_secs


def load_selfAttentive_model(N_HEADES,
                             INPUT_DIM,
                             HID_DIM,
                             OUTPUT_DIM,
                             N_ENCODERS,
                             FEEDFORWARD_DIM,
                             DROPOUT_RATE,
                             PAD_IDX,
                             DEVICE='cpu'):
    enc = Encoder(
        n_heads=N_HEADES,
        input_dim=INPUT_DIM,
        hid_dim=HID_DIM,  # 修改
        output_dim=OUTPUT_DIM,
        n_encoders=N_ENCODERS,
        feedforward_dim=FEEDFORWARD_DIM,
        dropout_rate=DROPOUT_RATE,
        device=DEVICE)

    # 获得模型
    model = SelfAttentive(enc, PAD_IDX)

    return model


def train_model(N_HEADES,
                INPUT_DIM,
                HID_DIM,
                OUTPUT_DIM,
                N_ENCODERS,
                FEEDFORWARD_DIM,
                DROPOUT_RATE,
                LEARNING_RATE,
                N_EPOCHS,
                CLIP,
                TRAIN_ITERATOR,
                VALID_ITERATOR,
                MODEL_OUTPUT_PATH,
                PAD_IDX,
                INDEX_TO_TENSOR,
                NN_EMBEDDING,
                INDEX_VEC_PATH,
                DEVICE='cpu'):
    # 获得模型
    model = load_selfAttentive_model(N_HEADES, INPUT_DIM, HID_DIM, OUTPUT_DIM,
                                     N_ENCODERS, FEEDFORWARD_DIM, DROPOUT_RATE,
                                     PAD_IDX, DEVICE)

    def count_parameters(model):
        return sum(p.numel() for p in model.parameters() if p.requires_grad)

    print(f'The model has {count_parameters(model):,} trainable parameters')

    # 随机初始化
    def initialize_weights(m):
        if hasattr(m, 'weight') and m.weight.dim() > 1:
            nn.init.xavier_uniform_(m.weight.data)

    model.apply(initialize_weights)

    # 损失函数和优化器的选择
    # MYEDIT
    # criterion = nn.BCELoss()
    criterion = My_loss()

    optimizer = torch.optim.Adam(model.parameters(), lr=LEARNING_RATE)

    # training
    best_valid_loss = float('inf')
    max_output = 0.

    for epoch in range(N_EPOCHS):

        start_time = time.time()

        train_loss = train(model, INDEX_TO_TENSOR, NN_EMBEDDING,
                           INDEX_VEC_PATH, TRAIN_ITERATOR, optimizer,
                           criterion, CLIP)

        valid_loss, temp_max_output = evaluate(model, INDEX_TO_TENSOR,
                                               NN_EMBEDDING, INDEX_VEC_PATH,
                                               TRAIN_ITERATOR, criterion)



        end_time = time.time()

        epoch_mins, epoch_secs = epoch_time(start_time, end_time)

        if valid_loss < best_valid_loss:
            best_valid_loss = valid_loss
            # 存储最大的epsilon值
            max_output = temp_max_output
            torch.save(model.state_dict(), MODEL_OUTPUT_PATH)

        print(f'Epoch: {epoch + 1:02} | Time: {epoch_mins}m {epoch_secs}s')
        print(
            f'\tTrain Loss: {train_loss:.3f}'
        )
        print(
            f'\t Val. Loss: {valid_loss:.3f}'
        )

    model.load_state_dict(torch.load(MODEL_OUTPUT_PATH))

    print("Begin to train epsilon")

    # train epsilon
    best_epsilon_loss = float('inf')
    best_epsilon = 0
    for epsilon in np.arange(0, max_output, max_output / 100):

        start_time = time.time()

        epsilon_loss = evaluateEpsilon(model, INDEX_TO_TENSOR, NN_EMBEDDING,
                                       INDEX_VEC_PATH, VALID_ITERATOR, epsilon)

        end_time = time.time()

        epoch_mins, epoch_secs = epoch_time(start_time, end_time)

        if epsilon_loss < best_epsilon_loss:
            best_epsilon_loss = epsilon_loss
            best_epsilon = epsilon

        print(
            f"epsilon: {epsilon} | Train loss: {epsilon_loss} | Best epsilon: {best_epsilon} | Best train loss: {best_epsilon_loss}"
        )

    print(f"End training epsilon...")
    print(f"Best epsilon: {best_epsilon} | Train loss: {best_epsilon_loss}")

    return best_epsilon


def test_model(N_HEADES,
               INPUT_DIM,
               HID_DIM,
               OUTPUT_DIM,
               N_ENCODERS,
               FEEDFORWARD_DIM,
               DROPOUT_RATE,
               MODEL_OUTPUT_PATH,
               TEST_ITERATOR,
               EPSILON,
               PAD_IDX,
               INDEX_TO_TENSOR,
               INDEX_VEC_PATH,
               NN_EMBEDDING,
               DEVICE='cpu'):
    # 获得模型
    model = load_selfAttentive_model(N_HEADES, INPUT_DIM, HID_DIM, OUTPUT_DIM,
                                     N_ENCODERS, FEEDFORWARD_DIM, DROPOUT_RATE,
                                     PAD_IDX, DEVICE)

    # 获得模型参数
    model.load_state_dict(torch.load(MODEL_OUTPUT_PATH))

    test_loss = evaluateEpsilon(model,
                                INDEX_TO_TENSOR,
                                NN_EMBEDDING,
                                INDEX_VEC_PATH=INDEX_VEC_PATH,
                                iterator=TEST_ITERATOR,
                                epsilon=EPSILON)

    print(f'| Test Loss: {test_loss:.3f} |')
