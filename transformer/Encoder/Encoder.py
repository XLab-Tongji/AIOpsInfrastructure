# -*- coding: UTF-8 -*-
"""
@Project : self-attentive
@File    : Encoder.py
@Author : mount_potato
@Date    : 2021/10/22 9:52 上午
"""
import time

import torch
import torch.nn as nn

import math


class Encoder(nn.Module):

    def __init__(self,
                 n_heads,
                 input_dim,
                 hid_dim,
                 output_dim,
                 n_encoders,
                 feedforward_dim,
                 dropout_rate,
                 device):
        """

        :param n_heads: 注意头的数量，允许并行计算，这个可调节，但要求hid_dim // n_heads
        :param input_dim: [EMBEDDING]的维度，在fasttext词向量下为300
        :param hid_dim: input_dim经过全连接层输出的维度，设为512
        :param output_dim: 输出维度，这里设为300
        :param n_encoders: EncoderLayer的数量，原论文为3
        :param feedforward_dim: 前馈网络中全连接层维度，比较大，原论文为2048
        :param dropout_rate:
        :param device:
        """

        self.device = device

        super().__init__()

        # self.tok_embedding = nn.Linear(vocab_size, embedding_dim)
        self.hid_dim = hid_dim

        self.input_linear = nn.Linear(input_dim, hid_dim)
        # self.pos_encoding = PositionalEncoding(hid_dim, dropout_rate)
        self.pos_embedding = nn.Embedding(5000, hid_dim)

        self.layers = nn.ModuleList([EncoderLayer(hid_dim,
                                                  n_heads,
                                                  feedforward_dim,
                                                  dropout_rate,
                                                  device)
                                     for _ in range(n_encoders)])
        self.dropout = nn.Dropout(dropout_rate)
        self.scale = torch.sqrt(torch.FloatTensor([hid_dim])).to(device)

        self.output_linear = nn.Linear(hid_dim, output_dim)

    def forward(self, src, src_mask):
        """

        Args:
            src: [batch_size,seq len]s
            nn_embedding: 预训练embedding,存储fasttext词向量信息
            src_mask: 先None吧，似乎论文里是不需要mask的
        Returns:
             一个[batch_size,词向量个数,EMBEDDING维度]的Tensor

        """

        # src=[batch_size, seq len,embedding_dim]

        batch_size = src.shape[0]
        seq_len = src.shape[1]

        src = self.input_linear(src)

        pos = torch.arange(0, seq_len).unsqueeze(
            0).repeat(batch_size, 1).to(self.device)

        # print("word_emb:",input_embedding_vec)

        src = src * self.scale

        # input_embedding_vec=[batch_size, log message len, embedding_length]

        # 位置编码，与position embedding向量相加后做dropout
        # print("位置编码:", self.pos_embedding(pos))

        src = self.dropout(src + self.pos_embedding(pos))
        # src = self.pos_encoding(src)

        # print("位置编码后结果：", src)

        # 经过n_encoders个Encoder编码
        for single_encoder in self.layers:
            src = single_encoder(src, src_mask)

        src = self.output_linear(src)

        # print("经过3个EncoderLayer后结果：", src)

        return src


class MultiHeadAttentionLayer(nn.Module):
    def __init__(self, hid_dim, n_heads, dropout, device):
        super().__init__()

        # 向量k或v的维度需要整除self-attention-head的数量
        assert hid_dim % n_heads == 0

        self.hid_dim = hid_dim
        self.n_heads = n_heads
        self.head_dim = hid_dim // n_heads

        # d_model指向量k或v的维度
        self.fc_q = nn.Linear(hid_dim, hid_dim)
        self.fc_k = nn.Linear(hid_dim, hid_dim)
        self.fc_v = nn.Linear(hid_dim, hid_dim)
        self.fc_o = nn.Linear(hid_dim, hid_dim)

        self.dropout = nn.Dropout(dropout)

        self.scale = torch.sqrt(torch.FloatTensor([self.head_dim])).to(device)

    def forward(self, query, key, value, mask=None):
        batch_size = query.shape[0]

        Q = self.fc_q(query)
        K = self.fc_k(key)
        V = self.fc_v(value)

        # 将Q,K,V原本的一维向量维度d_model,延拓为多头，即[n_heads,head_dim]
        # 论文中选取d_model=512,n_heads=8
        # 则Q[batch_size,len,512]被转化为[batch_size,len,8,64]
        # 再用permute将维度调换，转化为[batch_size,8,len,64]
        Q = Q.view(batch_size, -1, self.n_heads,
                   self.head_dim).permute(0, 2, 1, 3)
        K = K.view(batch_size, -1, self.n_heads,
                   self.head_dim).permute(0, 2, 1, 3)
        V = V.view(batch_size, -1, self.n_heads,
                   self.head_dim).permute(0, 2, 1, 3)

        # Q,K,V当前维度[batch size, n heads, len q\k\v, head dim]
        # 计算softmax(Q,K,V)
        # K矩阵转置n_heads，head_dim维度
        energy = torch.matmul(Q, K.permute(0, 1, 3, 2)) / self.scale
        # energy:[batch size, n heads, len q, len k]

        if mask is not None:
            energy = energy.masked_fill(mask == 0, -1e10)

        # softmax函数,dim=-1?
        attention = torch.softmax(energy, dim=-1)

        # [batch size, n heads, len q, len k]

        x = torch.matmul(self.dropout(attention), V)
        # softmax(Q,K,V)最终输出的x维度: [batch size, n heads, query len, head dim]

        # contiguous似乎是某种优化存储方式的函数?
        x = x.permute(0, 2, 1, 3).contiguous()
        # [batch size, query len,n heads, head dim]

        # 后两个维度合并为d_model
        x = x.view(batch_size, -1, self.hid_dim)
        # [batch size, query len,d_model]

        x = self.fc_o(x)

        # print(x)
        # exit(0)

        return x, attention


# 前馈层 Feed Forward Neural Network
class PositionwiseFeedforwardLayer(nn.Module):
    def __init__(self, hid_dim, dim_feedforward, dropout):
        super().__init__()
        # 全连接层1，从d_model到dim_feedforward，后者比较大
        self.fc1 = nn.Linear(hid_dim, dim_feedforward)
        # 全连接层2，从d_model转换会dim_feedforward
        self.fc2 = nn.Linear(dim_feedforward, hid_dim)

        # dropout层
        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        # 对输入变量z1进行全连接得到r1
        # 公式：FFN(x)=max(0,xW1+b1)W2+b2,中间对max(0,xW1+b1)进行dropout防止过拟合
        x = self.dropout(torch.relu(self.fc1(x)))
        x = self.fc2(x)

        return x


class EncoderLayer(nn.Module):
    def __init__(self, hid_dim, n_heads, dim_feedforward, dropout,
                 device):
        super().__init__()

        self.dropout = nn.Dropout(dropout)

        self.multi_head_layer_norm = nn.LayerNorm(hid_dim)  # 用于多头注意后的LayerNorm
        self.feedforward_layer_norm = nn.LayerNorm(
            hid_dim)  # 用于前馈神经网络后的LayerNorm

        self.self_attention = MultiHeadAttentionLayer(hid_dim, n_heads,
                                                      dropout, device)
        self.position_wise_feedforward = PositionwiseFeedforwardLayer(
            hid_dim, dim_feedforward, dropout)

    def forward(self, word_vec_matrix, src_mask):
        # word_vec_matrix: [batch size, len word vec, embedding_dim]
        # src_mask[batch_size, 1, 1, len word vec]
        _wvm, _ = self.self_attention(word_vec_matrix, word_vec_matrix,
                                      word_vec_matrix, src_mask)

        # print("经过Multi-head attention后的中间值：", _wvm)

        word_vec_matrix = self.multi_head_layer_norm(word_vec_matrix +
                                                     self.dropout(_wvm))
        # 维度不变[batch size, len word vec, d_model]
        # print("src与该中间值Add&Norm后的结果src", word_vec_matrix)

        _wvm = self.position_wise_feedforward(word_vec_matrix)

        # print("src经过前馈网络后的中间值", _wvm)

        word_vec_matrix = self.feedforward_layer_norm(word_vec_matrix +
                                                      self.dropout(_wvm))

        # print("src与中间值Add&Norm的结果", word_vec_matrix)

        return word_vec_matrix


# class PositionalEncoding(nn.Module):
#     def __init__(self, embedding_dim, dropout=0.1, max_len=5000):
#         super(PositionalEncoding, self).__init__()
#         self.dropout = nn.Dropout(p=dropout)
#         pe = torch.zeros(max_len, embedding_dim)
#         position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
#         div_term = torch.exp(
#             torch.arange(0, embedding_dim, 2).float() *
#             (-math.log(10000.0) / embedding_dim))
#         pe[:, 0::2] = torch.sin(position * div_term)
#         pe[:, 1::2] = torch.cos(position * div_term)
#         pe = pe.unsqueeze(0).transpose(0, 1)
#         self.register_buffer('pe', pe)
#
#     def forward(self, x):
#         x = x + self.pe[:x.size(0), :]
#         return self.dropout(x)
