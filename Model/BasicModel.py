import torch
from abc import abstractmethod, ABC
import torch.nn.functional as F
from torch import nn


class BasicModel(torch.nn.Module):
    def __init__(self, parser):
        super().__init__()
        self.userCount = parser.userCount
        self.itemCount = parser.itemCount
        self.dropOut = parser.dropOut
        self.embedSize = parser.embedSize
        self.embedUser = torch.nn.Embedding(
            num_embeddings=self.userCount, embedding_dim=self.embedSize).to(parser.device)
        self.embedItem = torch.nn.Embedding(
            num_embeddings=self.itemCount, embedding_dim=self.embedSize).to(parser.device)
        nn.init.normal_(self.embedUser.weight, std=0.1)
        nn.init.normal_(self.embedItem.weight, std=0.1)
        # 存储经过计算后的embedding
        self.finalEmbedUser = torch.nn.Embedding(
            num_embeddings=self.userCount, embedding_dim=self.embedSize).to(parser.device)
        self.finalEmbedItem = torch.nn.Embedding(
            num_embeddings=self.itemCount, embedding_dim=self.embedSize).to(parser.device)

    @abstractmethod
    def forward(self, users, seqs, posItems, negItems):
        pass

    @abstractmethod
    def getLoss(self, users, seqs, posItems, negItems):
        pass

    @abstractmethod
    def computer(self, args):
        pass

    @abstractmethod
    def predict(self, users, seqs, items):
        pass

    def getFinalEmbed(self):
        """
        获取模型经过一轮迭代后的嵌入表示
        :return: 2个Embedding类型， self.finalEmbedUser, self.finalEmbedItem
        """
        # finalEmbedUser = F.normalize(self.finalEmbedUser.weight.data, p=2, dim=1)
        # finalEmbedItem = F.normalize(self.finalEmbedItem.weight.data, p=2, dim=1)
        finalEmbedUser = self.finalEmbedUser.weight.data
        finalEmbedItem = self.finalEmbedItem.weight.data
        return finalEmbedUser, finalEmbedItem






