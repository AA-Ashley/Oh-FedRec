
from torch import nn
import torch
import torch.nn.functional as F
from myFedFrame.MyModules import *

LogSoftmax = nn.LogSoftmax(dim=1)
KL_Loss = nn.KLDivLoss(reduction='batchmean')
Softmax = nn.Softmax(dim=1)


class FedAvgOne(nn.Module):
    def __init__(self, parser, embedUserList, embedItemList, embedUser, embedItem):
        super().__init__()
        self.parser = parser
        self.userCount = parser.userCount
        self.itemCount = parser.itemCount
        self.embedSize = parser.embedSize
        # 获取自己的嵌入
        self.embedUser = torch.nn.Embedding(
            num_embeddings=self.userCount, embedding_dim=self.embedSize).cuda()
        self.embedItem = torch.nn.Embedding(
            num_embeddings=self.itemCount, embedding_dim=self.embedSize).cuda()

        self.embedUserList, self.embedItemList = [], []

        self.clientEmbedUser, self.clientEmbedItem = self.setClientEmbed(embedUserList, embedItemList)

        self.setServerEmbed(embedUser, embedItem)
        self.clientCount = len(self.embedUserList) + 1
        self.logistic = torch.nn.Sigmoid().cuda()
        self.sigmoid = nn.Sigmoid()

    def setServerEmbed(self, embedUser, embedItem):
        self.embedUser.weight.data = embedUser
        self.embedItem.weight.data = embedItem

    def setClientEmbed(self, embedUserList, embedItemList):
        embedUser, embedItem = embedUserList[0], embedItemList[0]
        for i in range(1, len(embedUserList)):
            embedUser += embedUserList[i]
            embedItem += embedItemList[i]
        embedUser = embedUser / len(embedItemList)
        embedItem = embedItem / len(embedItemList)
        return embedUser, embedItem

    # 总的一个forward
    def forward(self, users, seqs, posItems, negItems):
        embedUser = self.embedUser(users)
        embedPosItem = self.embedItem(posItems)
        embedNegItem = self.embedItem(negItems)
        posScores = (embedUser * embedPosItem).sum(dim=-1)
        negScores = (embedUser * embedNegItem).sum(dim=-1)
        return posScores, negScores

    # 总的一个forward
    def predict(self, users, seqs, items):
        embedUser = self.embedUser(users)
        embedItem = self.embedItem(items)
        scores = (embedUser * embedItem).sum(dim=-1)
        return scores

    def getLoss(self, users, seqs, posItems, negItems):
        posScore, negScore = self.forward(users, seqs, posItems, negItems)
        bprLoss = - (posScore - negScore).sigmoid().log().mean()
        localEmbed = torch.cat([self.embedUser.weight, self.embedItem.weight], dim=0)
        clientEmbed = torch.cat([self.clientEmbedUser, self.clientEmbedItem], dim=0)
        klLoss = KL_Loss(LogSoftmax(localEmbed), Softmax(clientEmbed))
        loss = 0.4 * bprLoss + 0.6 * klLoss
        return loss, 0

