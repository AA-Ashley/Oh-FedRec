from torch import nn
import torch
import torch.nn.functional as F
from myFedFrame.MyModules import *


class MyModel(nn.Module):
    def __init__(self, parser, embedUserList, embedItemList, embedUser, embedItem):
        super().__init__()
        self.parser = parser
        self.userCount = parser.userCount
        self.itemCount = parser.itemCount
        self.embedSize = parser.embedSize
        # 获取自己的嵌入 嵌入矩阵
        self.embedUser = torch.nn.Embedding(
            num_embeddings=self.userCount, embedding_dim=self.embedSize).cuda()
        self.embedItem = torch.nn.Embedding(
            num_embeddings=self.itemCount, embedding_dim=self.embedSize).cuda()

        self.embedUserList, self.embedItemList = [], []

        self.setServerEmbed(embedUser, embedItem)
        self.setClientEmbed(embedUserList, embedItemList)
        self.clientCount = len(self.embedUserList) + 1
        self.logistic = torch.nn.Sigmoid().cuda()
        self.layers = [self.embedSize * self.clientCount, self.embedSize]
        self.DNN = MLP(layers=self.layers, parser=parser)

        # self.IEU_G = IEU(parser.len, self.embedSize * 2, weight_type="bit",
        #                  bit_layers=1, outputSize=self.embedSize, mlpLayer=128).cuda()
        #
        # # IEU_W computes bit-level or vector-level weights.
        # self.IEU_W = IEU(parser.len, self.embedSize * 2, weight_type="vector",
        #                  bit_layers=1, outputSize=self.embedSize, mlpLayer=128).cuda()

        # # 每一个客户端都有一个decoder
        self.decoderList = []
        # self.Decoder = MLP(layers=[self.embedSize, self.embedSize * 2, self.embedSize * 2, self.embedSize])
        for i in range(self.clientCount - 1):
            self.decoderList.append(MLP(layers=[self.embedSize, self.embedSize]))

        #通常用于特征抽取和转换
        self.agg = Aggregation(layers=[self.embedSize * self.clientCount,  self.embedSize], parser=parser)

        #通常用于加权或调整某些输入特征的重要性，以用于后续的任务，例如注意力机制
        self.weightGate = nn.Sequential(
            nn.Linear(self.embedSize * self.clientCount, self.embedSize * self.clientCount * 2),
            nn.Linear(self.embedSize * self.clientCount * 2, self.embedSize * self.clientCount),
            nn.Linear(self.embedSize * self.clientCount, self.clientCount),
            # nn.Linear(self.embedSize, 3),
            # nn.Linear(self.embedSize , 2),
            nn.Softmax(dim=1)
        ).cuda()

        self.sigmoid = nn.Sigmoid()

    def setServerEmbed(self, embedUser, embedItem):
        self.embedUser.weight.data = embedUser
        self.embedItem.weight.data = embedItem

    def setClientEmbed(self, embedUserList, embedItemList):
        #将离散的用户和物品特征映射到连续的嵌入空间中
        for i in range(len(embedUserList)):
            if self.isNormalEmbedUser(embedUserList[i]):
                embedUser = torch.nn.Embedding(num_embeddings=self.userCount, embedding_dim=self.embedSize).cuda()
                embedUser.weight.data = embedUserList[i]
                self.embedUserList.append(embedUser)
        for i in range(len(embedItemList)):
            if self.isNormalEmbedItem(embedItemList[i]):
                embedItem = torch.nn.Embedding(num_embeddings=self.itemCount, embedding_dim=self.embedSize).cuda()
                embedItem.weight.data = embedItemList[i]
                self.embedItemList.append(embedItem)

    def isNormalEmbedUser(self, embedUser):
        proximalTerm = (embedUser.sum(axis=-1) - self.embedUser.weight.sum(axis=-1)).norm(2)
        proximalTerm /= self.userCount
        if proximalTerm < 0.1:
            return True
        return False

    def isNormalEmbedItem(self, embedItem):
        proximalTerm = (embedItem.sum(axis=-1) - self.embedItem.weight.data.sum(axis=-1)).norm(2)
        proximalTerm /= self.itemCount
        if proximalTerm < 0.1:
            return True
        return False

    # 总的一个forward
    def forward(self, users, seqs, posItems, negItems):
        allEmbedUser, allEmbedItem = self.computer(seqs)
        embedUser = allEmbedUser[users]
        embedPosItem = allEmbedItem[posItems]
        embedNegItem = allEmbedItem[negItems]
        posScores = (embedUser * embedPosItem).sum(dim=-1)
        negScores = (embedUser * embedNegItem).sum(dim=-1)
        return posScores, negScores
        # posScores = self.predict(users, seqs, posItems)
        # negScores = self.predict(users, seqs, negItems)
        # return posScores, negScores

    # 总的一个forward
    def predict(self, users, seqs, items):
        # allEmbedUser, allEmbedItem= self.computer(seqs)
        allEmbedUser, allEmbedItem = self.computer(seqs)
        embedUser = allEmbedUser[users]
        embedItem = allEmbedItem[items]
        #TODO：为什么这个是预测结果
        scores = (embedUser * embedItem).sum(dim=-1)
        # scores = self.logistic(scores)
        # scores = self.sigmoid(scores)
        return scores

    def computerDNN(self):
        embedUserList = [self.embedUser.weight]
        for e in self.embedUserList:  #收集服务器与客户端的用户嵌入权重
            embedUserList.append(e.weight)
        embedUser = torch.cat(embedUserList, dim=-1)
        embedItemList = [self.embedItem.weight]
        for e in self.embedItemList:
            embedItemList.append(e.weight)
        embedItem = torch.cat(embedItemList, dim=-1)
        embedUser = self.DNN(embedUser)
        embedItem = self.DNN(embedItem)
        return embedUser, embedItem

    def computerDecoder(self, clientIndex, embedUser, embedItem):
        decoder = self.decoderList[clientIndex]
        embedUser = decoder(embedUser)
        embedItem = decoder(embedItem)
        return embedUser, embedItem

    def computer(self, seq):
        embedUserDnn, embedItemDnn = self.computerDNN()
        embedUserList, embedItemList = [embedUserDnn], [embedItemDnn]  #0为dnn，1、2为两个客户端的decoder
        for i in range(self.clientCount - 1):
            # dec
            embedUserDec, embedItemDec = self.computerDecoder(i, self.embedUserList[i].weight, self.embedItemList[i].weight)
            embedUserList.append(embedUserDec)
            embedItemList.append(embedItemDec)
            # embedUserList.append(self.embedUserList[i].weight)
            # embedItemList.append(self.embedItemList[i].weight)
        # embedUserDec, embedItemDec = self.embedUserB.weight, self.embedItemB.weight
        allEmbedUser = torch.cat(embedUserList, dim=1)
        allEmbedItem = torch.cat(embedItemList, dim=1)
        aggEmbedUser, aggEmbedItem = self.agg(allEmbedUser), self.agg(allEmbedItem)  #进行变换和聚合
        weightUser, weightItem = self.weightGate(allEmbedUser), self.weightGate(allEmbedItem)
        weightUsers = weightUser.chunk(self.clientCount, dim=1)  #将权重信息切分
        weightItems = weightItem.chunk(self.clientCount, dim=1)
        #TODO：对应原文公式，但还没理清楚
        embedUser, embedItem = embedUserList[0] * weightUsers[0], embedItemList[0] * weightItems[0]
        for i in range(1, len(weightUsers)):
            embedUser += embedUserList[i] * weightUsers[i]
            embedItem += embedItemList[i] * weightItems[i]
        embedUser = 0.5 * aggEmbedUser + 0.5 * embedUser
        embedItem = 0.5 * aggEmbedItem + 0.5 * embedItem
        return embedUser, embedItem
        # return aggEmbedUser, aggEmbedItem

    def getLoss(self, users, seqs, posItems, negItems):
        posScore, negScore = self.forward(users, seqs, posItems, negItems)
        loss = - (posScore - negScore).sigmoid().log().mean()
        return loss, 0

