import math
import os

import Constant
import utils
import random
import numpy as np
import torch
from torch.utils.data import DataLoader
from DataProcess.DealDataset import MyDataset
from torch.autograd import Variable
from torch import optim
import pandas as pd


#随机种子
random_seed = 1
torch.manual_seed(random_seed)  #确保在同一个种子值下，每次运行代码时生成的随机数都是一样的
torch.cuda.manual_seed(random_seed)  #cuda
np.random.seed(random_seed)  #numpy
random.seed(random_seed)  #random transforms
torch.backends.cudnn.deterministic = True  #cudnn
def worker_init_fn(worker_id):
    np.random.seed(random_seed + worker_id)
def seed_worker(worker_id):
    worker_seed = torch.initial_seed() % 2**32
    np.random.seed(worker_seed)

class Trainer:
    def __init__(self, model, para):
        self.model = model
        self.max_epoch = para.Epochs
        self.opt = optim.Adam(filter(lambda p: p.requires_grad, self.model.parameters()), lr=para.learningRate)
        self.para = para
        self.trainData, self.testData = MyDataset(para).getDataset()
        self.trainLoader = DataLoader(self.trainData, batch_size=para.batchSize, shuffle=True)
        self.testLoader = DataLoader(self.testData, batch_size=para.batchSize, shuffle=True)
        # 设置在一个epoch中，当前训练数据为哪个批次
        self.step, self.iterator = 0, iter(self.trainLoader)
        self.bestResult = [0, 0, 0]
        self.bestResultEpoch, self.downEpoch = 0, 0

    def train(self):
        epochList, resultList = [], []
        for epoch in range(1, self.max_epoch):
            self.trainOneEpoch()
            print("epoch={0}已经完成".format(epoch))
            if epoch % 2 == 0:
                result = self.test(epoch)
                epochList.append(epoch)
                resultList.append(result)
            if self.downEpoch >= 100:
                break
        return epochList, resultList

    def trainOneEpoch(self):
        finalLoss = 0
        all_zero = True
        for i, batch in enumerate(self.trainLoader):
            user, sequences, posItem, negItems = self.data2Cuda(batch)
            loss, regLoss = self.model.getLoss(user, sequences, posItem, negItems)
            # loss = losses["loss"].mean()
            loss.backward()
            self.opt.step()
            finalLoss += loss
        finalLoss = finalLoss / len(self.trainLoader)
        print("loss = {0}".format(finalLoss))
        return finalLoss

    def data2Cuda(self, batch):
        user, sequences, posItem, negItems = batch
        device = self.para.device
        if not isinstance(sequences, list):
            sequences = Variable(sequences).to(device)
        user, posItem, negItems = Variable(user).to(device), Variable(posItem).to(device), Variable(negItems).to(device)
        return user, sequences, posItem, negItems

    def trainOneBatch(self):
        """
        训练一个batchSize大小的数据
        :return: 损失函数
        """
        try:
            # 从迭代器中获取数据，并且将其转化成cuda模式
            user, sequences, posItem, negItems = self.data2Cuda(self.iterator.next())
            loss, regLoss = self.model.getLoss(user, sequences, posItem, negItems)
            regLoss = regLoss * self.para.weightDecay
            loss = loss + regLoss
            self.step += 1
            return loss
        except StopIteration as e:
            return None

    def test(self, epoch=0):
        rank = None
        for step, data in enumerate(self.testLoader):
            user, sequences, posItem, negItems = self.data2Cuda(data)
            epochRank = torch.tensor([1] * len(user)).to(self.para.device)  #大小与用户数量相同的张量，存储每个用户的排名
            posScore = self.model.predict(user, sequences, posItem)  #预测用户与正样本的分数
            for i in range(self.para.negCount):
                negScore = self.model.predict(user, sequences, negItems[:, i])  #计算用户与当前负样本的分数
                res = ((posScore - negScore) <= 0).to(self.para.device)  #判断正样本分数是否小于等于负样本分数
                epochRank = epochRank + res  #更新用户排名
            # 将这一轮的结果写入到rank中
            if rank is None:
                rank = epochRank
            else:
                rank = torch.cat([rank, epochRank], dim=0)
        MRR = round(utils.calMRR(rank), 4)
        HIT5 = round(utils.calHit(rank, 5), 4)
        NDCG5 = round(utils.calNDCG(rank, 5), 4)
        text = "bestEpoch={0}\t MRR = {1}\t Hit@5={2}\t NDCG@5={3}\t dataset={4}\t model={5}\t downEpoch={6}".format(
            self.bestResultEpoch, self.bestResult[0], self.bestResult[1], self.bestResult[2], self.para.testPath,
            self.para.modelCode, self.downEpoch)

        if sum(self.bestResult) - MRR - HIT5 - NDCG5 < 0:
            self.bestResult = [MRR, HIT5, NDCG5]
            self.bestResultEpoch = epoch
            self.downEpoch = 0
            text = "bestEpoch={0}\t MRR = {1}\t Hit@5={2}\t NDCG@5={3}\t dataset={4}\t model={5}\t downEpoch={6}".format(
                self.bestResultEpoch, self.bestResult[0], self.bestResult[1], self.bestResult[2], self.para.testPath,
                self.para.modelCode, self.downEpoch)
            utils.writeLog(text)
            self.saveModel()
        else:
            self.downEpoch += 2
        print(text)
        return MRR + HIT5 + NDCG5

    def sendEmbedding(self):
        """
        训练器发送模型训练完的embedding
        :return:
        """
        return self.model.getFinalEmbed()

    def backward(self, loss):
        """
        根据损失函数调整
        :param loss: 损失函数
        :return:
        """
        self.opt.zero_grad()
        loss.backward()
        self.opt.step()

    def initTrainLoader(self, batchSize):
        self.trainLoader = DataLoader(self.trainData, batch_size=batchSize, shuffle=True)
        self.step, self.iterator = 0, iter(self.trainLoader)

    def saveModel(self):
        path = "./Save_hxh/{0}/{1}/model.pth".format(self.para.modelCode, self.para.dataSetName)
        userEmbed, itemEmbed = self.model.embedUser.weight.data, self.model.embedItem.weight.data
        if not os.path.exists(os.path.dirname(path)):
            os.makedirs(os.path.dirname(path))
        torch.save(userEmbed, "./Save_hxh/{0}/{1}/userEmbed.pth".format(self.para.modelCode, self.para.dataSetName))
        torch.save(itemEmbed, "./Save_hxh/{0}/{1}/itemEmbed.pth".format(self.para.modelCode, self.para.dataSetName))
        torch.save(self.model, path)

    def loadModel(self):
        path = "./Save/{0}-{1}.pth".format(self.para.modelCode, self.para.dataSetName)
        self.model = torch.load(path, map_location=torch.device('cuda'))
        self.model.eval()
