import utils
import torch
from torch.utils.data import DataLoader
from DataProcess.DealDataset import MyDataset
from torch.autograd import Variable
from torch import optim


class Trainer:
    def __init__(self, model, para):
        self.model = model
        self.max_epoch = para.Epochs
        self.opt = optim.Adam(filter(lambda p: p.requires_grad, self.model.parameters()), lr=para.learningRate)  #过滤出模型中需要梯度更新的参数
        self.para = para
        self.trainData, self.testData = MyDataset(para).getDataset()
        self.trainLoader = DataLoader(self.trainData, batch_size=para.batchSize, shuffle=True)
        self.testLoader = DataLoader(self.testData, batch_size=para.batchSize, shuffle=True)
        # 设置在一个epoch中，当前训练数据为哪个批次
        self.step, self.iterator = 0, iter(self.trainLoader)
        self.bestResult = [0, 0, 0]
        self.bestResultEpoch, self.downEpoch = 0, 0

    def loadStudentEmbed(self, embedUser, embedItem):
        self.model.setStudentEmbed(embedUser, embedItem)
        pass

    def loadTeacherEmbed(self, embedUser, embedItem):
        self.model.setTeacherEmbed(embedUser, embedItem)
        pass

    def train(self):
        epochList, resultList = [], []
        print("还没训练的测试{0}".format(self.test(0)))
        for epoch in range(0, self.max_epoch):
            self.trainOneEpoch()
            print("epoch={0}已经完成".format(epoch))
            if epoch % 1 == 0:
                result = self.test(epoch)
                epochList.append(epoch)
                resultList.append(result)
            if self.downEpoch >= 100:
                break
        return epochList, resultList

    def trainOneEpoch(self):
        finalLoss = 0
        for i, batch in enumerate(self.trainLoader):
            user, sequences, posItem, negItems = self.data2Cuda(batch)
            self.opt.zero_grad()
            loss, regLoss = self.model.getLoss(user, sequences, posItem, negItems)
            loss.backward()
            self.opt.step()
            finalLoss += loss
        finalLoss = finalLoss / len(self.trainLoader)
        print("loss = {0}".format(finalLoss))
        return finalLoss

    def data2Cuda(self, batch):
        user, sequences, posItem, negItems = batch
        if not isinstance(sequences, list):
            sequences = Variable(sequences).cuda()
        user, posItem, negItems = Variable(user).cuda(), Variable(posItem).cuda(), Variable(negItems).cuda()
        return user, sequences, posItem, negItems

    def test(self, epoch=0):
        rank = None
        for step, data in enumerate(self.testLoader):
            user, sequences, posItem, negItems = self.data2Cuda(data)
            epochRank = torch.tensor([1] * len(user)).cuda()  #当前批次用户的推荐结果排名
            posScore = self.model.predict(user, sequences, posItem)  #正样本预测分数：用户与项目和
            for i in range(self.para.negCount):
                negScore = self.model.predict(user, sequences, negItems[:, i])  #负样本预测分数：用户与项目和
                res = ((posScore - negScore) < 0.00001).cuda()
                epochRank = epochRank + res
            # 将这一轮的结果写入到rank中
            if rank is None:
                rank = epochRank
            else:
                rank = torch.cat([rank, epochRank], dim=0)
        MRR = round(utils.calMRR(rank), 4)
        HIT5 = round(utils.calHit(rank, 5), 4)
        NDCG5 = round(utils.calNDCG(rank, 5), 4)
        print("当前回合={0}\t MRR = {1}\t Hit@5={2}\t NDCG@5={3}\t".format(epoch,MRR,HIT5,NDCG5))
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
        else:
            self.downEpoch += 2
        print(text)
        return MRR + HIT5 + NDCG5

    def backward(self, loss):
        """
        根据损失函数调整
        :param loss: 损失函数
        :return:
        """
        self.opt.zero_grad()
        loss.backward()
        self.opt.step()



