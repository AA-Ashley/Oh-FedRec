import Constant
import utils
from Model.modelFactory import ModelFactory
from Trainer import Trainer
from Model.LightGCN import LightGCN
from Model.SASRec import SASRec
from Model.BPR import BPR
from Model.MF import MatrixFactorization


class Client:

    def __init__(self, parser):
        self.model = None
        self.trainer = None
        self.parser = parser
        self.initModelAndTrainer()
        pass

    def initModelAndTrainer(self):
        # 初始化模型
        self.model = ModelFactory.getModel(self.parser)
        # 初始化训练器
        self.trainer = Trainer(self.model, self.parser)
        pass

    def train(self):
        epochList, resultEpoch = self.trainer.train()
        figName = "./Result/the effect of {0} in {1}".format(self.parser.modelCode, self.parser.dataSetName)
        utils.drawFigures(figName, epochList, resultEpoch)

    def test(self, epoch):
        return self.trainer.test(epoch)

    def saveModel(self):
        self.trainer.saveModel()

    def loadModel(self):
        self.trainer.loadModel()

    def backward(self, loss):
        self.trainer.opt.zero_grad()
        loss.backward()
        self.trainer.opt.step()


if __name__ == '__main__':
    models = [Constant.GRU_CODE, Constant.BPR_CODE, Constant.MF_CODE, Constant.LightGCN_CODE]
    dataList = [Constant.JD_BUY, Constant.JD_CAR, Constant.JD_LIKE]
    client = utils.initClient(Constant.GRU_CODE, Constant.JD_BUY, dataType=Constant.JD_DATA)
    client.train()





