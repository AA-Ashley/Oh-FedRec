import argparse
import Constant
NEG_COUNT = 100


class BasicParser:
    def __init__(self):
        self.batchSize = 4096
        self.learningRate = 0.001
        self.embedSize = 32
        self.dropOut = 0.2
        self.Epochs = 10000
        self.weightDecay = 0.0
        self.l2Emb = 0.0
        self.negCount = 100
        self.device = "cuda:20"


class DataParser(BasicParser):
    def __init__(self, dataCode, dataType):
        super().__init__()
        if dataType == Constant.TMALL_DATA:
            self.userCount = Constant.TMALL_USERS
            self.itemCount = Constant.TMALL_ITEMS
            path = Constant.TMALL_PATH + dataCode
        elif dataType == Constant.JD_DATA:
            self.userCount = Constant.JD_USERS
            self.itemCount = Constant.JD_ITEMS
            path = Constant.JD_PATH + dataCode
        elif dataType == Constant.RR_DATA:
            self.userCount = Constant.RR_USERS
            self.itemCount = Constant.RR_ITEMS
            path = Constant.RR_PATH + dataCode
        self.trainerCode = Constant.CF_TRAINER_CODE
        self.dataSetName = dataCode
        # 加载数据集
        if self.trainerCode == Constant.CF_TRAINER_CODE:
            self.trainPath = path + "GraphTrain.csv"
            self.testPath = path + "GraphTest.csv"
        else:
            self.trainPath = path + "SequenceTrain.csv"
            self.testPath = path + "SequenceTest.csv"
        self.lapMatrix = path + "LapMatrix.npz"
        self.modelCode = "none"

    def setPath(self, dataCode, dataType=True):
        self.dataSetName = dataCode
        path = ''
        # 加载数据集
        if dataType == Constant.TMALL_DATA:
            path = Constant.TMALL_PATH + dataCode
        elif dataType == Constant.JD_DATA:
            path = Constant.JD_PATH + dataCode
        elif dataType == Constant.RR_DATA:
            path = Constant.RR_PATH + dataCode
        if self.trainerCode == Constant.CF_TRAINER_CODE:
            self.trainPath = path + "GraphTrain.csv"
            self.testPath = path + "GraphTest.csv"
        else:
            self.trainPath = path + "SequenceTrain.csv"
            self.testPath = path + "SequenceTest.csv"
        self.lapMatrix = path + "LapMatrix.npz"

class MyParser(DataParser):
    def __init__(self, modelCode, dataCode, dataType=True):
        super().__init__(dataCode, dataType)
        self.modelCode = modelCode

class MySequenceParser(MyParser):
    def __init__(self, modelCode, dataCode, dataType=True):
        super().__init__(modelCode, dataCode, dataType)
        self.len = Constant.SEQ_LEN
        self.trainerCode = Constant.SEQ_TRAINER_CODE
        self.setPath(dataCode, dataType)

class BPRParser(MyParser):
    def __init__(self, dataCode, dataType=True):
        super().__init__(Constant.BPR_CODE, dataCode, dataType)


class MFParser(MyParser):
    def __init__(self, dataCode, dataType=True):
        super().__init__(Constant.MF_CODE, dataCode, dataType)


class MLPParser(MyParser):
    def __init__(self, dataCode, dataType=True):
        super().__init__(Constant.MLP_CODE, dataCode, dataType)
        self.layers = [32, 32, 32]


class NCLParser(MyParser):
    def __init__(self, dataCode, dataType=True):
        super().__init__(Constant.NCL_CODE, dataCode, dataType)
        self.layer = 2
        self.sslTemp = 0.1
        self.protoReg = 1e-7
        self.sslReg = 1e-6
        self.alpha = 1
        self.hyperLayers = 1
        self.numClusters = 1000


class LCGNParser(MyParser):
    def __init__(self, dataCode, dataType=True):
        super().__init__(Constant.LightGCN_CODE, dataCode, dataType)
        self.layer = 2


class NGCFParser(MyParser):
    def __init__(self, dataCode, dataType=True):
        super().__init__(Constant.NGCF_CODE, dataCode, dataType)
        self.layer = 2


class SASRecParser(MySequenceParser):
    def __init__(self, dataCode, dataType=True):
        super().__init__(Constant.SASREC_CODE, dataCode, dataType)
        self.numBlocks = 2
        self.numHeads = 1
        self.dropOut = 0.2


class GRU4RecParser(MySequenceParser):
    def __init__(self, dataCode, dataType=True):
        super().__init__(Constant.GRU_CODE, dataCode, dataType)


class NCFParser(MyParser):
    def __init__(self, dataCode, dataType=True):
        super().__init__(Constant.NCF_CODE, dataCode, dataType)
        self.mlpLayers = [32, 64, 32]


class Bert4RecParser(MySequenceParser):
    def __init__(self, dataCode, dataType=True):
        super().__init__(Constant.SASREC_CODE, dataCode, dataType)
        self.numBlocks = 2
        self.numHeads = 1
        self.maskProb = 0.1


class FMLPRecParser(MySequenceParser):
    def __init__(self, dataCode, dataType=True):
        super().__init__(Constant.FMLPREC_CODE, dataCode, dataType)
        self.numBlocks = 2
        self.numHeads = 1
        self.maskProb = 0.1
        self.attentionDropOut = 0.1
        self.hiddenDropOut = 0.1
        self.hiddenAct = "gelu"
        self.noFilters = True
        self.initializerRange = 0.02


class ParserFactory:
    parserMap = {
        Constant.LightGCN_CODE: LCGNParser,
        Constant.NGCF_CODE: NGCFParser,
        Constant.BPR_CODE: BPRParser,
        Constant.SASREC_CODE: SASRecParser,
        Constant.BERT_CODE: Bert4RecParser,
        Constant.FMLPREC_CODE: FMLPRecParser,
        Constant.MLP_CODE: MLPParser,
        Constant.MF_CODE: MFParser,
        Constant.GRU_CODE:GRU4RecParser,
        ###########################
        Constant.DIFF_CODE:DiffParser
        ###########################
    }

    @staticmethod
    def getParser(modelCode, dataCode, dataType):
        parserClass = ParserFactory.parserMap[modelCode]
        return parserClass(dataCode, dataType)




