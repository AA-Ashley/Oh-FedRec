from Model.NCL import NCL
from Model.NCF import NeuMF
from Model.FMLP import FMLPRecModel
from Model.BPR import BPR
from Model.LightGCN import LightGCN
from Model.Bert4Rec import BERT
from Model.SASRec import SASRec
from Model.MLP import MultiLayerPerceptron
from Model.MF import MatrixFactorization
from Model.GRU4Rec import GRU4Rec
import Constant


class ModelFactory:
    modelMap = {
        Constant.LightGCN_CODE: LightGCN,
        Constant.BPR_CODE: BPR,
        Constant.SASREC_CODE: SASRec,
        Constant.BERT_CODE: BERT,
        Constant.FMLPREC_CODE: FMLPRecModel,
        Constant.NCF_CODE: NeuMF,
        Constant.MLP_CODE: MultiLayerPerceptron,
        Constant.MF_CODE: MatrixFactorization,
        Constant.GRU_CODE: GRU4Rec
    }

    @staticmethod
    def getModel(parser):
        model = ModelFactory.modelMap[parser.modelCode]
        return model(parser)
