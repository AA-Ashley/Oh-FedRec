import torch
import os
torch.cuda.manual_seed(1024)
from myFedFrame.myTrainer import Trainer
from myFedFrame.MyModel0321 import MyModel
from myFedFrame.FedAvgOne import FedAvgOne
from myFedFrame.FedProxOne import FedProxOne
import Constant
from myFedFrame.Common import EmbedPath
from ParameterSetting import MyParser, BasicParser, DataParser
os.environ["CUDA_VISIBLE_DEVICES"] = "0"


if __name__ == '__main__':
    # parser = DataParser(Constant.TMALL_BUY, True)
    # parser.len = 5

    '''JD'''
    parser = DataParser(Constant.JD_LIKE, dataType=Constant.JD_DATA)
    parser.len = 5
    parser.setPath(Constant.JD_LIKE, dataType=Constant.JD_DATA)
    # 获取主参与方的嵌入
    serverEmbedUser, serverEmbedItem = EmbedPath(Constant.MF_CODE, Constant.JD_CAR, parser).getEmbed()
    # 获取各个参与方的嵌入
    embedUser1, embedItem1 = EmbedPath(Constant.MF_CODE, Constant.JD_LIKE, parser).getEmbed()
    embedUser2, embedItem2 = EmbedPath(Constant.MF_CODE, Constant.JD_BUY, parser).getEmbed()


    '''Tmall'''
    # parser = DataParser(Constant.TMALL_LIKE, dataType=Constant.TMALL_DATA)
    # parser.len = 5
    # parser.setPath(Constant.TMALL_LIKE, dataType=Constant.TMALL_DATA)
    # # 获取主参与方的嵌入
    # serverEmbedUser, serverEmbedItem = EmbedPath(Constant.LightGCN_CODE, Constant.TMALL_BUY, parser).getEmbed()
    # # 获取各个参与方的嵌入
    # embedUser1, embedItem1 = EmbedPath(Constant.MF_CODE, Constant.TMALL_LIKE, parser).getEmbed()
    # embedUser2, embedItem2 = EmbedPath(Constant.GRU_CODE, Constant.TMALL_CAR, parser).getEmbed()

    clientEmbedUsers, clientEmbedItems = [embedUser1, embedUser2], [embedItem1, embedItem2]
    # 设置恶意攻击者
    # for i in range(2):
    #     evilEmbedUser = torch.nn.Embedding(num_embeddings=Constant.JD_USERS, embedding_dim=32).cuda().weight.data * 100
    #     evilEmbedItem = torch.nn.Embedding(num_embeddings=Constant.JD_ITEMS, embedding_dim=32).cuda().weight.data * 100
    #     clientEmbedUsers.append(evilEmbedUser)
    #     clientEmbedItems.append(evilEmbedItem)
    # model = FedProxOne(parser, clientEmbedUsers, clientEmbedItems, serverEmbedUser, serverEmbedItem)
    # model = FedAvgOne(parser, clientEmbedUsers, clientEmbedItems, serverEmbedUser, serverEmbedItem)
    model = MyModel(parser, clientEmbedUsers, clientEmbedItems, serverEmbedUser, serverEmbedItem)
    print(model)
    trainer = Trainer(model, parser)
    trainer.train()
    pass










