import torch
import scipy.sparse as sp
from scipy.linalg import inv
import numpy as np
import matplotlib.pyplot as plt
##############################
# from DiffClient import Client
##############################
from Client import Client
from ParameterSetting import ParserFactory


def calMRR(rank):
    """
    计算Mean Reciprocal Rank(MRR)=(1 / Q) * Sum(1 / rank_i) i属于1到Q Q为用户总数
    :param rank:用户行为的物品得分在101个物品得分中的排名
    :return:MRR值
    """
    mrr = 0
    for v in rank:
        mrr += 1.0 / v
    mrr = mrr / len(rank)
    return mrr.cpu().item()


def calHit(rank, k):
    """
    计算Hit Ratio
    :param rank:用户行为的物品得分在101个物品得分中的排名
    :param k:HR@K中的K
    :return:HR@K的值
    """
    hit = 0
    for v in rank:
        if v <= k:
            hit += 1
    hit = hit / len(rank)
    return hit


def calNDCG(rank, k):
    """
    计算NDCG@K
    :param rank: 用户行为的物品得分在101个物品得分中的排名
    :param k: NDCG@K中的K
    :return: NDCG@K的值
    """
    ndcg = 0
    for v in rank:
        if v > k:
            continue
        else:
            ndcg += torch.log(torch.Tensor([2])) / torch.log(torch.Tensor([v + 1]))
    ndcg = ndcg / len(rank)
    return ndcg[0].cpu().item()


def printer(s):
    print("——————————————————————————————————————{0}——————————————————————————————————————".format(s))


def calGraph(adjacency: sp.coo_matrix):
    """
    生成归一化矩阵
    :param adjacency: 邻接矩阵
    :return: 归一化矩阵D-1/2 A D-1/2
    """
    adjacency = adjacency.tolil()
    adjacencyUU = sp.identity(adjacency.shape[0])
    adjacencyII = sp.identity(adjacency.shape[1])
    adjacencyT = adjacency.T
    adjacencyUI = sp.hstack([adjacencyUU, adjacency])
    adjacencyIU = sp.hstack([adjacencyT, adjacencyII])
    adjacencyMatrix = sp.vstack([adjacencyUI, adjacencyIU])
    adjacencyMatrix = adjacencyMatrix.tolil()
    printer("获得邻接矩阵")
    degrees = np.array(adjacencyMatrix.sum(1))  # 按行求和得到rowsum, 即每个节点的度
    dInvSqrt = np.power(degrees, -0.5).flatten()  # (行和rowsum)^(-1/2)
    dInvSqrt[np.isinf(dInvSqrt)] = 0.  # isinf部分赋值为0
    d_mat_inv_sqrt = sp.diags(dInvSqrt)  # 对角化; 将d_inv_sqrt 赋值到对角线元素上, 得到度矩阵^-1/2
    return adjacencyMatrix.dot(d_mat_inv_sqrt).transpose().dot(d_mat_inv_sqrt).tocoo()  # (度矩阵^-1/2)*邻接矩阵*(度矩阵^-1/2)


def cooMatrix2Torch(parser, matrix: sp.coo_matrix):
    """
    将numpy的coo矩阵转化为pytorch可以使用的矩阵
    :param matrix: coo_matrix
    :return: pytorch可以使用的矩阵
    """
    values = matrix.data
    indices = np.vstack((matrix.row, matrix.col))
    i = torch.LongTensor(indices)
    v = torch.Tensor(values)
    shape = matrix.shape
    torchMatrix = torch.sparse.FloatTensor(i, v, torch.Size(shape))
    torchMatrix = torchMatrix.float()
    return torchMatrix.cpu()


def drawFigures(figName, figIndex, figValues):
    plt.rcParams["font.sans-serif"] = ["SimHei"]  # 设置字体
    plt.rcParams["axes.unicode_minus"] = False  # 该语句解决图像中的“-”负号的乱码问题
    plt.plot(figIndex, figValues)
    plt.xlabel("epoch")
    plt.ylabel("sum")
    plt.title(figName)
    plt.savefig(figName+".png")
    plt.show()
    pass


def writeLog(text):
    note = open("log.txt", "a+")
    note.write(text + "\n")
    note.close()


def modelTest(modelCode, dataCode):
    seed = 1024
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    client = initClient(modelCode, dataCode)
    client.train()


def initClient(modelCode, dataCode, dataType):
    parser = ParserFactory.getParser(modelCode, dataCode, dataType)
    client = Client(parser)
    return client


def initSeed(seed=1024):
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)

