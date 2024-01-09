import torch
import Constant


class EmbedPath:
    def __init__(self, modelCode, dataSetName, parser):
        self.modelName = modelCode
        self.dataSetName = dataSetName
        self.parser = parser

    def getEmbed(self):
        userPath = "./save/{0}/{1}/userEmbed.pth".format(self.modelName, self.dataSetName)
        itemPath = "./save/{0}/{1}/itemEmbed.pth".format(self.modelName, self.dataSetName)
        userEmbed = torch.load(userPath, map_location='cuda:0')
        itemEmbed = torch.load(itemPath, map_location='cuda:0')
        return userEmbed, itemEmbed


if __name__ == '__main__':
    userEmbedTeacher, itemEmbedTeacher = EmbedPath(Constant.BPR_CODE, Constant.TMALL_LIKE).getEmbed()
    userEmbedStudent, itemEmbedStudent = EmbedPath(Constant.BPR_CODE, Constant.TMALL_CAR).getEmbed()
