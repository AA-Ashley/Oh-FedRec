from abc import ABC
import torch
import torch.nn as nn
from Model.BasicModel import BasicModel


class MultiLayerPerceptron(BasicModel, ABC):
    def __init__(self, parser):
        super(MultiLayerPerceptron, self).__init__(parser)
        self.layers = parser.layers
        self.fcLayers = nn.ModuleList()
        for idx, (inSize, outSize) in enumerate(zip(self.layers[:-1], self.layers[1:])):
            self.fcLayers.append(nn.Linear(inSize, outSize).cuda())

        self.affineOut = nn.Linear(in_features=self.layers[-1], out_features=1).cuda()
        self.logistic = nn.Sigmoid().cuda()

    def forward(self, users, seqs, posItems, negItems):
        posScores = self.predict(users, seqs, posItems)
        negScores = self.predict(users, seqs, negItems)
        return posScores, negScores

    def predict(self, users, seqs, items):
        embedUser = self.embedUser(users)
        embedItem = self.embedItem(items)
        vector = torch.cat([embedUser, embedItem], dim=-1)  # the concat latent vector
        for idx, _ in enumerate(range(len(self.fcLayers))):
            vector = self.fcLayers[idx](vector)
            vector = nn.ReLU()(vector)
        scores = self.affineOut(vector)
        scores = self.logistic(scores)
        return scores

    def getLoss(self, users, seqs, posItems, negItems):
        posScore, negScore = self.forward(users, seqs, posItems, negItems)
        loss = - (posScore - negScore).sigmoid().log().mean()
        return loss, 0
