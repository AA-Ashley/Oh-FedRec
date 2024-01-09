from abc import ABC
import torch
import torch.nn as nn
from Model.BasicModel import BasicModel


class MatrixFactorization(BasicModel, ABC):
    def __init__(self, parser):
        super(MatrixFactorization, self).__init__(parser)
        self.affineOut = nn.Linear(in_features=self.embedSize, out_features=1).to(parser.device)
        self.logistic = nn.Sigmoid().to(parser.device)

    def forward(self, users, seqs, posItems, negItems):
        embedUser = self.embedUser(users)
        embedPosItem = self.embedItem(posItems)
        embedNegItem = self.embedItem(negItems)
        posScores, negScores = torch.mul(embedUser, embedPosItem), torch.mul(embedUser, embedNegItem)
        posScores, negScores = self.affineOut(posScores), self.affineOut(negScores)
        posScores, negScores = self.logistic(posScores), self.logistic(negScores)
        self.finalEmbedUser.weight.data = self.embedUser.weight.data
        self.finalEmbedItem.weight.data = self.embedItem.weight.data
        return posScores, negScores

    def predict(self, users, seqs, items):
        embedUser = self.finalEmbedUser(users)
        embedItem = self.finalEmbedItem(items)
        scores = torch.mul(embedUser, embedItem)
        scores = self.affineOut(scores)
        scores = self.logistic(scores)
        return scores.squeeze()

    def getLoss(self, users, seqs, posItems, negItems):
        posScore, negScore = self.forward(users, seqs, posItems, negItems)
        loss = - (posScore - negScore).sigmoid().log().mean()
        return loss, 0