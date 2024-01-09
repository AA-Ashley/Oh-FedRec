from abc import ABC

import numpy as np
import torch
from torch import nn
import torch.nn.functional as F
from Model.BasicModel import BasicModel

class GRU4Rec(BasicModel):
    def __init__(self, para):
        super(GRU4Rec, self).__init__(para)
        self.Gru = nn.GRU(self.embedSize, self.embedSize, 1, bias=False).cuda()
        self.logistic = torch.nn.Sigmoid().cuda()

    def computer(self, logSeqs):
        seqs = self.embedItem(logSeqs)
        output, temp = self.Gru(seqs)
        output = self.logistic(output)
        return output

    def forward(self, users, seqs, posItems, negItems):
        logFeats = self.computer(seqs)[:, -1, :]
        embedPos = self.embedItem(posItems)
        embedNeg = self.embedItem(negItems)
        posScore = (logFeats * embedPos).sum(dim=-1)
        negScore = (logFeats * embedNeg).sum(dim=-1)
        return posScore, negScore

    def getLoss(self, users, seqs, posItems, negItems):
        posScores, negScores = self.forward(users, seqs, posItems, negItems)
        loss = - (posScores - negScores).sigmoid().log().mean()
        return loss, 0

    def predict(self, users, seqs, items):
        logFeats = self.computer(seqs)
        finalFeat = logFeats[:, -1, :]
        self.embedUser.weight.data[users.long()] = finalFeat
        # user_ids hasn't been used yet
        # only use last QKV classifier, a waste
        # (U, I, C)
        embedItems = self.embedItem(items)
        embedUsers = self.embedUser(users)
        scores = torch.mul(embedUsers, embedItems)
        scores = torch.sum(scores, dim=1)
        scores = self.logistic(scores)
        return scores
