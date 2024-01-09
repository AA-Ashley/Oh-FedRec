from abc import ABC

import torch.nn as nn
import torch.nn.functional as F
from Model.BasicModel import BasicModel


class BPR(BasicModel, ABC):
    def __init__(self, parser):
        """
        user_num: number of users;
        item_num: number of items;
        factor_num: number of predictive factors.
        """
        super(BPR, self).__init__(parser)

    def forward(self, users, seqs, posItems, negItems):
        embedUser = self.embedUser(users)
        embedPosItem = self.embedItem(posItems)
        embedNegItem = self.embedItem(negItems)
        posScores = (embedUser * embedPosItem).sum(dim=-1)
        negScores = (embedUser * embedNegItem).sum(dim=-1)
        self.finalEmbedUser.weight.data = self.embedUser.weight.data
        self.finalEmbedItem.weight.data = self.embedItem.weight.data
        return posScores, negScores

    def getLoss(self, users, seqs, posItems, negItems):
        posScore, negScore = self.forward(users, seqs, posItems, negItems)
        loss = - (posScore - negScore).sigmoid().log().mean()
        return loss, 0

    def predict(self,  users, seqs, items):
        # embedUser = self.embedUser(users)
        # embedItem = self.embedItem(items)
        embedUser = self.finalEmbedUser(users)
        embedItem = self.finalEmbedItem(items)
        scores = (embedUser * embedItem).sum(dim=-1)
        return scores


