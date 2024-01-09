
import torch
from torch import nn
import utils
import torch.nn.functional as F
import scipy.sparse as sp
from Model.BasicModel import BasicModel
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")


class LightGCN(BasicModel):
    def __init__(self, parser):
        super(LightGCN, self).__init__(parser)
        self.parser = parser
        graph = sp.load_npz(self.parser.lapMatrix)
        self.graph = utils.cooMatrix2Torch(self.parser, graph)
        self.layers = parser.layer
        self.keepProb = 1 - parser.dropOut
        self.graph = self.graph.coalesce()
        self.f = nn.Sigmoid()

    def dropout(self, x, keepProb):
        size = x.size()
        index = x.indices().t()
        values = x.values()
        random_index = torch.rand(len(values)) + keepProb
        random_index = random_index.int().bool()
        index = index[random_index]
        values = values[random_index] / keepProb
        g = torch.sparse.FloatTensor(index.t(), values, size)
        return g

    def computer(self, args):
        """
        propagate methods for lightGCN
        """
        embedUser = self.embedUser.weight
        embedItem = self.embedItem.weight
        allEmbed = torch.cat([embedUser, embedItem])
        #   torch.split(allEmbed , [self.userCount, self.itemCount])
        embeds = [allEmbed]
        graph = self.dropout(self.graph, keepProb=self.keepProb)
        for layer in range(self.layers):
            tempEmbed = [torch.sparse.mm(graph, allEmbed)]
            side_emb = torch.cat(tempEmbed, dim=0)
            allEmbed = side_emb
            embeds.append(allEmbed)
        embeds = torch.stack(embeds, dim=1)
        output = torch.mean(embeds, dim=1)
        users, items = torch.split(output, [self.userCount, self.itemCount])
        return users, items

    def getLoss(self, users, seqs, posItems, negItems):
        posScores, negScores = self.forward(users, seqs=None, posItems=posItems, negItems=negItems)
        # loss = torch.mean(torch.nn.functional.softplus(neg_scores - pos_scores))
        loss = - (posScores - negScores).sigmoid().log().mean()
        return loss, 0

    def forward(self, users, seqs, posItems, negItems):
        # compute embedding
        allUsers, allItems = self.computer(args=None)
        embedUser = allUsers[users]
        embedPosItem = allItems[posItems]
        embedNegItem = allItems[negItems]
        posScores = torch.mul(embedUser, embedPosItem)
        posScores = torch.sum(posScores, dim=1)
        negScores = torch.mul(embedUser, embedNegItem)
        negScores = torch.sum(negScores, dim=1)
        self.finalEmbedUser.weight.data = self.embedUser.weight.data
        self.finalEmbedItem.weight.data = self.embedItem.weight.data
        return posScores, negScores

    def predict(self, users, seqs, items):
        allUsers, allItems = self.computer(args=None)
        embedUser = allUsers[users]
        embedItem = allItems[items]
        score = torch.mul(embedUser, embedItem)
        score = torch.sum(score, dim=1)
        return score
