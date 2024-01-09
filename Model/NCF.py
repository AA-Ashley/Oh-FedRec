from abc import ABC

import torch
import torch.nn as nn
from Model.BasicModel import BasicModel


class NeuMF(BasicModel, ABC):
    def __init__(self, args):
        super(NeuMF, self).__init__()
        self.userCount = args.userCount
        self.itemCount = args.itemCount
        self.embedSizeMF = args.embedSize
        self.embedSizeMLP = int(args.mlpLayers[0] / 2)
        self.layers = args.mlpLayers
        self.dropout = args.dropOut

        self.embedUserMLP = nn.Embedding(num_embeddings=self.userCount, embedding_dim=self.embedSizeMLP).cuda()
        self.embedItemMLP = nn.Embedding(num_embeddings=self.itemCount, embedding_dim=self.embedSizeMLP).cuda()

        self.embedUserMF = nn.Embedding(num_embeddings=self.userCount, embedding_dim=self.embedSizeMF).cuda()
        self.embedItemMF = nn.Embedding(num_embeddings=self.itemCount, embedding_dim=self.embedSizeMF).cuda()

        self.fcLayers = nn.ModuleList()
        for idx, (inSize, outSize) in enumerate(zip(self.layers[:-1], self.layers[1:])):
            self.fcLayers.append(torch.nn.Linear(inSize, outSize).cuda())
            self.fcLayers.append(nn.ReLU().cuda())

        self.affineOut = nn.Linear(in_features=self.layers[-1] + self.embedSizeMF, out_features=1).cuda()
        self.logistic = nn.Sigmoid().cuda()
        self.initWeight()

    def initWeight(self):
        nn.init.normal_(self.embedUserMLP.weight, std=0.01)
        nn.init.normal_(self.embedItemMLP.weight, std=0.01)
        nn.init.normal_(self.embedUserMF.weight, std=0.01)
        nn.init.normal_(self.embedItemMF.weight, std=0.01)

        for m in self.fcLayers:
            if isinstance(m, nn.Linear):
                nn.init.xavier_uniform_(m.weight)

        nn.init.xavier_uniform_(self.affineOut.weight)

        for m in self.modules():
            if isinstance(m, nn.Linear) and m.bias is not None:
                m.bias.data.zero_()

    def predict(self, users, seqs, items):
        embedUserMLP = self.embedUserMLP(users)
        embedItemMLP = self.embedItemMLP(items)
        embedUserMF = self.embedUserMF(users)
        embedItemMF = self.embedItemMF(items)
        mlpVector = torch.cat([embedUserMLP, embedItemMLP], dim=-1)  # the concat latent vector
        mfVector = torch.mul(embedUserMF, embedItemMF)
        for idx, _ in enumerate(range(len(self.fcLayers))):
            mlpVector = self.fcLayers[idx](mlpVector)
        scores = torch.cat([mlpVector, mfVector], dim=-1)
        scores = self.affineOut(scores)
        scores = self.logistic(scores)
        return scores.squeeze()

    def forward(self, users, seqs, posItems, negItems):
        posScores = self.predict(users, seqs, posItems)
        negScores = self.predict(users, seqs, negItems)
        return posScores, negScores

    def getLoss(self, users, pos, neg):
        bceCriterion = nn.BCELoss()
        posScores = self.forward(users, pos)
        negScores = self.forward(users, neg)
        posLabels, negLabels = torch.ones(posScores.shape).cuda(), torch.zeros(negScores.shape).cuda()
        indices = torch.where(pos != 0)
        loss = bceCriterion(posScores[indices], posLabels[indices])
        loss += bceCriterion(negScores[indices], negLabels[indices])
        return loss, 0

    def getFinalEmbed(self):
        return self.user