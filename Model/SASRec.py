from abc import ABC

import numpy as np
import torch
import torch.nn.functional as F
from Model.BasicModel import BasicModel


class PointWiseFeedForward(torch.nn.Module):
    def __init__(self, hidden_units, dropout_rate):
        super(PointWiseFeedForward, self).__init__()

        self.conv1 = torch.nn.Conv1d(hidden_units, hidden_units, kernel_size=1).cuda()
        self.dropout1 = torch.nn.Dropout(p=dropout_rate).cuda()
        self.relu = torch.nn.ReLU().cuda()
        self.conv2 = torch.nn.Conv1d(hidden_units, hidden_units, kernel_size=1).cuda()
        self.dropout2 = torch.nn.Dropout(p=dropout_rate).cuda()

    def forward(self, inputs):
        outputs = self.dropout2(self.conv2(self.relu(self.dropout1(self.conv1(inputs.transpose(-1, -2))))))
        outputs = outputs.transpose(-1, -2)  # as Conv1D requires (N, C, Length)
        outputs += inputs
        return outputs


class SASRec(BasicModel):
    def __init__(self, para):
        super(SASRec, self).__init__(para)
        self.embedPos = torch.nn.Embedding(para.len, para.embedSize).cuda()  # TO IMPROVE
        self.embDropout = torch.nn.Dropout(p=para.dropOut).cuda()

        self.attentionLayerNorms = torch.nn.ModuleList().cuda()
        self.attentionLayers = torch.nn.ModuleList().cuda()
        self.forwardLayerNorms = torch.nn.ModuleList().cuda()
        self.forwardLayers = torch.nn.ModuleList().cuda()

        self.lastLayerNorm = torch.nn.LayerNorm(para.embedSize, eps=1e-8).cuda()
        self.logistic = torch.nn.Sigmoid().cuda()

        for _ in range(para.numBlocks):
            newAttentionLayerNorm = torch.nn.LayerNorm(para.embedSize, eps=1e-8).cuda()
            self.attentionLayerNorms.append(newAttentionLayerNorm)

            newAttentionLayer = torch.nn.MultiheadAttention(para.embedSize, para.numHeads, para.dropOut).cuda()
            self.attentionLayers.append(newAttentionLayer)

            newForwardLayerNorm = torch.nn.LayerNorm(para.embedSize, eps=1e-8).cuda()
            self.forwardLayerNorms.append(newForwardLayerNorm)

            newForwardLayer = PointWiseFeedForward(para.embedSize, para.dropOut).cuda()
            self.forwardLayers.append(newForwardLayer)

        for name, param in self.named_parameters():
            try:
                torch.nn.init.xavier_normal_(param.data)
            except:
                pass  # just ignore those failed init layers

    def computer(self, logSeqs):
        seqs = self.embedItem(logSeqs)
        seqs *= self.embedItem.embedding_dim ** 0.5
        positions = np.tile(np.array(range(logSeqs.shape[1])), [logSeqs.shape[0], 1])
        seqs += self.embedPos(torch.LongTensor(positions).cuda())
        seqs = self.embDropout(seqs)

        timelineMask = (logSeqs == 0)
        seqs *= ~timelineMask.unsqueeze(-1)

        tl = seqs.shape[1]
        attentionMask = ~torch.tril(torch.ones((tl, tl), dtype=torch.bool))
        attentionMask = attentionMask.cuda()

        for i in range(len(self.attentionLayers)):
            seqs = torch.transpose(seqs, 0, 1)
            Q = self.attentionLayerNorms[i](seqs)
            mhaOutputs, _ = self.attentionLayers[i](Q, seqs, seqs, attn_mask=attentionMask)
            seqs = Q + mhaOutputs
            seqs = torch.transpose(seqs, 0, 1)

            seqs = self.forwardLayerNorms[i](seqs)
            seqs = self.forwardLayers[i](seqs)
            seqs *= ~timelineMask.unsqueeze(-1)
        # (U, T, C) -> (U, -1, C)
        logFeats = self.lastLayerNorm(seqs)

        return logFeats

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
