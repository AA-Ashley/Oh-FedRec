
import torch
from torch import nn
import utils
import torch.nn.functional as F
import scipy.sparse as sp
from sklearn.cluster import KMeans
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")


class NCL(nn.Module):
    def __init__(self, parser):
        super().__init__()
        self.userCount = parser.userCount
        self.itemCount = parser.itemCount
        self.dropOut = parser.dropOut
        graph = sp.load_npz(self.parser.lapMatrix)
        self.graph = utils.cooMatrix2Torch(graph)
        self.embedSize = parser.embedSize
        self.layers = parser.layer
        self.keepProb = 1 - parser.dropOut
        self.embedUser = torch.nn.Embedding(
            num_embeddings=self.userCount, embedding_dim=self.embedSize).cuda()
        self.embedItem = torch.nn.Embedding(
            num_embeddings=self.itemCount, embedding_dim=self.embedSize).cuda()
        nn.init.normal_(self.embedUser.weight, std=0.1)
        nn.init.normal_(self.embedItem.weight, std=0.1)
        self.userCentroids = None
        self.user2cluster = None
        self.itemCentroids = None
        self.item2cluster = None
        self.sslTemp = parser.sslTemp
        self.protoReg = parser.protoReg
        self.sslReg = parser.sslReg
        self.alpha = parser.alpha
        self.hyperLayers = parser.hyperLayers
        self.numClusters = parser.numClusters
        self.graph = self.graph.coalesce()
        self.f = nn.Sigmoid()
        # 存储经过计算后的embedding
        self.finalEmbedUser, self.finalEmbedItem = None, None

    def eStep(self):
        embedUsers = self.user_embedding.weight.detach().cpu().numpy()
        embedItems = self.item_embedding.weight.detach().cpu().numpy()
        self.userCentroids, self.user2cluster = self.run_kmeans(embedUsers)
        self.itemCentroids, self.item2cluster = self.run_kmeans(embedItems)

    @staticmethod
    def dropout(x, keepProb):
        size = x.size()
        index = x.indices().t()
        values = x.values()
        random_index = torch.rand(len(values)) + keepProb
        random_index = random_index.int().bool()
        index = index[random_index]
        values = values[random_index] / keepProb
        g = torch.sparse.FloatTensor(index.t(), values, size)
        return g

    def ProtoNCELoss(self, nodeEmbedding, user, item):
        allEmbedUsers, allEmbedItems = torch.split(nodeEmbedding, [self.userCount, self.itemCount])

        embedUsers = allEmbedUsers[user]     # [B, e]
        normEmbedUsers = F.normalize(embedUsers)
        user2cluster = self.user2cluster[user]     # [B,]
        user2centroids = self.userCentroids[user2cluster]   # [B, e]
        posScoreUsers = torch.mul(normEmbedUsers, user2centroids).sum(dim=1)
        posScoreUsers = torch.exp(posScoreUsers / self.sslTemp)
        ttlScoreUsers = torch.matmul(normEmbedUsers, self.user_centroids.transpose(0, 1))
        ttlScoreUsers = torch.exp(ttlScoreUsers / self.ssl_temp).sum(dim=1)
        protoNceLossUser = -torch.log(posScoreUsers / ttlScoreUsers).sum()
        embedItems = allEmbedItems[item]
        normEmbedItems = F.normalize(embedItems)
        item2cluster = self.item2cluster[item]  # [B, ]
        item2centroids = self.itemCentroids[item2cluster]  # [B, e]
        posScoreItems = torch.mul(normEmbedItems, item2centroids).sum(dim=1)
        posScoreItems = torch.exp(posScoreItems / self.sslTemp)
        ttlScoreItems = torch.matmul(normEmbedItems, self.item_centroids.transpose(0, 1))
        ttlScoreItems = torch.exp(ttlScoreItems / self.sslTemp).sum(dim=1)
        protoNceLossItem = -torch.log(posScoreItems / ttlScoreItems).sum()
        protoNceLoss = self.protoReg * (protoNceLossUser + protoNceLossItem)
        return protoNceLoss

    def sslLayerLoss(self, currentEmbedding, previousEmbedding, user, item):
        currentEmbedUsers, currentEmbedItems = torch.split(currentEmbedding, [self.userCount, self.itemCount])
        previousAllEmbedUsers, previousAllEmbedItems = torch.split(previousEmbedding, [self.userCount, self.itemCount])

        currentEmbedUsers = currentEmbedUsers[user]
        previousEmbedUsers = previousAllEmbedUsers[user]
        normUserEmb1 = F.normalize(currentEmbedUsers)
        normUserEmb2 = F.normalize(previousEmbedUsers)
        normAllUserEmb = F.normalize(previousAllEmbedUsers)
        posScoreUsers = torch.mul(normUserEmb1, normUserEmb2).sum(dim=1)
        ttlScoreUsers = torch.matmul(normUserEmb1, normAllUserEmb.transpose(0, 1))
        posScoreUsers = torch.exp(posScoreUsers / self.sslTemp)
        ttlScoreUsers = torch.exp(ttlScoreUsers / self.sslTemp).sum(dim=1)
        sslLossUser = -torch.log(posScoreUsers / ttlScoreUsers).sum()

        currentEmbedItems = currentEmbedItems[item]
        previousEmbedItems = previousAllEmbedItems[item]
        normItemEmb1 = F.normalize(currentEmbedItems)
        normItemEmb2 = F.normalize(previousEmbedItems)
        normAllItemEmb = F.normalize(previousAllEmbedItems)
        posScoreItems = torch.mul(normItemEmb1, normItemEmb2).sum(dim=1)
        ttlScoreItems = torch.matmul(normItemEmb1, normAllItemEmb.transpose(0, 1))
        posScoreItems = torch.exp(posScoreItems / self.sslTemp)
        ttlScoreItems = torch.exp(ttlScoreItems / self.sslTemp).sum(dim=1)

        sslLossItem = -torch.log(posScoreItems / ttlScoreItems).sum()

        sslLoss = self.sslReg * (sslLossUser + self.alpha * sslLossItem)
        return sslLoss

    def computer(self):
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
            tempEmbed = []
            tempEmbed.append(torch.sparse.mm(graph, allEmbed))
            side_emb = torch.cat(tempEmbed, dim=0)
            allEmbed = side_emb
            embeds.append(allEmbed)
        embeds = torch.stack(embeds, dim=1)
        output = torch.mean(embeds, dim=1)
        users, items = torch.split(output, [self.userCount, self.itemCount])
        return users, items, embeds

    def getUsersRating(self, users):
        allUsers, allItems, embeds = self.computer()
        embedUser = allUsers[users.long()]
        embedItem = allItems
        rating = self.f(torch.matmul(embedUser, embedItem.t()))
        return rating

    def runKmeans(self, x):
        """Run K-means algorithm to get k clusters of the input tensor x
               """
        kmeans = faiss.Kmeans(d=self.embedSize, k=self.numClusters, gpu=True)
        kmeans.train(x)
        cluster_cents = kmeans.centroids

        _, I = kmeans.index.search(x, 1)

        # convert to cuda Tensors for broadcast
        centroids = torch.Tensor(cluster_cents).cuda()
        centroids = F.normalize(centroids, p=2, dim=1)

        node2cluster = torch.LongTensor(I).squeeze().cuda()
        return centroids, node2cluster

    def getLoss(self, users, pos, neg):
        userAllEmbeds, itemAllEmbeds, embeds = self.computer()
        center_embedding = embeds[0]
        context_embedding = embeds[self.hyperLayers * 2]

        sslLoss = self.sslLayerLoss(context_embedding, center_embedding, users, pos)
        protoLoss = self.ProtoNCELoss(center_embedding, users, pos)
        embedUser = userAllEmbeds[users]
        embedPosItem = itemAllEmbeds[pos]
        embedNegItem = itemAllEmbeds[neg]
        posScores = torch.mul(embedUser, embedPosItem)
        posScores = torch.sum(posScores, dim=1)
        negScores = torch.mul(embedUser, embedNegItem)
        negScores = torch.sum(negScores, dim=1)
        # loss = torch.mean(torch.nn.functional.softplus(neg_scores - pos_scores))
        bprLoss = - (posScores - negScores).sigmoid().log().mean()
        loss = bprLoss + sslLoss + protoLoss
        return loss, 0

    def forward(self, users, items):
        # compute embedding
        allUsers, allItems, embeds = self.computer()
        # print('forward')
        # allUsers, allItems = self.computer()
        embedUser = allUsers[users]
        embedItem = allItems[items]
        result = torch.mul(embedUser, embedItem)
        result = torch.sum(result, dim=1)
        return result

    def getFinalEmbed(self):
        finalEmbedUser = F.normalize(self.finalEmbedUser, p=2, dim=1)
        finalEmbedItem = F.normalize(self.finalEmbedItem, p=2, dim=1)
        return finalEmbedUser, finalEmbedItem
