from torch import nn
import torch
import torch.nn.functional as F


class MLP(nn.Module):
    def __init__(self, layers=None, parser=None):
        super().__init__()
        if layers is None:
            layers = [64, 32]
        self.layers = layers
        self.fcLayers = nn.ModuleList()
        for idx, (inSize, outSize) in enumerate(zip(self.layers[:-1], self.layers[1:])):
            self.fcLayers.append(torch.nn.Linear(inSize, outSize).cuda())
            # self.fcLayers.append(torch.nn.BatchNorm1d(outSize)).cuda()
            # self.fcLayers.append(nn.ReLU().cuda())
            # self.fcLayers.append(torch.nn.Dropout(p=0.2)).cuda()
        self.fcLayers.append(nn.ReLU().cuda())

        for m in self.fcLayers:
            if isinstance(m, nn.Linear):
                nn.init.xavier_uniform_(m.weight)

    def forward(self, embed):
        for idx, _ in enumerate(range(len(self.fcLayers))):
            embed = self.fcLayers[idx](embed)
        return embed


class Aggregation(nn.Module):
    def __init__(self, layers=None, parser=None):
        super().__init__()
        if layers is None:
            layers = [64, 32]
        self.layers = layers
        self.fcLayers = nn.ModuleList()
        for idx, (inSize, outSize) in enumerate(zip(self.layers[:-1], self.layers[1:])):
            self.fcLayers.append(torch.nn.Linear(inSize, outSize).cuda())
            # self.fcLayers.append(torch.nn.BatchNto(orm1d(outSize)).cuda()
            # self.fcLayers.append(torch.nn.Dropout(p=0.2)).cuda()
        self.func = nn.PReLU().cuda()

        for m in self.fcLayers:
            if isinstance(m, nn.Linear):
                nn.init.xavier_uniform_(m.weight)

    def forward(self, embed):
        for idx, _ in enumerate(range(len(self.fcLayers))):
            embed = self.fcLayers[idx](embed)
        embed = self.func(embed)
        return embed


class SelfAttention(nn.Module):
    def __init__(self, inputSize, outputSize):
        """
        :param embed_dim:
        :param att_size:
        """
        super(SelfAttention, self).__init__()
        self.transQ = nn.Linear(inputSize, outputSize)
        self.transK = nn.Linear(inputSize, outputSize)
        self.transV = nn.Linear(inputSize, outputSize)
        self.projection = nn.Linear(outputSize, outputSize)
        # self.scale = 1.0/ torch.LongTensor(embed_dim)
        # self.scale = torch.sqrt(1.0 / torch.tensor(embed_dim).float())
        # self.dropout = nn.Dropout(0.5)
        # self.layer_norm = nn.LayerNorm(embed_dim)

    def forward(self, x,):
        """
        :param x: B,F,E
        :return: B,F,E
        """
        Q = self.transQ(x)
        K = self.transK(x)
        V = self.transV(x)
        attention = torch.matmul(Q, K.permute(0, 2, 1))  # B,F,F
        attention_score = F.softmax(attention, dim=-1)
        context = torch.matmul(attention_score, V)
        context = self.projection(context)
        return context


class IEU(nn.Module):
    """
    Information extraction Unit (IEU) for FRNet
    (1) Self-attention
    (2) DNN
    """
    def __init__(self, length, inputSize, weight_type="bit",
                 bit_layers=1, outputSize=32, mlpLayer=256):
        """
        :param field_length:
        :param embed_dim:
        :param type: vector or bit
        :param bit_layers:
        :param att_size:
        :param mlp_layer:
        """
        super(IEU, self).__init__()
        self.inputSize = length * inputSize
        self.weight_type = weight_type

        # Self-attention unit, which is used to capture cross-feature relationships.
        self.vector_info = SelfAttention(inputSize, outputSize)

        #  contextual information extractor(CIE), we adopt MLP to encode contextual information.
        self.mlp = MLP([self.inputSize, mlpLayer])
        self.bitProjection = nn.Linear(mlpLayer, outputSize)
        self.activation = nn.ReLU()
        # self.activation = nn.PReLU()

    def forward(self, emb):
        """
        :param x_emb: B,F,E
        :return: B,F,E (bit-level weights or complementary fetures)
                 or B,F,1 (vector-level weights)
        """

        # （1）self-attetnion unit
        vector = self.vector_info(emb)  # B,F,E

        # (2) CIE unit
        bit = self.mlp(emb.view(-1, self.inputSize))
        bit = self.bitProjection(bit).unsqueeze(1) # B,1,e
        bit = self.activation(bit)

        # （3）integration unit
        out = bit * vector

        if self.weight_type == "vector":
            # To compute vector-level importance in IEU_W
            out = torch.sum(out, dim=2, keepdim=True)
            # B,F,1
            return out
        return out
