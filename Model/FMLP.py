
import copy
from abc import ABC

import math
import torch
import torch.nn as nn
import torch.nn.functional as F
from Model.BasicModel import BasicModel


def gelu(x):
    """Implementation of the gelu activation function.
        For information: OpenAI GPT's gelu is slightly different
        (and gives slightly different results):
        0.5 * x * (1 + torch.tanh(math.sqrt(2 / math.pi) *
        (x + 0.044715 * torch.pow(x, 3))))
        Also see https://arxiv.org/abs/1606.08415
    """
    return x * 0.5 * (1.0 + torch.erf(x / math.sqrt(2.0)))


def swish(x):
    return x * torch.sigmoid(x)


ACT2FN = {"gelu": gelu, "relu": F.relu, "swish": swish}


class LayerNorm(nn.Module):
    def __init__(self, embedSize, eps=1e-12):
        """Construct a layernorm module in the TF style (epsilon inside the square root).
        """
        super(LayerNorm, self).__init__()
        self.weight = nn.Parameter(torch.ones(embedSize))
        self.bias = nn.Parameter(torch.zeros(embedSize))
        self.variance_epsilon = eps

    def forward(self, x):
        u = x.mean(-1, keepdim=True)
        s = (x - u).pow(2).mean(-1, keepdim=True)
        x = (x - u) / torch.sqrt(s + self.variance_epsilon)
        return self.weight * x + self.bias


class SelfAttention(nn.Module):
    def __init__(self, parser):
        super(SelfAttention, self).__init__()
        if parser.embedSize % parser.numHeads != 0:
            raise ValueError(
                "The hidden size (%d) is not a multiple of the number of attention "
                "heads (%d)" % (parser.embedSize, parser.numHeads))
        self.num_attention_heads = parser.numHeads
        self.attention_head_size = int(parser.embedSize / parser.numHeads)
        self.all_head_size = self.num_attention_heads * self.attention_head_size

        self.query = nn.Linear(parser.embedSize, self.all_head_size)
        self.key = nn.Linear(parser.embedSize, self.all_head_size)
        self.value = nn.Linear(parser.embedSize, self.all_head_size)

        self.attn_dropout = nn.Dropout(parser.attentionDropOut)

        # 做完self-attention 做一个前馈全连接 LayerNorm 输出
        self.dense = nn.Linear(parser.embedSize, parser.embedSize)
        self.LayerNorm = LayerNorm(parser.embedSize, eps=1e-12)
        self.out_dropout = nn.Dropout(parser.hiddenDropOut)

    def transpose_for_scores(self, x):
        new_x_shape = x.size()[:-1] + (self.num_attention_heads, self.attention_head_size)
        x = x.view(*new_x_shape)
        return x.permute(0, 2, 1, 3)

    def forward(self, input_tensor, attentionMask):
        mixed_query_layer = self.query(input_tensor)
        mixed_key_layer = self.key(input_tensor)
        mixed_value_layer = self.value(input_tensor)

        query_layer = self.transpose_for_scores(mixed_query_layer)
        key_layer = self.transpose_for_scores(mixed_key_layer)
        value_layer = self.transpose_for_scores(mixed_value_layer)

        # Take the dot product between "query" and "key" to get the raw attention scores.
        attention_scores = torch.matmul(query_layer, key_layer.transpose(-1, -2))

        attention_scores = attention_scores / math.sqrt(self.attention_head_size)
        # Apply the attention mask is (precomputed for all layers in BertModel forward() function)
        # [batch_size heads seq_len seq_len] scores
        # [batch_size 1 1 seq_len]
        attention_scores = attention_scores + attentionMask

        # Normalize the attention scores to probabilities.
        attention_probs = nn.Softmax(dim=-1)(attention_scores)
        # This is actually dropping out entire tokens to attend to, which might
        # seem a bit unusual, but is taken from the original Transformer paper.
        # Fixme
        attention_probs = self.attn_dropout(attention_probs)
        context_layer = torch.matmul(attention_probs, value_layer)
        context_layer = context_layer.permute(0, 2, 1, 3).contiguous()
        new_context_layer_shape = context_layer.size()[:-2] + (self.all_head_size,)
        context_layer = context_layer.view(*new_context_layer_shape)
        hidden_states = self.dense(context_layer)
        hidden_states = self.out_dropout(hidden_states)
        hidden_states = self.LayerNorm(hidden_states + input_tensor)

        return hidden_states


class FilterLayer(nn.Module):
    def __init__(self, parser):
        super(FilterLayer, self).__init__()
        # 做完self-attention 做一个前馈全连接 LayerNorm 输出
        self.complex_weight = nn.Parameter(torch.randn(1, parser.len//2 + 1, parser.embedSize, 2, dtype=torch.float32) * 0.02)
        self.out_dropout = nn.Dropout(parser.hiddenDropOut)
        self.LayerNorm = LayerNorm(parser.embedSize, eps=1e-12)

    def forward(self, input_tensor):
        # [batch, seq_len, hidden]
        batch, seq_len, hidden = input_tensor.shape
        x = torch.fft.rfft(input_tensor, dim=1, norm='ortho')
        weight = torch.view_as_complex(self.complex_weight)
        x = x * weight
        embedSeq_fft = torch.fft.irfft(x, n=seq_len, dim=1, norm='ortho')
        hidden_states = self.out_dropout(embedSeq_fft)
        hidden_states = self.LayerNorm(hidden_states + input_tensor)
        return hidden_states


class Intermediate(nn.Module):
    def __init__(self, parser):
        super(Intermediate, self).__init__()
        self.dense_1 = nn.Linear(parser.embedSize, parser.embedSize * 4)
        if isinstance(parser.hiddenAct, str):
            self.intermediate_act_fn = ACT2FN[parser.hiddenAct]
        else:
            self.intermediate_act_fn = parser.hiddenAct

        self.dense_2 = nn.Linear(4 * parser.embedSize, parser.embedSize)
        self.LayerNorm = LayerNorm(parser.embedSize, eps=1e-12)
        self.dropout = nn.Dropout(parser.hiddenDropOut)

    def forward(self, input_tensor):
        hidden_states = self.dense_1(input_tensor)
        hidden_states = self.intermediate_act_fn(hidden_states)

        hidden_states = self.dense_2(hidden_states)
        hidden_states = self.dropout(hidden_states)
        hidden_states = self.LayerNorm(hidden_states + input_tensor)

        return hidden_states


class Layer(nn.Module):
    def __init__(self, parser):
        super(Layer, self).__init__()
        self.noFilters = parser.noFilters
        if self.noFilters:
            self.attention = SelfAttention(parser)
        else:
            self.filterlayer = FilterLayer(parser)
        self.intermediate = Intermediate(parser)

    def forward(self, hidden_states, attentionMask):
        if self.noFilters:
            hidden_states = self.attention(hidden_states, attentionMask)
        else:
            hidden_states = self.filterlayer(hidden_states)

        intermediate_output = self.intermediate(hidden_states)
        return intermediate_output


class Encoder(nn.Module):
    def __init__(self, parser):
        super(Encoder, self).__init__()
        layer = Layer(parser)
        self.layer = nn.ModuleList([copy.deepcopy(layer)
                                    for _ in range(parser.numBlocks)])

    def forward(self, hidden_states, attentionMask, output_all_encoded_layers=True):
        all_encoder_layers = []
        for layer_module in self.layer:
            hidden_states = layer_module(hidden_states, attentionMask)
            if output_all_encoded_layers:
                all_encoder_layers.append(hidden_states)
        if not output_all_encoded_layers:
            all_encoder_layers.append(hidden_states)
        return all_encoder_layers


class FMLPRecModel(BasicModel, ABC):
    def __init__(self, parser):
        super(FMLPRecModel, self).__init__(parser)
        self.parser = parser
        self.embedPos = nn.Embedding(parser.len, parser.embedSize).cuda()
        self.LayerNorm = LayerNorm(parser.embedSize, eps=1e-12).cuda()
        self.dropout = nn.Dropout(parser.hiddenDropOut).cuda()
        self.itemEncoder = Encoder(parser).cuda()
        self.sigmoid = torch.nn.Sigmoid().cuda()
        self.apply(self.init_weights)

    def init_weights(self, module):
        """ Initialize the weights.
        """
        if isinstance(module, (nn.Linear, nn.Embedding)):
            # Slightly different from the TF version which uses truncated_normal for initialization
            # cf https://github.com/pytorch/pytorch/pull/5617
            module.weight.data.normal_(mean=0.0, std=self.parser.initializerRange)
        elif isinstance(module, LayerNorm):
            module.bias.data.zero_()
            module.weight.data.fill_(1.0)
        if isinstance(module, nn.Linear) and module.bias is not None:
            module.bias.data.zero_()

    def addPositionEmbed(self, sequence):
        seqLen = sequence.size(1)
        positionIds = torch.arange(seqLen, dtype=torch.long).cuda()
        positionIds = positionIds.unsqueeze(0).expand_as(sequence)
        embedItem = self.embedItem(sequence)
        embedPos = self.embedPos(positionIds)
        embedSeq = embedItem + embedPos
        embedSeq = self.LayerNorm(embedSeq)
        embedSeq = self.dropout(embedSeq)
        return embedSeq

    # same as SASRec
    def computer(self, seqs):
        attentionMask = (seqs > 0).long().cuda()
        extentAttentionMask = attentionMask.unsqueeze(1).unsqueeze(2).cuda() # torch.int64
        max_len = attentionMask.size(-1)
        attentionShape = (1, max_len, max_len)
        subSequence = torch.triu(torch.ones(attentionShape), diagonal=1).cuda() # torch.uint8
        subSequence = (subSequence == 0).unsqueeze(1).cuda()
        subSequence = subSequence.long()
        subSequence = subSequence.cuda()
        extentAttentionMask = extentAttentionMask * subSequence
        extentAttentionMask = extentAttentionMask.to(dtype=next(self.parameters()).dtype) # fp16 compatibility
        extentAttentionMask = (1.0 - extentAttentionMask) * -10000.0
        embedSeq = self.addPositionEmbed(seqs)
        itemEncoderLayers = self.itemEncoder(embedSeq, extentAttentionMask, output_all_encoded_layers=True)
        output = itemEncoderLayers[-1]
        return output

    def forward(self, users, seqs, posItems, negItems):
        logFeats = self.log2feats(seqs)[:, -1]
        embedPos = self.embedItem(posItems)
        embedNeg = self.embedItem(negItems)
        # 获取经过计算后的embedding
        self.finalEmbedUser.weight.data[users.long()] = logFeats
        self.finalEmbedItem.weight.data = self.embedItem.weight.data
        posScore = (logFeats * embedPos).sum(dim=-1)
        negScore = (logFeats * embedNeg).sum(dim=-1)
        return posScore, negScore

    def getLoss(self, users, seqs, posItems, negItems):
        bceCriterion = torch.nn.BCEWithLogitsLoss()
        posScores, negScores = self.forward(users, seqs, posItems, negItems)
        posLabels, negLabels = torch.ones(posScores.shape).cuda(), torch.zeros(negScores.shape).cuda()
        indices = torch.where(posItems != 0)
        # loss = bceCriterion(posScores[indices], posLabels[indices])
        # loss += bceCriterion(negScores[indices], negLabels[indices])
        loss = bceCriterion(posScores[indices], posLabels[indices])
        # loss = - (posScores[indices] - negScores[indices]).sigmoid().log().mean()
        return loss, 0

    def predict(self, users, seqs, items):
        # user_ids hasn't been used yet
        logFeats = self.log2feats(seqs)
        # only use last QKV classifier, a waste














        finalFeat = logFeats[:, -1, :]
        # (U, I, C)
        embedItems = self.embedItem(items)
        scores = torch.mul(finalFeat, embedItems)
        scores = torch.sum(scores, dim=1)
        # scores = self.sigmoid(scores)
        return scores



