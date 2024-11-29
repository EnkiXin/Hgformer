import torch as t
from torch import nn
from Params import args
init = nn.init.xavier_uniform_
uniformInit = nn.init.uniform

class Model(nn.Module):
    def __init__(self, gtLayer):
        super(Model, self).__init__()

        self.gcnLayers = nn.Sequential(*[GCNLayer() for i in range(args.gcn_layer)])
        self.gcnLayer = GCNLayer()
        self.gtLayers = gtLayer

    def forward(self, adj,embeds):
        embedsLst = [embeds]
        emb, _ = self.gtLayers(adj, embeds)
        for i, gcn in enumerate(self.gcnLayers):
            embeds = gcn(adj, embedsLst[-1])
            embedsLst.append(embeds)
            embeds, _ = self.gtLayers(adj, embedsLst[-1])
            embedsLst.append(embeds)
        embeds = sum(embedsLst)
        return embeds


class GCNLayer(nn.Module):
    def __init__(self):
        super(GCNLayer, self).__init__()

    def forward(self, adj, embeds):
        return t.spmm(adj, embeds)



class GTLayer(nn.Module):
    def __init__(self):
        super(GTLayer, self).__init__()
        self.qTrans = nn.Parameter(init(t.empty(args.latdim, args.latdim)))
        self.kTrans = nn.Parameter(init(t.empty(args.latdim, args.latdim)))
        self.vTrans = nn.Parameter(init(t.empty(args.latdim, args.latdim)))

    def makeNoise(self, scores):
        noise = t.rand(scores.shape).cuda()
        noise = -t.log(-t.log(noise))
        return scores + 0.01*noise

    def forward(self, adj, embeds, flag=False):
        indices = adj._indices()
        rows, cols = indices[0, :], indices[1, :]
        rowEmbeds = embeds[rows]
        colEmbeds = embeds[cols]

        qEmbeds = (rowEmbeds @ self.qTrans).view([-1, args.head, args.latdim // args.head])
        kEmbeds = (colEmbeds @ self.kTrans).view([-1, args.head, args.latdim // args.head])
        vEmbeds = (colEmbeds @ self.vTrans).view([-1, args.head, args.latdim // args.head])

        att = t.einsum('ehd, ehd -> eh', qEmbeds, kEmbeds)
        att = t.clamp(att, -10.0, 10.0)
        expAtt = t.exp(att)
        tem = t.zeros([adj.shape[0], args.head]).cuda()
        attNorm = (tem.index_add_(0, rows, expAtt))[rows]
        att = expAtt / (attNorm + 1e-8)

        resEmbeds = t.einsum('eh, ehd -> ehd', att, vEmbeds).view([-1, args.latdim])
        tem = t.zeros([adj.shape[0], args.latdim]).cuda()
        resEmbeds = tem.index_add_(0, rows, resEmbeds)  # nd
        return resEmbeds, att
