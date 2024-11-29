import torch
import torch.nn as nn
import torch.nn.functional as F
from recbole_gnn.model.hyp_layers import HypLinear,HypAct
import scipy.sparse as sp

class HyperFormerConv(nn.Module):
    '''
    one layer of NodeFormer that attentive aggregates all nodes over a latent graph
    return: node embeddings for next layer, edge loss at this layer
    '''
    def __init__(self,
                 manifold,
                 curve,
                 in_channels,
                 out_channels,
                 num_heads,
                 temp,config):
        super(HyperFormerConv, self).__init__()
        self.manifold=manifold
        self.curve=curve
        self.num_heads = num_heads
        self.Wk=nn.ModuleList()
        self.Wq=nn.ModuleList()
        self.Wv=nn.ModuleList()
        self.config=config

        for i in range(self.num_heads):
            self.Wk.append(HypLinear(manifold,in_channels,out_channels,curve,curve))
            self.Wq.append(HypLinear(manifold,in_channels,out_channels,curve,curve))
            self.Wv.append(HypLinear(manifold,in_channels,out_channels,curve,curve))
        self.out_channels = out_channels
        self.temp=temp


    def hyper_agg(self,tensor):
        # 得到tensor:num_heads*num_nodes*dim
        num_heads,num_nodes,dim=tensor.size(0), tensor.size(1),tensor.size(2)
        tensor=tensor.permute(1,0,2)
        weighted_tensor = torch.full((num_nodes,num_heads,1), 1/num_heads).to(tensor.device)
        tensor = self.manifold.hyperbolic_mean(weighted_tensor,tensor,self.curve)
        # [B,N,D]
        return tensor

    def hyper_softmax(self,
                     query,
                     key,
                     value,
                     adjs
                      ):
        # Q:[H.N,D]（users）
        # K:[H,L,D]（items）
        # V:[H.L,D]（items）
        # 计算双曲距离（距离越小说明越相关）
        if self.config['hyper_version']=='dist':
           sqdist=self.manifold.hyper_dist(query,key,self.curve)
           wij = F.softmax(-sqdist / self.temp, dim=-1)
        elif self.config['hyper_version']=='dot':
            sqdist = self.manifold.minkowski_tensor_dot(query, key)
            wij = F.softmax((sqdist+1/self.curve)/ self.temp, dim=-1)
        # 负双曲距离作为权重，越大说明越相关
        # wij:i是query，j是key
        value=torch.einsum("hnl,hld->hnd",wij,value)
        hyper_coefficient=(1 /(self.curve ** 0.5)) /torch.sqrt(torch.abs(self.manifold.tensor_minkouski_norm(value)))
        z_output = torch.einsum('hnd,hni->hnd', value,hyper_coefficient ) #[H,N,D]
        return z_output

    def forward(self, e,adjs):
        # 三个线性层
        # 从batch_size*head.dim到batch_size#head*dim最后到head*batch_size*dim
        # head*batch_size*dim
        key=[]
        query=[]
        value=[]
        for k, layer in enumerate(self.Wk):
            key.append(layer(e))
        for k, layer in enumerate(self.Wq):
            query.append(layer(e))
        for k, layer in enumerate(self.Wv):
            value.append(layer(e))
        key=torch.stack(key,dim=0)
        query=torch.stack(query,dim=0)
        value=torch.stack(value,dim=0)
        # _u_value:聚合users embeddings的向量
        _value=self.hyper_softmax(query,key,value,adjs[0])
        num_heads=value.shape[0]
        dim=value.shape[2]
        final_value=self.hyper_agg(_value)
        # link_loss = torch.mean(weight)
        return final_value

class HyperFormer(nn.Module):
    def __init__(self,
                 manifold,
                 curve,
                 in_channels,
                 hidden_channels,
                 num_layers,
                 num_heads,
                 temp,
                 config
                 ):
        super(HyperFormer, self).__init__()
        self.dropout=0.1
        self.manifold=manifold
        self.curve=curve
        self.convs = nn.ModuleList()
        self.fcs = nn.ModuleList()
        self.fcs.append(HypLinear(
                                  manifold,
                                  in_channels,
                                  hidden_channels,
                                  curve,
                                  curve
                                  ))
        for i in range(num_layers):
            self.convs.append(
                HyperFormerConv(
                    self.manifold,
                    self.curve,
                    in_channels,
                    hidden_channels,
                    num_heads,
                    temp,
                    config=config
                ))
        self.activation =HypAct(manifold=self.manifold,c_in=self.curve,c_out=self.curve)
    def forward(self, e,adjs):
        link_loss_ = []
        for k, conv in enumerate(self.convs):
            e= conv(e,adjs)
        return e,link_loss_