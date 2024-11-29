import torch
import torch.nn as nn
import torch.nn.functional as F
from recbole_gnn.model.hyp_layers import HypLinear,HypAct

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
                 temp,
                 config):
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

    def forward(self, u,i,adjs):
        # u：user的embedding
        # i：item的embedding
        # 三个线性层
        # 从batch_size*head.dim到batch_size#head*dim最后到head*batch_size*dim
        # head*batch_size*dim
        u_key=[]
        u_query=[]
        u_value=[]
        i_key=[]
        i_query=[]
        i_value=[]
        for k, layer in enumerate(self.Wk):
            u_key.append(layer(u))
            i_key.append(layer(i))
        for k, layer in enumerate(self.Wq):
            u_query.append(layer(u))
            i_query.append(layer(i))
        for k, layer in enumerate(self.Wv):
            u_value.append(layer(u))
            i_value.append(layer(i))
        u_key=torch.stack(u_key,dim=0)
        u_query=torch.stack(u_query,dim=0)
        u_value=torch.stack(u_value,dim=0)
        i_key=torch.stack(i_key,dim=0)
        i_query=torch.stack(i_query,dim=0)
        i_value=torch.stack(i_value,dim=0)
        # _u_value:聚合users embeddings的向量
        _u_value=self.hyper_softmax(u_query,i_key,i_value,adjs[0])
        _i_value=self.hyper_softmax(i_query,u_key,u_value,'skip')
        num_heads=u_value.shape[0]
        dim=u_value.shape[2]
        u_value=self.manifold.mobius_add(u_value.reshape(-1,dim),_u_value.reshape(-1,dim),self.curve)
        u_value=u_value.reshape(num_heads,-1,dim)
        # i_u_weight=F.softmax(-self.manifold.tensor_sqdist(i_query,u_key,self.curve)/self.temp,dim=-1)
        i_value=self.manifold.mobius_add(i_value.reshape(-1,dim),_i_value.reshape(-1,dim),self.curve)
        i_value = i_value.reshape(num_heads, -1, dim)
        final_u_value=self.hyper_agg(u_value)
        final_i_value=self.hyper_agg(i_value)
        # link_loss = torch.mean(weight)
        return final_u_value,final_i_value

class BiHyperFormer(nn.Module):
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
        super(BiHyperFormer, self).__init__()
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
    def forward(self, u,i,adjs):
        link_loss_ = []
        for k, conv in enumerate(self.convs):
            u,i= conv(u,i,adjs)
        return u,i,link_loss_