import math,os
import torch
import numpy as np
import torch.nn as nn
import torch.nn.functional as F
from torch_sparse import SparseTensor, matmul
from torch_geometric.utils import degree
from hyperbolic_gnn.model.hgcn.layers.hyp_layers import HypLinear
import hyperbolic_gnn.model.hgcn.manifolds.hyperboloid as manifolds
BIG_CONSTANT = 1e8

def create_products_of_givens_rotations(dim,
                                        seed):
    nb_givens_rotations = dim * int(math.ceil(math.log(float(dim))))
    q = np.eye(dim, dim)
    np.random.seed(seed)
    for _ in range(nb_givens_rotations):
        random_angle = math.pi * np.random.uniform()
        random_indices = np.random.choice(dim, 2)
        index_i = min(random_indices[0], random_indices[1])
        index_j = max(random_indices[0], random_indices[1])
        slice_i = q[index_i]
        slice_j = q[index_j]
        new_slice_i = math.cos(random_angle) * slice_i + math.cos(random_angle) * slice_j
        new_slice_j = -math.sin(random_angle) * slice_i + math.cos(random_angle) * slice_j
        q[index_i] = new_slice_i
        q[index_j] = new_slice_j
    return torch.tensor(q, dtype=torch.float32)

class LightHyperFormerConv(nn.Module):
    '''
    one layer of NodeFormer that attentive aggregates all nodes over a latent graph
    return: node embeddings for next layer, edge loss at this layer
    '''
    def __init__(self,
                 config,
                 tau,
                 curve,
                 in_channels,
                 out_channels,
                 num_heads,
                 manifold,
                 projection_matrix_type='a',
                 nb_random_features=10,
                 use_gumbel=True,
                 nb_gumbel_sample=10,
                 rb_order=0,
                 rb_trans='sigmoid',
                 use_edge_loss=False,
                 simple=False
                 ):
        super(LightHyperFormerConv, self).__init__()
        self.config=config
        self.tau=tau
        self.Wk=nn.ModuleList()
        self.Wq=nn.ModuleList()
        self.Wv=nn.ModuleList()
        self.manifold = manifold
        self.num_heads = num_heads
        self.curve=torch.tensor(curve)
        self.in_channels=in_channels
        for i in range(self.num_heads):
            self.Wk.append(HypLinear(self.manifold,
                                     in_channels,
                                     out_channels,
                                     curve,
                                     curve))
            self.Wq.append(HypLinear(self.manifold,
                                     in_channels,
                                     out_channels,
                                     curve,
                                     curve))
            self.Wv.append(HypLinear(self.manifold,
                                     in_channels,
                                     out_channels,
                                     curve,
                                     curve))
        self.out_channels = out_channels
        self.projection_matrix_type = projection_matrix_type
        self.nb_random_features = nb_random_features
        self.use_gumbel = use_gumbel
        self.nb_gumbel_sample = nb_gumbel_sample
        self.rb_order = rb_order
        self.rb_trans = rb_trans
        self.use_edge_loss = use_edge_loss
        self.simple=simple
        self.b = torch.nn.Parameter(torch.FloatTensor(rb_order, num_heads), requires_grad=True)

    def _create_projection_matrix(self,m, d, seed=0, scaling=0, struct_mode=False):
        nb_full_blocks = int(m / d)
        block_list = []
        current_seed = seed
        for _ in range(nb_full_blocks):
            torch.manual_seed(current_seed)
            if struct_mode:
                q = create_products_of_givens_rotations(d, current_seed)
            else:
                unstructured_block = torch.randn((d, d))
                q, _ = torch.qr(unstructured_block)
                q = torch.t(q)
            block_list.append(q)
            current_seed += 1
        remaining_rows = m - nb_full_blocks * d
        if remaining_rows > 0:
            torch.manual_seed(current_seed)
            if struct_mode:
                q = create_products_of_givens_rotations(d, current_seed)
            else:
                unstructured_block = torch.randn((d, d))
                q, _ = torch.qr(unstructured_block)
                q = torch.t(q)
            block_list.append(q[0:remaining_rows])
        final_matrix = torch.vstack(block_list)
        current_seed += 1
        torch.manual_seed(current_seed)
        if scaling == 0:
            multiplier = torch.norm(torch.randn((m, d)), dim=1)
        elif scaling == 1:
            multiplier = torch.sqrt(torch.tensor(float(d))) * torch.ones(m)
        else:
            raise ValueError("Scaling must be one of {0, 1}. Was %s" % scaling)
        return torch.matmul(torch.diag(multiplier), final_matrix)
    def _softmax_kernel_transformation(self,
                                       data,
                                       projection_matrix,
                                       numerical_stabilizer=0.000001):
        # print(data)
        data_time=data[:,:,:,0].unsqueeze(-1)
        data_space=data[:,:,:,1:]
        # 根号m分之一
        m = 1.0 / torch.sqrt(torch.tensor(projection_matrix.shape[0], dtype=torch.float32)).to(data.device)
        # 在输入矩阵上乘以随机形成的矩阵，得到x~
        wx = torch.exp(torch.einsum("bnhd,dm->bnhm", data_space, projection_matrix)/torch.sqrt(torch.tensor(self.tau)))
        # data_dash:batch*num_nodes*num_heads*random_feats
        front =(torch.exp( (self.curve/self.tau-torch.square((data_time))/self.tau )/2.0)) * m
        data_dash =  torch.einsum('ijkl,ijkm->ijkl', wx, front)
        return data_dash

    def _denominator(self,qs, ks):
        all_ones = torch.ones([ks.shape[0]]).to(qs.device)
        ks_sum = torch.einsum("nbhm,n->bhm", ks, all_ones)  # ks_sum refers to O_k in the paper
        return torch.einsum("nbhm,bhm->nbh", qs, ks_sum)

    def minkowski_tensor_dot(self,X,Y):
        # [H,B,N,D]
        # 计算两个tensor所有的pair的闵可夫斯基内积
        X_time = X[..., 0].unsqueeze(-1)
        X_space = X[..., 1:]
        Y_time = Y[..., 0].unsqueeze(-1)
        Y_space = Y[..., 1:]
        res = -torch.einsum('...nd,...ld->...nl',X_time,Y_time) + torch.einsum('...nd,...ld->...nl',X_space,Y_space)
        return res

    def hyper_dist(self,tensor1,tensor2):
        theta=-self.minkowski_tensor_dot(tensor1,tensor2)*self.curve
        sqdist=torch.acosh(theta)**2/self.curve
        return sqdist

    def pair_wise_minkowski_tensor_dot(self, X, Y):
        # [H,B,N,D]
        # 计算两个同形状tensor两两间的闵可夫斯基内积
        X_time = X[..., 0].unsqueeze(-1)
        X_space = X[..., 1:]
        Y_time = Y[..., 0].unsqueeze(-1)
        Y_space = Y[..., 1:]
        res=-(X_time*Y_time)+torch.sum(X_space*Y_space,dim=-1).unsqueeze(-1)
        return res

    def pair_wize_hyper_dist(self,tensor1,tensor2):
        theta=-self.pair_wise_minkowski_tensor_dot(tensor1,tensor2)*self.curve
        sqdist=torch.acosh(theta)**2/self.curve
        return sqdist


    def tensor_minkouski_norm(self,x):
        # 所有向量与自己做闵可夫斯基内积
        x_time=x[..., 0].unsqueeze(-1)
        x_space=x[..., 1:]
        norm=-(x_time*x_time)+torch.sum(x_space*x_space,dim=-1).unsqueeze(-1)
        return norm

    def kernelized_gumbel_softmax(self,
                                  query,
                                  key,
                                  value,
                                  edge_index=None,
                                  ):
        '''
        fast computation of all-pair attentive aggregation with linear complexity
        input: query/key/value [B, N, H, D]
        return: updated node emb, attention weight (for computing edge loss)
        B = graph number (always equal to 1 in Node Classification), N = node number, H = head number,
        M = random feature dimension, D = hidden size, K = number of Gumbel sampling
        '''
        #query = query / math.sqrt(tau)
        #key = key / math.sqrt(tau)
        # [B,N,H,D]
        # 为了进行随机映射而生成的矩阵
        projection_matrix = self._create_projection_matrix(self.in_channels-1,
                                                           self.nb_random_features).to('cuda')
        # 将Q和K随机映射
        query_prime = self._softmax_kernel_transformation(
                                                          query,
                                                          projection_matrix).permute(2,0,1,3)  # [H,B,N,M]
        key_prime = self._softmax_kernel_transformation(
                                                        key,
                                                        projection_matrix).permute(2,0,1,3) # [H,B,N,M]
        value = value.permute(2,0,1,3) # [H,B,N,D]
        # print(torch.einsum("hbnm,hbkm->hbnk", query_prime, key_prime))
        # print( torch.einsum('hbnm,hblm->hbnl', query_prime,key_prime))
        # 先计算QV
        last_ = torch.einsum("hbnm,hbnd->hbmd", key_prime, value)  # [H,B,M,D]
        # 将QV乘上去
        z_output = torch.einsum("hbnm,hbmd->hbnd", query_prime, last_) # [H,B,N,D]
        # 计算双曲加权平均
        hyper_coefficient=(1 /(self.curve ** 0.5)) /torch.sqrt(torch.abs(self.tensor_minkouski_norm(z_output)))
        z_output = torch.einsum('hbnd,hbnq->hbnd', z_output,hyper_coefficient) #[H,B,N,D]
        # graph的src和dst
        start, end = edge_index
        # 取出src和dst对应的向量
        # [H,B,N,M]中，从N所在维度取
        query_end, key_start = query_prime[:,:,end,:], key_prime[:,:,start,:]  # [H, B, E, M]
        # 计算对应的权重
        edge_attn_num = torch.einsum("hbem,hbem->hbe", query_end, key_start)  # [H,B,E]
        A_weight = edge_attn_num.permute(1, 2, 0)  # [B, E, H]
        # attn_normalizer = self._denominator(query_prime, key_prime)  # [N, B, H]
        # edge_attn_dem = attn_normalizer[end]  # [E, B, H]
        # edge_attn_dem = edge_attn_dem.permute(1, 0, 2)  # [B, E, H]
        # A_weight = edge_attn_num / edge_attn_dem  # [B, E, H]
        return z_output, A_weight

    def hyperbolic_mean(self,weight_tensor,value_tensor):
        eu_weighted_sum=torch.einsum('nhd,nhi->nd',value_tensor,weight_tensor)
        # 每个向量计算一个norm
        x_norm=self.tensor_minkouski_norm(eu_weighted_sum)
        # num_heads*batch_size
        coefficient = (1 /(self.curve ** 0.5)) / torch.sqrt(torch.abs(x_norm))
        hyper_weighted_sum=torch.einsum('nd,ni->nd',eu_weighted_sum,coefficient)
        return hyper_weighted_sum

    def hyper_agg(self,tensor):
        tensor=tensor.squeeze(1)
        # 得到tensor:num_heads*num_nodes*dim
        num_heads,num_nodes,dim=tensor.size(0), tensor.size(1),tensor.size(2)
        tensor=tensor.permute(1,0,2)
        weighted_tensor = torch.full((num_nodes,num_heads,1), 1/num_heads).to(tensor.device)
        tensor = self.hyperbolic_mean(weighted_tensor,tensor)
        # [B,N,D]
        return tensor

    def forward(self, z, adjs):
        B, N = z.size(0), z.size(1)
        z=z.squeeze()
        z = self.manifold.proj(self.manifold.expmap0(z, self.curve),self.curve)
        key=[]
        query=[]
        value=[]
        # batch_size*num_nodes*dim
        for k, layer in enumerate(self.Wk):
            key.append(layer(z))
        for k, layer in enumerate(self.Wq):
            query.append(layer(z))
        for k, layer in enumerate(self.Wv):
            value.append(layer(z))
        key=torch.stack(key,dim=0)
        #print(key.shape)
        query=torch.stack(query,dim=0)
        value=torch.stack(value,dim=0)
        # batch_size*num_nodes*num_heads*dim
        # 只有一个图，所以batch_size=1
        query = query.reshape(-1,
                              N,
                              self.num_heads,
                              self.out_channels)
        key = key.reshape(-1,
                          N,
                          self.num_heads,
                          self.out_channels)
        value = value.reshape(-1,
                              N,
                              self.num_heads,
                              self.out_channels)
        # batch_size*num_nodes*num_heads*dim
        z_next, weight = self.kernelized_gumbel_softmax(
                query,
                key,
                value,
                adjs[0],
            )
        z_next = self.hyper_agg(z_next)

        if self.use_edge_loss: # compute edge regularization loss on input adjacency
            row, col = adjs[0]
            link_loss = torch.mean(weight.to(row.device))
            return z_next, link_loss
        else:
            return z_next

class LightHyperFormer(nn.Module):
    '''
    NodeFormer model implementation
    return: predicted node labels, a list of edge losses at every layer
    '''
    def __init__(self,
                 manifold,
                 config,
                 temp,
                 in_channels,
                 hidden_channels,
                 out_channels,
                 num_layers,
                 num_heads,
                 dropout=0.0,
                 nb_random_features=30,
                 use_bn=True,
                 use_gumbel=True,
                 use_residual=True,
                 use_act=False,
                 use_jk=False,
                 nb_gumbel_sample=10,
                 rb_order=0,
                 rb_trans='sigmoid',
                 use_edge_loss=False,
                 curve=0.5
                 ):
        super(LightHyperFormer, self).__init__()
        self.manifold = getattr(manifolds, "Hyperboloid")()
        self.convs = nn.ModuleList()
        self.fcs = nn.ModuleList()
        self.fcs.append(HypLinear(self.manifold,
                                     in_channels,
                                     out_channels,
                                     curve,
                                     curve))
        for i in range(num_layers):
            self.convs.append(
                LightHyperFormerConv(config,
                                temp,
                                curve,
                                hidden_channels,
                                hidden_channels,
                                manifold=self.manifold,
                                num_heads=num_heads,
                                nb_random_features=nb_random_features,
                                use_gumbel=use_gumbel,
                                nb_gumbel_sample=nb_gumbel_sample,
                                rb_order=rb_order, rb_trans=rb_trans, use_edge_loss=use_edge_loss))
        self.fcs.append(HypLinear(self.manifold,
                                     in_channels,
                                     out_channels,
                                     curve,
                                     curve))
        self.dropout = dropout
        self.activation = F.elu
        self.use_residual = use_residual
        self.use_act = use_act
        self.use_jk = use_jk
        self.use_edge_loss = use_edge_loss

    def forward(self,
                x,
                adjs,
                tau=1.0):
        x = x.unsqueeze(0) # [B, N, D]
        link_loss_ = []
        z=x.squeeze()
        z=z.unsqueeze(0)
        for i, conv in enumerate(self.convs):
            z = conv(z, adjs)
        z=z.squeeze()
        x_out = self.fcs[-1](z).squeeze(0)
        return x_out,link_loss_