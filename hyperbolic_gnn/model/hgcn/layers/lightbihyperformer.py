import torch
import torch.nn as nn
import torch.nn.functional as F
import scipy.sparse as sp
from scipy.sparse import coo_matrix
from hyperbolic_gnn.model.hgcn.layers.hyp_layers import HypLinear,HypAct,LorentzBatchNorm
BIG_CONSTANT = 1e8
import math
import numpy as np
from recbole.utils import init_seed

init_seed(2024,True)
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

class LightBiHyperFormerConv(nn.Module):
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
                 n_users,
                 n_items,
                 config
                 ):
        super(LightBiHyperFormerConv, self).__init__()
        self.n_users=n_users
        self.n_items=n_items
        self.manifold=manifold
        self.in_channels=in_channels
        self.curve=curve
        self.num_heads = num_heads
        self.Wk=nn.ModuleList()
        self.Wq=nn.ModuleList()
        self.Wv=nn.ModuleList()
        self.config=config
        self.nb_random_features=64
        self.tau = config['temp']
        for i in range(self.num_heads):
            self.Wk.append(HypLinear(manifold,in_channels,out_channels,curve,curve))
            self.Wq.append(HypLinear(manifold,in_channels,out_channels,curve,curve))
            self.Wv.append(HypLinear(manifold,in_channels,out_channels,curve,curve))
        self.out_channels = out_channels
        self.temp=temp
        self.layer_norm = LorentzBatchNorm(self.manifold, in_channels, self.curve)
    def hyper_agg(self,tensor):
        # 得到tensor:num_heads*num_nodes*dim
        num_heads,num_nodes,dim=tensor.size(0), tensor.size(1),tensor.size(2)
        tensor=tensor.permute(1,0,2)
        weighted_tensor = torch.full((num_nodes,num_heads,1), 1/num_heads).to(tensor.device)
        tensor = self.manifold.hyperbolic_mean(weighted_tensor,tensor,self.curve)
        # [B,N,D]
        return tensor
    def _create_projection_matrix(self,
                                  m,
                                  d,
                                  seed=0,
                                  scaling=0,
                                  struct_mode=False):
        # 构造一个随机映射的矩阵
        # d太小可能不太好
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
                                       numerical_stabilizer=0.00000001):
        data_time=data[...,:,0].unsqueeze(-1)
        data_space=data[...,1:]
        # 根号m分之一
        # m挺关键
        m = 1.0 / torch.sqrt(torch.tensor(projection_matrix.shape[0], dtype=torch.float32)).to(data.device)
        # 在输入矩阵上乘以随机形成的矩阵，得到x~
        wx = torch.exp(torch.einsum("...hd,dm->...hm", data_space, projection_matrix)/torch.sqrt(torch.tensor(self.tau)))
        # data_dash:batch*num_nodes*num_heads*random_feats
        K = 1/self.curve
        front =(torch.exp( (K/self.tau-torch.square((data_time))/self.tau )/2.0)) * m+numerical_stabilizer
        #self.check_zero(front)
        data_dash =  torch.einsum('jkl,jkm->jkl', wx, front)
        return data_dash
    def kernelized_gumbel_softmax(self,
                                  query,
                                  key,
                                  value,
                                  ):
        seed = torch.ceil(torch.abs(torch.sum(query) * BIG_CONSTANT)).to(torch.int32)
        projection_matrix = self._create_projection_matrix(self.in_channels-1,
                                                           self.nb_random_features,
                                                           seed=seed
                                                           ).to('cuda')
        # 将Q和K随机映射
        query_prime = self._softmax_kernel_transformation(
                                                          query,
                                                          projection_matrix)
        # [H,N,M]
        key_prime = self._softmax_kernel_transformation(
                                                        key,
                                                        projection_matrix)
        # [H,N,M]
        # self.check_nan('key_prime',key_prime)
        # 先计算QV
        last_ = torch.einsum("hnm,hnd->hmd", key_prime, value)  # [H,B,M,D]
        # self.check_nan('last_', last_)
        # 将QV乘上去
        z_output = torch.einsum("hnm,hmd->hnd", query_prime, last_) # [H,B,N,D]
        # 计算双曲加权平均
        # hyper_coefficient/z_output是nan
        hyper_coefficient=(1 /(self.curve ** 0.5)) /torch.sqrt(torch.abs(self.manifold.tensor_minkouski_norm(z_output)+1e-7))
        # self.check_nan('hyper_coefficient', hyper_coefficient)
        z_output = torch.einsum('hnd,hnq->hnd', z_output,hyper_coefficient) #[H,B,N,D]
        return z_output
    def get_norm_adj_mat(self,
                         interaction_matrix,
                         n_users=None,
                         n_items=None):
        if n_users == None or n_items == None:
            n_users, n_items = interaction_matrix.shape
        A = sp.dok_matrix((n_users + n_items, n_users + n_items), dtype=np.float32)
        inter_M = interaction_matrix
        inter_M_t = interaction_matrix.transpose()
        data_dict = dict(zip(zip(inter_M.row, inter_M.col + n_users), [1] * inter_M.nnz))
        data_dict.update(dict(zip(zip(inter_M_t.row + n_users, inter_M_t.col), [1] * inter_M_t.nnz)))
        A._update(data_dict)
        # norm adj matrix
        sumArr = (A > 0).sum(axis=1)
        # add epsilon to avoid divide by zero Warning
        diag = np.array(sumArr.flatten())[0] + 1e-7
        diag = np.power(diag, -0.5)
        D = sp.diags(diag)
        L = D * A * D
        # covert norm_adj matrix to tensor
        L = sp.coo_matrix(L)
        row = L.row
        col = L.col
        i = torch.LongTensor([row, col])
        data = torch.FloatTensor(L.data)
        SparseL = torch.sparse.FloatTensor(i, data, torch.Size(L.shape))
        print('interaction_matrix', SparseL)
        return SparseL

    def dist_softmax(self,
                     query,
                     key,
                     value,
                     adjs=None,
                     ):

        att=F.softmax(self.manifold.minkowski_tensor_dot(query,key)/self.temp,dim=-1)
        weighted_sum = torch.einsum("hui,hid->hud", att, value)
        norm=torch.abs(self.manifold.tensor_minkouski_norm(weighted_sum))+1e-7
        norm_=1/(norm**(1/2))
        value=weighted_sum * norm_ * (1 / (self.curve**(1/2)))
        if adjs!=None:
           src_embedding = query[:, adjs.row, :]
           dst_embedding = key[:, adjs.col, :]
           weight=self.manifold.pair_wize_hyper_dist(src_embedding, dst_embedding, self.curve).squeeze()
           return value,weight
        else:
            return value
    def check_nan(self,x_name,x):
        print(x_name,torch.any(torch.isnan(x)))
    def forward(self,u,i,adjs):
        # u,i=torch.split(self.layer_norm(torch.cat([u, i], dim=0)),[self.n_users, self.n_items])
        # 三个线性层
        # 从batch_size*head.dim到batch_size#head*dim最后到head*batch_size*dim
        # head*batch_size*dim]
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
        if self.config['simplify']==True:
          # 双曲空间上的三个向量
          _u_value=self.kernelized_gumbel_softmax(u_query,i_key,i_value)
          _i_value=self.kernelized_gumbel_softmax(i_query,u_key,u_value)
        else:
            _u_value = self.dist_softmax(u_query, i_key, i_value)
            _i_value = self.dist_softmax(i_query, u_key, u_value)
        num_heads=u_value.shape[0]
        dim=u_value.shape[2]
        u_value=self.manifold.mobius_add(u_value.reshape(-1,dim),_u_value.reshape(-1,dim),self.curve)
        u_value=u_value.reshape(num_heads,-1,dim)
        i_value=self.manifold.mobius_add(i_value.reshape(-1,dim),_i_value.reshape(-1,dim),self.curve)
        i_value = i_value.reshape(num_heads, -1, dim)
        final_u_value=self.hyper_agg(u_value)
        final_i_value=self.hyper_agg(i_value)
        return final_u_value,final_i_value
class LightBiHyperFormer(nn.Module):
    def __init__(self,
                 manifold,
                 curve,
                 in_channels,
                 hidden_channels,
                 num_layers,
                 num_heads,
                 temp,
                 interaction,
                 config,
                 n_users,
                 n_items
                 ):
        super(LightBiHyperFormer, self).__init__()
        self.n_users=n_users
        self.n_items=n_items
        self.manifold=manifold
        self.curve=curve
        self.interaction=interaction
        self.bns= nn.ModuleList()
        self.fcs = nn.ModuleList()
        self.convs=nn.ModuleList()
        self.fcs.append(HypLinear(
                                  manifold,
                                  in_channels,
                                  hidden_channels,
                                  curve,
                                  curve
                                  ))
        self.layer_norm=LorentzBatchNorm(self.manifold,hidden_channels,self.curve)
        for i in range(num_layers):
            self.convs.append(
                LightBiHyperFormerConv(
                    self.manifold,
                    self.curve,
                    in_channels,
                    hidden_channels,
                    num_heads,
                    temp,
                    n_users,
                    n_items,
                    config=config
                ))
        self.activation =HypAct(manifold=self.manifold,c_in=self.curve,c_out=self.curve)

    def check_nan(self,x_name,x):
        print(x_name,torch.any(torch.isnan(x)))

    def forward(self, u,i,adjs):
        link_loss_ = []
        for k, conv in enumerate(self.convs):
            u,i= conv(u,i,self.interaction)
            all_embeddings = torch.cat([u, i], dim=0)
            all_embeddings = self.layer_norm(all_embeddings)
            u,i=torch.split(all_embeddings, [self.n_users, self.n_items])
        return u,i