import numpy as np
import torch
from recbole.model.loss import BPRLoss, EmbLoss
from recbole.model.init import xavier_uniform_initialization
from recbole.utils import InputType
import scipy.sparse as sp
from recbole_gnn.model.abstract_recommender import GeneralGraphRecommender
from hyperbolic_gnn.model.hgcn.layers.hyp_layers import HGCN
import hyperbolic_gnn.model.hgcn.manifolds as manifolds
class HGCF(GeneralGraphRecommender):
    input_type = InputType.PAIRWISE
    def __init__(self, config, dataset):
        super(HGCF, self).__init__(config, dataset)
        # load parameters info
        self.config=config
        self.inter_num=dataset.inter_num
        self.interaction_matrix = dataset.inter_matrix(form='coo').astype(np.float32)
        self.latent_dim = config['embedding_size']  # int type:the embedding size of lightGCN
        self.n_layers = config['gcn_layers']  # int type:the layer num of lightGCN
        self.margin=config['margin']
        self.curve=config['curve']
        self.manifold= getattr(manifolds, "Hyperboloid")()
        self.embedding = torch.nn.Embedding(num_embeddings=self.n_users+self.n_items,
                                            embedding_dim=self.latent_dim)
        self.embedding.state_dict()['weight'].uniform_(-config['scale'], config['scale'])
        self.embedding.weight = torch.nn.Parameter(self.manifold.expmap0(self.embedding.state_dict()['weight'], self.curve))
        self.embedding.weight = manifolds.ManifoldParameter(self.embedding.weight, True, self.manifold, self.curve)
        self.norm_adj_matrix = self.get_norm_adj_mat(self.interaction_matrix,
                                                     self.n_users,
                                                     self.n_items).to(self.device)
        self.gcn_conv = HGCN(in_dim=self.latent_dim,
                             out_dim=self.latent_dim,
                             num_layers=self.n_layers,
                             manifold=self.manifold,
                             curve=self.curve,
                             conv=config['conv'])
        self.restore_user_e = None
        self.restore_item_e = None
        if config['learner']=='adam':
           self.apply(xavier_uniform_initialization)
        self.other_parameter_name = ['restore_user_e', 'restore_item_e']
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
        return SparseL
    def forward(self):
        # all_embeddings：双曲空间上的向量
        all_embeddings,norm_adj_matrix = self.get_ego_embeddings()
        all_embeddings = self.manifold.proj(all_embeddings, c=self.curve)
        # all_embeddings在双曲空间上
        gcn_all_embeddings = self.gcn_conv(all_embeddings, norm_adj_matrix)
        user_all_embeddings, item_all_embeddings = torch.split(gcn_all_embeddings, [self.n_users, self.n_items])
        return user_all_embeddings, item_all_embeddings
    def get_ego_embeddings(self):
        r"""Get the embedding of users and items and combine to an embedding matrix.
        Returns:
            Tensor of the embedding matrix. Shape of [n_items+n_users, embedding_dim]
        """
        ego_embeddings = self.embedding.weight
        norm_adj_matrix = self.norm_adj_matrix
        return ego_embeddings,norm_adj_matrix
    def decode(self,
               user_embedding,
               pos_embeddings,
               neg_embeddings):
        pos_score = self.manifold.sqdist(user_embedding, pos_embeddings, self.curve)
        neg_score = self.manifold.sqdist(user_embedding, neg_embeddings, self.curve)
        loss = pos_score - neg_score + self.margin
        loss[loss<0]=0
        loss=torch.sum(loss)
        return loss
    def calculate_loss(self, interaction):
        # clear the storage variable when training
        if self.restore_user_e is not None or self.restore_item_e is not None:
            self.restore_user_e, self.restore_item_e = None, None
        user = interaction[self.USER_ID]
        pos_item = interaction[self.ITEM_ID]
        neg_item = interaction[self.NEG_ITEM_ID]
        user_all_embeddings, item_all_embeddings = self.forward()
        u_embeddings = user_all_embeddings[user]
        pos_embeddings = item_all_embeddings[pos_item]
        neg_embeddings = item_all_embeddings[neg_item]
        loss_dist = self.decode(u_embeddings, pos_embeddings, neg_embeddings)
        loss_dist[loss_dist < 0] = 0
        loss = torch.sum(loss_dist)
        return loss

    def full_sort_predict(self, interaction):
        user = interaction[self.USER_ID]
        if self.restore_user_e is None or self.restore_item_e is None:
            self.restore_user_e, self.restore_item_e = self.forward()
        u_embeddings = self.restore_user_e[user]
        scores = -self.manifold.hyper_dist(u_embeddings, self.restore_item_e,self.curve)
        if self.config['tail_analysis']==True:
            return self.head_item,self.tail_item,scores.view(-1)
        elif self.config['popularity_analysis']== True:
            return self.rank1item,self.rank2item,self.rank3item,self.rank4item,self.rank5item,scores.view(-1)
        else:
            return scores.view(-1)
