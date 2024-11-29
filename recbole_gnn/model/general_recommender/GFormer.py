import numpy as np
import torch
import scipy.sparse as sp
from hyperbolic_gnn.model.hgcn.layers.bihyperformer import BiHyperFormer
from hyperbolic_gnn.model.hgcn.layers.lighthyperformer import LightHyperFormer
from hyperbolic_gnn.model.hgcn.layers.hyperformer import HyperFormer
from hyperbolic_gnn.model.hgcn.layers.lightbihyperformer import LightBiHyperFormer
from recbole_gnn.model.hyp_layers import SkipGCN
from recbole.model.abstract_recommender import GeneralRecommender
from recbole.model.init import xavier_uniform_initialization
from recbole.utils import InputType
import hyperbolic_gnn.model.hgcn.manifolds.hyperboloid as manifolds


class RecFormer(GeneralRecommender):
    input_type = InputType.PAIRWISE
    def __init__(self, config, dataset):
        super(RecFormer, self).__init__(config, dataset)
        self.config=config
        self.interaction_matrix = dataset.inter_matrix(form='coo').astype(np.float32)
        self.user_degree_count = torch.from_numpy(self.interaction_matrix.sum(axis=1)).to(self.device)
        self.item_degree_count = torch.from_numpy(self.interaction_matrix.sum(axis=0)).transpose(0, 1).to(self.device)
        self.norm_adj_matrix = self.get_norm_adj_mat(self.interaction_matrix, self.n_users,self.n_items).to(self.device)
        self.user_src, self.item_dst = dataset.get_interactions()
        self.margin=config['margin']
        self.latent_dim = config['embedding_size']  # int type:the embedding size of lightGCN
        self.n_layers = config['n_layers']  # int type:the layer num of lightGCN
        self.num_heads=config['num_heads']
        self.manifold_name=config['manifold']
        self.manifold = getattr(manifolds, "Hyperboloid")()
        # define layers and loss
        #self.curve = nn.Parameter(torch.tensor(config['curve']))
        self.curve =config['curve']
        self.temp=config['temp']
        self.user_embedding = torch.nn.Embedding(num_embeddings=self.n_users,
                                                 embedding_dim=self.latent_dim)
        self.item_embedding = torch.nn.Embedding(num_embeddings=self.n_items,
                                                 embedding_dim=self.latent_dim)


        self.gnn = SkipGCN(latent_dim=self.latent_dim,
                            num_layers=config['gcn_layers'],
                            manifold=self.manifold,
                            curve=self.curve,
                            conv=config['gcn'])


        self.restore_user_e = None
        self.restore_item_e = None
        if config['transformer_type']=='hyperbolic_trm' and config['graph_type']=='bi':
            self.transformer=BiHyperFormer(
                                         manifold=self.manifold,
                                         curve=self.curve,
                                         in_channels=self.latent_dim,
                                         hidden_channels=self.latent_dim,
                                         num_layers=self.n_layers,
                                         num_heads=self.num_heads,
                                         temp=self.temp,config=config)
        elif config['transformer_type']=='light_hyperbolic_trm' and config['graph_type']=='bi':
            self.transformer=LightBiHyperFormer(
                                         manifold=self.manifold,
                                         curve=self.curve,
                                         in_channels=self.latent_dim,
                                         hidden_channels=self.latent_dim,
                                         num_layers=self.n_layers,
                                         num_heads=self.num_heads,
                                         temp=self.temp,config=config)

        elif config['transformer_type']=='light_hyperbolic_trm' and config['graph_type']=='full':
            self.transformer=LightHyperFormer(
                                         manifold=self.manifold,
                                         curve=self.curve,
                                         in_channels=self.latent_dim,
                                         out_channels=self.latent_dim,
                                         hidden_channels=self.latent_dim,
                                         num_layers=self.n_layers,
                                         num_heads=self.num_heads,
                                         temp=self.temp,config=config)
        elif config['transformer_type']=='hyperbolic_trm' and config['graph_type']=='full':
            self.transformer=HyperFormer(
                                         manifold=self.manifold,
                                         curve=self.curve,
                                         in_channels=self.latent_dim,
                                         hidden_channels=self.latent_dim,
                                         num_layers=self.n_layers,
                                         num_heads=self.num_heads,
                                         temp=self.temp,config=config)

        # generate intermediate data
        # parameters initialization
        self.apply(xavier_uniform_initialization)
        self.other_parameter_name = ['restore_user_e', 'restore_item_e']

    def get_ego_embeddings(self):
        r"""Get the embedding of users and items and combine to an embedding matrix.
        Returns:
            Tensor of the embedding matrix. Shape of [n_items+n_users, embedding_dim]
        """
        user_embeddings = self.user_embedding.weight
        item_embeddings = self.item_embedding.weight
        norm_adj_matrix = self.norm_adj_matrix
        return user_embeddings,item_embeddings,norm_adj_matrix

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
        user_embeddings,item_embeddings,norm_adj_matrix = self.get_ego_embeddings()
        adjs = [torch.stack([self.user_src, self.item_dst])]
        user_embeddings = self.manifold.proj(self.manifold.expmap0(user_embeddings, self.curve), self.curve)
        item_embeddings = self.manifold.proj(self.manifold.expmap0(item_embeddings, self.curve), self.curve)

        if self.config['graph_type']=='bi':
           user_embeddings, item_embeddings,link_loss = self.transformer(user_embeddings, item_embeddings,adjs)
           all_embeddings = torch.cat([user_embeddings, item_embeddings], dim=0)
           #self.check_nan('all_embeddings',all_embeddings)
           if self.config['gcn_layers']>0:
             all_embeddings = self.gnn(all_embeddings, norm_adj_matrix)
             all_embeddings = self.manifold.proj(self.manifold.expmap0(all_embeddings, self.curve), self.curve)
           user_all_embeddings, item_all_embeddings = torch.split(all_embeddings, [self.n_users, self.n_items])
        else:
            all_embeddings = torch.cat([user_embeddings, item_embeddings], dim=0)
            all_embeddings,link_loss= self.transformer(all_embeddings,adjs)
            user_all_embeddings, item_all_embeddings = torch.split(all_embeddings, [self.n_users, self.n_items])
        #self.check_nan('user_all_embeddings', user_all_embeddings)
        #self.check_nan('item_all_embeddings', item_all_embeddings)
        return user_all_embeddings, item_all_embeddings

    def decode(self, user_embedding,
               pos_embeddings,
               neg_embeddings):
        pos_score = self.manifold.pair_wize_hyper_dist(user_embedding, pos_embeddings, self.curve)
        neg_score = self.manifold.pair_wize_hyper_dist(user_embedding, neg_embeddings, self.curve)
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
        loss = self.decode(u_embeddings,pos_embeddings,neg_embeddings)
        final_loss = loss.sum()
        return final_loss



    def check_nan(self,x_name,x):
        print(x_name,torch.any(torch.isnan(x)))

    def predict(self, interaction):
        user = interaction[self.USER_ID]
        item = interaction[self.ITEM_ID]
        user_all_embeddings, item_all_embeddings,_ = self.forward()
        u_embeddings = user_all_embeddings[user]
        i_embeddings = item_all_embeddings[item]
        scores = torch.mul(u_embeddings, i_embeddings).sum(dim=1)
        return scores

    def full_sort_predict(self, interaction):
        user = interaction[self.USER_ID]
        if self.restore_user_e is None or self.restore_item_e is None:
            self.restore_user_e, self.restore_item_e = self.forward()
        u_embeddings = self.restore_user_e[user]
        scores = -self.manifold.hyper_dist(u_embeddings, self.restore_item_e,self.curve)
        return scores.view(-1)


