import numpy as np
import torch
import scipy.sparse as sp
from hyperbolic_gnn.model.hgcn.layers.lightbihyperformer import LightBiHyperFormer
from hyperbolic_gnn.model.hgcn.layers.hyp_layers import HGCN,LorentzBatchNorm,HypLinear
from recbole.model.abstract_recommender import GeneralRecommender
from recbole.model.init import xavier_uniform_initialization
from recbole.utils import InputType
import hyperbolic_gnn.model.hgcn.manifolds.hyperboloid as manifolds
from recbole.utils import init_seed
from graph_transformer.hypformer import HypFormer
init_seed(2024,True)
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


        if config['encoder']=='hypformer':
           self.gnn = HGCN( in_dim=self.latent_dim+1,
                            out_dim=self.latent_dim+1,
                            num_layers=config['gcn_layers'],
                            manifold=self.manifold,
                            curve=self.curve,
                            conv=config['gcn'])
           self.hyper_linear = HypLinear(self.manifold, self.latent_dim, self.latent_dim + 1, self.curve, self.curve)
        else:
            self.gnn = HGCN(in_dim=self.latent_dim,
                            out_dim=self.latent_dim,
                            num_layers=config['gcn_layers'],
                            manifold=self.manifold,
                            curve=self.curve,
                            conv=config['gcn'])

        self.ln=LorentzBatchNorm(self.manifold,
                                 self.latent_dim,
                                 self.curve)
        self.restore_user_e = None
        self.restore_item_e = None
        if self.config['encoder']=='light_hyperbolic_trm':
            self.encoder = LightBiHyperFormer(
                                         manifold=self.manifold,
                                         curve=self.curve,
                                         in_channels=self.latent_dim,
                                         hidden_channels=self.latent_dim,
                                         num_layers=self.n_layers,
                                         num_heads=self.num_heads,
                                         temp=self.temp,
                                         interaction=self.interaction_matrix,
                                         config=config,
                                         n_users=self.n_users,
                                         n_items=self.n_items)
        elif self.config['encoder']=='hypformer':
            self.encoder = HypFormer(in_channels=self.latent_dim,
                                     hidden_channels=self.latent_dim,
                                     out_channels=self.latent_dim,
                                     args=self.config)



        elif self.config['encoder']=='hgcn':
            self.encoder = HGCN(in_dim=self.latent_dim,
                                out_dim=self.latent_dim,
                                num_layers=config['gcn_layers'],
                                manifold=self.manifold,
                                curve=self.curve,
                                conv=config['encoder'])
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

    def hyperbolic_combine(self, tensor1, tensor2):
        if self.config['feat_extra']==True:
          tensor1=self.feat_extra(tensor1)
          tensor2=self.feat_extra(tensor2)
        tensor1_tan = self.manifold.logmap0(tensor1, c=self.curve)
        tensor2_tan = self.manifold.logmap0(tensor2, c=self.curve)
        tensor_tan=self.config['alpha']*tensor1_tan+(1-self.config['alpha'])*tensor2_tan
        tensor=self.manifold.proj(self.manifold.expmap0(tensor_tan, self.curve), self.curve)
        return tensor
    def combine_embeddings(self):
        user_embeddings, item_embeddings, norm_adj_matrix = self.get_ego_embeddings()
        if self.config['encoder']=='hypformer':
            all_embeddings=torch.cat([user_embeddings, item_embeddings], dim=0)
            all_embeddings=self.encoder(all_embeddings,norm_adj_matrix)
            trans_user_embeddings, trans_item_embeddings =torch.split(all_embeddings, [self.n_users, self.n_items])
        else:
            user_embeddings = self.manifold.proj(self.manifold.expmap0(user_embeddings, self.curve), self.curve)
            item_embeddings = self.manifold.proj(self.manifold.expmap0(item_embeddings, self.curve), self.curve)
            trans_user_embeddings, trans_item_embeddings = self.encoder(user_embeddings, item_embeddings, norm_adj_matrix)
        if self.config['no_gcn']==True:
            return trans_user_embeddings, trans_item_embeddings
        elif self.config['no_transformer'] == True:
            all_embeddings = torch.cat([user_embeddings, item_embeddings], dim=0)
            all_embeddings = self.gnn(all_embeddings, norm_adj_matrix)
            gcn_user_embeddings, gcn_item_embeddings = torch.split(all_embeddings, [self.n_users, self.n_items])
            return gcn_user_embeddings, gcn_item_embeddings
        else:
           all_embeddings = torch.cat([user_embeddings, item_embeddings], dim=0)
           if self.config['encoder']=='hypformer':
               all_embeddings=self.hyper_linear(all_embeddings)
           all_embeddings = self.gnn(all_embeddings, norm_adj_matrix)
           gcn_user_embeddings, gcn_item_embeddings = torch.split(all_embeddings, [self.n_users, self.n_items])
           user_all_embeddings=self.hyperbolic_combine(trans_user_embeddings,gcn_user_embeddings)
           item_all_embeddings = self.hyperbolic_combine(trans_item_embeddings, gcn_item_embeddings)
           return user_all_embeddings, item_all_embeddings

    def forward(self):

        if self.config['encoder']=='light_hyperbolic_trm':
            user_all_embeddings, item_all_embeddings = self.combine_embeddings()
        elif self.config['encoder']=='hypformer':
            user_all_embeddings, item_all_embeddings = self.combine_embeddings()
        else:
            user_embeddings, item_embeddings, norm_adj_matrix = self.get_ego_embeddings()
            all_embeddings = torch.cat([user_embeddings, item_embeddings], dim=0)
            all_embeddings = self.encoder(all_embeddings, norm_adj_matrix)
            user_all_embeddings, item_all_embeddings = torch.split(all_embeddings, [self.n_users, self.n_items])
        return user_all_embeddings,item_all_embeddings
    def decode(self, user_embedding,
               pos_embeddings,
               neg_embeddings):
        pos_score = self.manifold.pair_wize_hyper_dist(user_embedding, pos_embeddings, self.curve)
        neg_score = self.manifold.pair_wize_hyper_dist(user_embedding, neg_embeddings, self.curve)
        loss = pos_score - neg_score + self.margin
        loss[loss<0]=0
        loss=torch.sum(loss)
        return loss

    def find_value_positions(self,tensor, value):
        mask = torch.eq(tensor, value)
        positions = torch.nonzero(mask, as_tuple=False)
        return positions

    def extract_elements_by_positions(self,tensor, positions):
        """
        根据给定的 positions，从 tensor 中提取对应位置的元素。

        参数：
        - tensor: PyTorch张量，待提取的目标张量
        - positions: PyTorch张量，包含位置索引（一维或多维）

        返回：
        - extracted_elements: 从目标张量中提取的元素
        """
        # 将 positions 转换为一维索引
        if positions.dim() > 1:
            # 如果 positions 是二维张量（例如形状 [N, 1]），转换为一维
            positions = positions.view(-1)

        # 使用索引提取元素
        extracted_elements = tensor[positions]

        return extracted_elements
    def calculate_loss(self, interaction):
        positions=self.find_value_positions(self.user_src,4)

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
        if self.config['tail_analysis']==True:
            return self.head_item,self.tail_item,scores.view(-1)
        elif self.config['popularity_analysis']== True:
            return self.rank1item,self.rank2item,self.rank3item,self.rank4item,self.rank5item,scores.view(-1)
        else:
            return scores.view(-1)