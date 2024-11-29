import numpy as np
import torch
from recbole.model.init import xavier_uniform_initialization
from recbole.model.loss import BPRLoss, EmbLoss
from recbole.utils import InputType
from recbole_gnn.model.abstract_recommender import GeneralGraphRecommender
from recbole_gnn.model.general_recommender.lightgcn import LightGCN
from recbole_gnn.model.layers import LightGCNConv
from graph_transformer.nodeformer import NodeFormer
from graph_transformer.sgformer import SGFormer
from graph_transformer.hypformer import HypFormer
class GraphTransformer(LightGCN):
    input_type = InputType.PAIRWISE
    def __init__(self,
                 config,
                 dataset):
        super(LightGCN, self).__init__(config,
                                       dataset)
        # load parameters info
        self.config=config
        self.latent_dim = config['embedding_size']  # int type:the embedding size of lightGCN
        self.n_layers = config['n_layers']  # int type:the layer num of lightGCN
        self.reg_weight = config['reg_weight']  # float32 type: the weight decay for l2 normalization
        self.require_pow = config['require_pow']  # bool type: whether to require pow when regularization
        # define layers and loss
        self.user_embedding = torch.nn.Embedding(num_embeddings=self.n_users, embedding_dim=self.latent_dim)
        self.item_embedding = torch.nn.Embedding(num_embeddings=self.n_items, embedding_dim=self.latent_dim)
        if self.config['transformer_type']=='nodeformer':
           self.transformer = NodeFormer(config=self.config,
                                         in_channels=self.latent_dim,
                                         hidden_channels=self.latent_dim,
                                         out_channels=self.latent_dim,
                                         num_layers=self.n_layers,
                                         num_heads=self.config['num_heads'],
                                         nb_random_features=self.config['nb_random_features'],
                                         num_items=self.n_items,
                                         num_users=self.n_users,
                                         dropout=0.0,
                                         nb_gumbel_sample=10,
                                         rb_order=0,
                                         rb_trans='sigmoid',
                                         use_edge_loss=True)
        elif self.config['transformer_type']=='sgformer':
           self.transformer = SGFormer(in_channels=self.latent_dim,
                                       hidden_channels=self.latent_dim,
                                       out_channels=self.latent_dim,
                                       num_layers=self.n_layers,
                                       num_heads=self.config['num_heads'],
                                       alpha=0.5,
                                       dropout=0.5,
                                       use_bn=True,
                                       use_residual=True,
                                       use_weight=True,
                                       use_graph=True,
                                       use_act=False,
                                       graph_weight=self.config['graph_weight'],
                                       gnn=None,
                                       aggregate='add')

        elif self.config['transformer_type'] == 'hygformer':
            self.transformer = HypFormer(in_channels=self.latent_dim,
                                        hidden_channels=self.latent_dim,
                                        out_channels=self.latent_dim)

        self.mf_loss = BPRLoss()
        self.reg_loss = EmbLoss()
        # storage variables for full sort evaluation acceleration
        self.restore_user_e = None
        self.restore_item_e = None
        # parameters initialization
        self.apply(xavier_uniform_initialization)
        self.other_parameter_name = ['restore_user_e', 'restore_item_e']

    def forward(self):
        all_embeddings = self.get_ego_embeddings()
        if self.config['transformer_type']=='nodeformer':
           all_embeddings = self.transformer(all_embeddings, self.user_interations, self.item_interations)
        elif self.config['transformer_type']=='sgformer':
            all_embeddings = self.transformer(all_embeddings, self.edge_index, self.edge_weight )
        elif self.config['transformer_type'] == 'hygformer':
            all_embeddings = self.transformer(all_embeddings, self.edge_index)

        user_all_embeddings, item_all_embeddings = torch.split(all_embeddings, [self.n_users, self.n_items])

        return user_all_embeddings, item_all_embeddings
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
        # calculate BPR Loss
        pos_scores = torch.mul(u_embeddings, pos_embeddings).sum(dim=1)
        neg_scores = torch.mul(u_embeddings, neg_embeddings).sum(dim=1)
        mf_loss = self.mf_loss(pos_scores, neg_scores)
        # calculate regularization Loss
        #loss = mf_loss + self.config['lamda'] * sum(link_loss) / len(link_loss)
        return mf_loss
    def predict(self, interaction):
        user = interaction[self.USER_ID]
        item = interaction[self.ITEM_ID]
        user_all_embeddings, item_all_embeddings = self.forward()
        u_embeddings = user_all_embeddings[user]
        i_embeddings = item_all_embeddings[item]
        scores = torch.mul(u_embeddings, i_embeddings).sum(dim=1)
        return scores
    def full_sort_predict(self, interaction):
        user = interaction[self.USER_ID]
        if self.restore_user_e is None or self.restore_item_e is None:
          self.restore_user_e, self.restore_item_e= self.forward()
        # get user embedding from storage variable
        u_embeddings = self.restore_user_e[user]
        # dot with all item embedding to accelerate
        scores = torch.matmul(u_embeddings, self.restore_item_e.transpose(0, 1))
        if self.config['tail_analysis'] == True:
           return self.head_item, self.tail_item, scores.view(-1)
        elif self.config['popularity_analysis']== True:
            return self.rank1item,self.rank2item,self.rank3item,self.rank4item,self.rank5item,scores.view(-1)
        else:
            return scores.view(-1)