import torch
from recbole.utils import InputType
from hyperbolic_gnn.model.hgcn.layers.direction_diffusion import diffusion
from recbole_gnn.model.general_recommender.recformer import RecFormer
class HyperRecDiff(RecFormer):
    input_type = InputType.PAIRWISE
    def __init__(self, config, dataset):
        super(HyperRecDiff, self).__init__(config, dataset)
        self.diffusion=diffusion(config=config,
                                 interaction_matrix=self.interaction_matrix,
                                 n_users=self.n_users,
                                 n_items=self.n_items)

    def get_ego_embeddings(self):
        r"""Get the embedding of users and items and combine to an embedding matrix.
        Returns:
            Tensor of the embedding matrix. Shape of [n_items+n_users, embedding_dim]
        """
        user_embeddings = self.user_embedding.weight
        item_embeddings = self.item_embedding.weight
        norm_adj_matrix = self.norm_adj_matrix
        return user_embeddings,item_embeddings,norm_adj_matrix
    # forward用hyperbolic gcn进行encode
    def forward(self):
        if self.config['encoder']=='light_hyperbolic_trm':
            user_all_embeddings, item_all_embeddings = self.combine_embeddings()
        else:
            user_embeddings, item_embeddings, norm_adj_matrix = self.get_ego_embeddings()
            all_embeddings = torch.cat([user_embeddings, item_embeddings], dim=0)
            all_embeddings = self.encoder(all_embeddings, norm_adj_matrix)
            user_all_embeddings, item_all_embeddings = torch.split(all_embeddings, [self.n_users, self.n_items])
        return user_all_embeddings,item_all_embeddings
    def calculate_loss(self, interaction):
        # clear the storage variable when training
        if self.restore_user_e is not None or self.restore_item_e is not None:
            self.restore_user_e, self.restore_item_e = None, None
        user = interaction[self.USER_ID]
        pos_item = interaction[self.ITEM_ID]
        neg_item = interaction[self.NEG_ITEM_ID]
        # diffrec
        user_all_embeddings, item_all_embeddings= self.forward()
        # 对两组embeddings进行加噪，还原的训练
        diff_loss = self.diffusion(user_all_embeddings,item_all_embeddings,self.norm_adj_matrix)
        u_embeddings = user_all_embeddings[user]
        pos_embeddings = item_all_embeddings[pos_item]
        neg_embeddings = item_all_embeddings[neg_item]
        loss = self.decode(u_embeddings,
                           pos_embeddings,
                           neg_embeddings)
        final_loss = loss.sum()+diff_loss
        return final_loss
    def full_sort_predict(self, interaction):
        user = interaction[self.USER_ID]
        if self.restore_user_e is None or self.restore_item_e is None:
            self.restore_user_e, self.restore_item_e= self.forward()
        out= self.diffusion.p_sample_loop(self.restore_user_e, self.restore_item_e,self.norm_adj_matrix)
        user_embeddings, item_embeddings = torch.split(out, [self.n_users, self.n_items])
        u_embeddings = user_embeddings[user]
        scores = -self.manifold.hyper_dist(u_embeddings,
                                           item_embeddings,
                                           self.curve)
        return scores.view(-1)

