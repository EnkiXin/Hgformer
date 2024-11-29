from recbole.utils import InputType
from recbole_gnn.model.general_recommender.hgcf import HGCF
import torch
class HRCF(HGCF):
    input_type = InputType.PAIRWISE
    def __init__(self, config, dataset):
        super(HRCF, self).__init__(config, dataset)
        # load parameters info
    def geometric_regularizer(self, embeddings):
        embeddings_tan = self.manifold.logmap0(embeddings, c=self.curve)
        item_mean_norm = ((1e-6 + embeddings_tan.pow(2).sum(dim=1)).mean()).sqrt()
        return 1.0 / item_mean_norm
    def decode(self, user_embedding,
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
        loss = self.decode(u_embeddings, pos_embeddings, neg_embeddings)
        gr_loss = self.geometric_regularizer(item_all_embeddings)
        return loss+gr_loss


