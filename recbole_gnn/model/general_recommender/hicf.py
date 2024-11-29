import torch
from recbole.utils import InputType
from recbole_gnn.model.general_recommender.hgcf import HGCF

class HICF(HGCF):
    input_type = InputType.PAIRWISE
    def __init__(self, config, dataset):
        super(HICF, self).__init__(config, dataset)
    def hratio(self, user, item):
        # Separate the first column from the rest of the tensor
        user0 = user[:, 0]
        item0 = item[:, 0]
        # Calculate the factor to multiply with the score
        factor = 1.0 / (user0 * item0)
        # Calculate the score using the Minkowski dot product and subtract user0 and item0
        score = 1 - self.manifold.minkowski_dot(user, item, keepdim=False) - user0 - item0
        # Multiply by the factor calculated earlier and negate the resulting tensor
        score = score * factor
        # Also add an extra dimension to obtain a 2D tensor
        return -score.view(-1, 1)
    def decode_pos(self, user_embedding, pos_embeddings):
        emb_in = user_embedding
        emb_out = pos_embeddings
        sqdist_h = self.manifold.sqdist(emb_in, emb_out, self.curve)
        w = self.hratio(emb_in, emb_out).sigmoid()
        return w, sqdist_h
    def decode_neg(self,
                   user,
                   pos_item,
                   neg_item,
                   user_all_embeddings,
                   item_all_embeddings):
        # Unpack anchor, negative, and positive indices
        # Get embeddings for the anchor and positive examples
        emb_anchor = user_all_embeddings[user]
        emb_pos = item_all_embeddings[pos_item]
        # Compute pairwise L2 distances between each negative example and the positive example,
        # here we can also compute the hyperbolic distance
        pos_neg_dist = ((item_all_embeddings[neg_item] - emb_pos) ** 2).sum(dim=-1).unsqueeze(-1)
        # Select the closest negative example (i.e., with the smallest L2 distance to the positive example)
        hard_idx = pos_neg_dist.min(dim=1).indices.view(-1, 1)
        hard_neg = torch.gather(neg_item.unsqueeze(-1).cuda(), 1, hard_idx).squeeze()
        emb_hard_neg = item_all_embeddings[hard_neg,]
        # Compute squared distance between anchor and closest negative example in hyperbolic space
        sqdist_neg = self.manifold.sqdist(emb_anchor, emb_hard_neg, self.curve)
        return sqdist_neg


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
        w, pos_scores = self.decode_pos(u_embeddings, pos_embeddings)
        neg_scores = self.decode_neg(user,
                                     pos_item,
                                     neg_item,
                                     user_all_embeddings,
                                     item_all_embeddings)
        loss_dist = pos_scores - neg_scores + self.margin * w
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
