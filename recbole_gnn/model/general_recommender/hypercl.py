import torch
import torch.nn.functional as F
from recbole.utils import InputType
from recbole_gnn.model.general_recommender.hgcf import HGCF
from recbole_gnn.model.hyp_layers import HGCN,LorentzBatchNorm
# 再做一个离散版本
class HyperCL(HGCF):
    input_type = InputType.PAIRWISE
    def __init__(self, config, dataset):
        super(HyperCL, self).__init__(config, dataset)
        self.tau=config['tau']
        self.gcn_conv = HGCN(in_dim=self.latent_dim,
                             out_dim=self.latent_dim,
                             num_layers=1,
                             manifold=self.manifold,
                             curve=self.curve,
                             conv=config['conv'])
        self.cl_rate=config['cl_rate']
        self.eps=config['eps']
        self.norm=LorentzBatchNorm(manifold=self.manifold,dim=self.latent_dim,curve=self.curve)
    def add_hyper_noise(self,euc_noise,hyper_vector):
        # 将向量放到north pole ppoint的
        euc_noise[:, 0] = 0
        # 将上面的noise转移到hyper_vector的tangent space上
        # 推导：noise具有的信息更好？叠加到embedding所具有的信息更多？noise叠加的时间复杂度。
        hyper_noise_tangent = self.manifold.ptransp0(hyper_vector,euc_noise,torch.tensor(self.curve))
        hypernoise = self.manifold.expmap(hyper_noise_tangent,hyper_vector,self.curve)
        return hypernoise
    def forward(self, perturbed=False):
        all_embeddings,norm_adj_matrix = self.get_ego_embeddings()
        all_embeddings = self.manifold.proj(self.manifold.expmap0(all_embeddings, self.curve), self.curve)
        for layer_idx in range(self.n_layers):
            all_embs = self.gcn_conv(all_embeddings, norm_adj_matrix)
            if perturbed:
                random_noise = torch.rand_like(all_embeddings, device=all_embs.device)*torch.sign(all_embeddings)
                random_noise = F.normalize(random_noise, dim=-1) * self.eps
                all_embs=self.add_hyper_noise(random_noise,all_embs)
        user_all_embeddings, item_all_embeddings = torch.split(all_embs, [self.n_users, self.n_items])
        return user_all_embeddings, item_all_embeddings
    def calculate_cl_loss1(self, x1, x2):
        pos_score = -self.manifold.pair_wize_hyper_dist(x1,x2,self.curve)
        pos_score = torch.exp(pos_score / self.tau)
        # 下面的值过大
        ttl_score = -self.manifold.hyper_dist(x1, x2,self.curve)
        ttl_score = torch.exp(ttl_score / self.tau).sum(dim=1)
        return -torch.log(pos_score / ttl_score).sum()
    def calculate_cl_loss(self, x1, x2):
        pos_score = self.manifold.pair_wize_hyper_dist(x1, x2, self.curve)
        # 下面的值过大
        ttl_score = self.manifold.hyper_dist(x1, x2, self.curve)
        m_score = pos_score- ttl_score+ self.margin
        m_score[m_score<0]=0
        return torch.sum(m_score)
    def calculate_loss(self, interaction):
        loss = super().calculate_loss(interaction)
        user = interaction[self.USER_ID]
        pos_item = interaction[self.ITEM_ID]
        perturbed_user_embs_1, perturbed_item_embs_1 = self.forward(perturbed=True)
        perturbed_user_embs_2, perturbed_item_embs_2 = self.forward(perturbed=True)
        user_cl_loss = self.calculate_cl_loss(perturbed_user_embs_1[user], perturbed_user_embs_2[user])
        item_cl_loss = self.calculate_cl_loss(perturbed_item_embs_1[pos_item], perturbed_item_embs_2[pos_item])
        return loss + self.cl_rate * (user_cl_loss + item_cl_loss)
    def get_ego_embeddings(self):
        ego_embeddings = self.embedding.weight
        norm_adj_matrix = self.norm_adj_matrix
        return ego_embeddings,norm_adj_matrix
    def decode(self, user_embedding,
               pos_embeddings,
               neg_embeddings):
        pos_score = self.manifold.pair_wize_hyper_dist(user_embedding, pos_embeddings, self.curve)
        neg_score = self.manifold.pair_wize_hyper_dist(user_embedding, neg_embeddings, self.curve)
        loss = pos_score - neg_score + self.margin
        loss[loss<0]=0
        loss=torch.sum(loss)
        return loss
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
            self.restore_user_e, self.restore_item_e = self.forward()
        u_embeddings = self.restore_user_e[user]
        scores = -self.manifold.hyper_dist(u_embeddings, self.restore_item_e,self.curve)
        if self.config['tail_analysis'] == True:
           return self.head_item, self.tail_item, scores.view(-1)
        else:
            return scores.view(-1)