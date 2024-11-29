import torch
from torch import nn
import random
from recbole.model.abstract_recommender import SequentialRecommender
from recbole.model.layers import TransformerEncoder
from recbole.model.loss import BPRLoss
import torch.nn.functional as F
import pandas as pd

class dist_func(nn.Module):
    def __init__(self,feat_len):
        super(dist_func, self).__init__()
        self.act = nn.ReLU()
        self.fc1=nn.Linear(feat_len,feat_len)
        self.fc2 = nn.Linear(feat_len, feat_len)
    def forward(self, x):
        c=self.fc1(x)
        c=self.act(c)
        out=self.fc2(c)
        return out
class SASCTS(SequentialRecommender):

    r"""
    SASRec is the first sequential recommender based on self-attentive mechanism.

    NOTE:
        In the author's implementation, the Point-Wise Feed-Forward Network (PFFN) is implemented
        by CNN with 1x1 kernel. In this implementation, we follows the original BERT implementation
        using Fully Connected Layer to implement the PFFN.
    """

    def __init__(self, config, dataset):
        super(SASCTS, self).__init__(config, dataset)
        # load parameters info
        self.lambda_cts = 0.1
        self.config = config
        self.n_layers = config['n_layers']
        self.n_heads = config['n_heads']
        self.hidden_size = config['hidden_size']  # same as embedding_size
        self.inner_size = config['inner_size']  # the dimensionality in feed-forward layer
        self.hidden_dropout_prob = config['hidden_dropout_prob']
        self.attn_dropout_prob = config['attn_dropout_prob']
        self.hidden_act = config['hidden_act']
        self.layer_norm_eps = config['layer_norm_eps']
        self.initializer_range = config['initializer_range']
        self.loss_type = config['loss_type']
        # define layers and loss
        self.item_embedding = nn.Embedding(self.n_items, self.hidden_size, padding_idx=0)
        self.dist_func=dist_func(self.hidden_size)
        self.position_embedding = nn.Embedding(self.max_seq_length, self.hidden_size)
        self.trm_encoder = TransformerEncoder(
            n_layers=self.n_layers,
            n_heads=self.n_heads,
            hidden_size=self.hidden_size,
            inner_size=self.inner_size,
            hidden_dropout_prob=self.hidden_dropout_prob,
            attn_dropout_prob=self.attn_dropout_prob,
            hidden_act=self.hidden_act,
            layer_norm_eps=self.layer_norm_eps
        )
        self.LayerNorm = nn.LayerNorm(self.hidden_size, eps=self.layer_norm_eps)
        self.dropout = nn.Dropout(self.hidden_dropout_prob)
        if self.loss_type == 'BPR':
            self.loss_fct = BPRLoss()
        elif self.loss_type == 'CE':
            self.loss_fct = nn.CrossEntropyLoss()
        else:
            raise NotImplementedError("Make sure 'loss_type' in ['BPR', 'CE']!")
        self.batch_size = config['train_batch_size']
        self.mask_default = self.mask_correlated_samples(batch_size=self.batch_size)
        self.CrossEntropy = nn.CrossEntropyLoss()
        # parameters initialization
        self.apply(self._init_weights)
    def device_as(self, t1, t2):
        """
        Moves t1 to the device of t2
        """
        return t1.to(t2.device)


        #输入两个N*D的矩阵，拼接起来
    def calc_similarity_batch(self, a, b):
        representations = torch.cat([a, b], dim=0)
        #得到一个2N*2N的相似度矩阵
        return F.cosine_similarity(representations.unsqueeze(1), representations.unsqueeze(0), dim=2)



       #用于对比学习的loss
    def cts_loss_2(self, proj_1, proj_2, temp):
        #初始化一个对角矩阵，对角线元素是1，其他元素为0
        #2N*2N的对角矩阵
        mask = (~torch.eye(proj_1.shape[0] * 2, proj_1.shape[0] * 2, dtype=bool)).float()
        batch_size = proj_1.shape[0]
        #将输入进来的两组embedding进行标准化，有的论文中说可以不用标准化
        z_i = F.normalize(proj_1, p=2, dim=1)
        z_j = F.normalize(proj_2, p=2, dim=1)
        #见前面定义的相似度的计算
        similarity_matrix = self.calc_similarity_batch(z_i, z_j)
        #取上移batch_size的对角线
        sim_ij = torch.diag(similarity_matrix, batch_size)
        #取下移batch_size的对角线
        sim_ji = torch.diag(similarity_matrix, -batch_size)
        #两对角线都是正样本
        positives = torch.cat([sim_ij, sim_ji], dim=0)
        #对比学习的分子，正样本的值要越大越好
        nominator = torch.exp(positives / temp)
        #对比学习的分母，但是要mask掉对角线元素，然后任意两两不同的embedding要拉远
        denominator = self.device_as(mask, similarity_matrix) * torch.exp(similarity_matrix / temp)
        all_losses = -torch.log(nominator / torch.sum(denominator, dim=1))
        loss = torch.sum(all_losses) / (2 * batch_size)
        return loss

#跟cts_loss_2一致，但实现更加效率高
    def cts_loss(self, z_i, z_j, temp, batch_size): #B * D    B * D
        N = 2 * batch_size
        # 将输入的两个batch的数据拼接起来，得到一个2B * D的矩阵，D是向量的维度
        z = torch.cat((z_i, z_j), dim=0)   #2B * D

        sim = torch.mm(z, z.T) / temp   # 2B * 2B
        #两个不同的forward出来的结果，自己和平行宇宙的自己的相似度（需要最相似，才说明平行宇宙和该宇宙是稳定相似的）
        sim_i_j = torch.diag(sim, batch_size)    #B*1
        #跟上一个一样
        sim_j_i = torch.diag(sim, -batch_size)   #B*1
        #正样本，数目是batch的两倍
        positive_samples = torch.cat((sim_i_j, sim_j_i), dim=0).reshape(N, 1)
        if batch_size != self.batch_size:
            mask = self.mask_correlated_samples(batch_size)
        else:
            mask = self.mask_default
        #负样本
        #是去掉四个矩阵对角线元素后的矩阵
        negative_samples = sim[mask].reshape(N, -1)
        labels = torch.zeros(N).to(positive_samples.device).long()
        logits = torch.cat((positive_samples, negative_samples), dim=1)  # N * C
        return logits, labels

    def _mask_correlated_samples1(self, labels_dis):
        mask = torch.ones((labels_dis.shape[0], labels_dis.shape[1]),device=labels_dis.device)
        mask =torch.tensor(mask-labels_dis,dtype=bool)
        return mask

    def _mask_correlated_samples2(self, labels_dis):
        mask =torch.tensor(labels_dis,dtype=bool,device=labels_dis.device)
        return mask

    def cts_loss1(self, z_i, z_j, temp, batch_size):
        #print(z_i.shape)
        #print(z_j.shape)
        z_i=F.normalize(z_i)
        z_j=F.normalize(z_j)
        z_negative = torch.rand([int(batch_size *2), z_i.shape[1]], device=z_i.device)
        z_negative=F.normalize(z_negative)
        z_negative.requires_grad = True
        # 计算两个正例间的相似度
        # B*B
        cos_sim = torch.mm(z_i,z_j.T) / temp
        # 计算正例与随机噪声构造的负例间的相似度
        # B*K
        cos_sim_negative =torch.mm(z_i,z_negative.T) / temp
        # B*(B+K)
        cos_sim = torch.cat([cos_sim, cos_sim_negative], 1)
        labels_dis = torch.cat(
            [torch.eye(cos_sim.size(0), device=cos_sim.device), torch.zeros_like(cos_sim_negative)], -1)
        #alpha=cos_sim.quantile(q=0.7)
        weights = torch.where(cos_sim> 0.8, 0, 1)

        mask_weights = torch.eye(cos_sim.size(0), device=cos_sim.device) - torch.diag_embed(torch.diag(weights))
        weights = weights + torch.cat([mask_weights, torch.zeros_like(cos_sim_negative)], -1)
        filter_cos_sim =cos_sim * weights
        mask1 = self._mask_correlated_samples1(labels_dis)
        mask2 = self._mask_correlated_samples2(labels_dis)
        negative_samples = filter_cos_sim[mask1].reshape(batch_size, -1)
        positive_samples = filter_cos_sim[mask2].reshape(batch_size, -1)
        labels = torch.zeros(batch_size).to(positive_samples.device).long()
        logits = torch.cat((positive_samples, negative_samples), dim=1)  # N * C
        return logits, labels

#需要mask掉的是一个2batch_size*2batch_size的矩阵
#mask掉每个小矩阵的对角线元素
    def mask_correlated_samples(self, batch_size):
        N = 2 * batch_size
        mask = torch.ones((N, N), dtype=bool)
        mask = mask.fill_diagonal_(0)
        for i in range(batch_size):
            mask[i, batch_size + i] = 0
            mask[batch_size + i, i] = 0
        return mask

    def _init_weights(self, module):
        """ Initialize the weights """
        if isinstance(module, (nn.Linear, nn.Embedding)):
            # Slightly different from the TF version which uses truncated_normal for initialization
            # cf https://github.com/pytorch/pytorch/pull/5617
            module.weight.data.normal_(mean=0.0, std=self.initializer_range)
        elif isinstance(module, nn.LayerNorm):
            module.bias.data.zero_()
            module.weight.data.fill_(1.0)
        if isinstance(module, nn.Linear) and module.bias is not None:
            module.bias.data.zero_()
#sas是在transformer上做改进
    def _partial_shuffle(self,id):
        index = id.tolist()
        head = random.randint(0, len(index))
        tail = random.randint(head, len(index))
        a = index[head:tail]
        random.shuffle(a)
        index[head:tail] = a
        id = id[index]
        return id

    def forward(self, item_seq, item_seq_len):
        #先去进行positional encoding
        position_ids = torch.arange(item_seq.size(1), dtype=torch.long, device=item_seq.device)
        position_ids = position_ids.unsqueeze(0).expand_as(item_seq)
        position_embedding = self.position_embedding(position_ids)

        item_emb = self.item_embedding(item_seq)
    #每个item都和自己对应的position_embedding相加
        input_emb = item_emb + position_embedding
        input_emb = self.LayerNorm(input_emb)
        input_emb = self.dropout(input_emb)
    #将前面加进去的padding给mask掉
        extended_attention_mask = self.get_attention_mask(item_seq)
    #将一个batch的数据输入进去transformer，输出还是一样的数据
        trm_output = self.trm_encoder(input_emb, extended_attention_mask, output_all_encoded_layers=True)

        """
        num_layers*[batch_size,num_item,embedidng_size]的tensor
        即2048个batch
        每个batch里有50个64维的向量
        """

        output = trm_output[-1]
        output = self.gather_indexes(output, item_seq_len - 1)
        """
        每个batch取出最后一个embedding
        """
        return output  # [N D]



    def forward1(self, item_seq, item_seq_len):
        #先去进行positional encoding
        position_ids = torch.arange(item_seq.size(1), dtype=torch.long, device=item_seq.device)
        position_ids =self._partial_shuffle(position_ids)
        position_ids = position_ids.unsqueeze(0).expand_as(item_seq)
        position_embedding = self.position_embedding(position_ids)
        item_emb = self.item_embedding(item_seq)
    #每个item都和自己对应的position_embedding相加
        input_emb = item_emb + position_embedding
        input_emb = self.LayerNorm(input_emb)
        input_emb = self.dropout(input_emb)
    #将前面加进去的padding给mask掉
        extended_attention_mask = self.get_attention_mask(item_seq)
    #将一个batch的数据输入进去transformer，输出还是一样的数据
        trm_output = self.trm_encoder(input_emb, extended_attention_mask, output_all_encoded_layers=True)
    #将一个batch中，最后一个embedding拿出来作为output，
        output = trm_output[-1]
        output = self.gather_indexes(output, item_seq_len - 1)
        return output  # [N D]


#第一个看的函数
    def calculate_loss(self, interaction):
    #传入进来的是N * L的数据，N是一个batch的数据量，L是序列长度，但是是将短的全部补上padding之后的情况
    #ITEM_SEQ是序列实际的长度，即除开padding以后的长度
        item_seq = interaction[self.ITEM_SEQ]   #N * L
        item_seq_len = interaction[self.ITEM_SEQ_LEN]
    #将序列和序列长度传入forward函数中，forward的实现看前面
        seq_output = self.forward(item_seq, item_seq_len)  # N * D

        pos_items = interaction[self.POS_ITEM_ID]
        if self.loss_type == 'BPR':
            neg_items = interaction[self.NEG_ITEM_ID]
            pos_items_emb = self.item_embedding(pos_items)
            neg_items_emb = self.item_embedding(neg_items)
            pos_score = torch.sum(seq_output * pos_items_emb, dim=-1)  # [B]
            neg_score = torch.sum(seq_output * neg_items_emb, dim=-1)  # [B]
            loss = self.loss_fct(pos_score, neg_score)
        else:  # self.loss_type = 'CE'
            test_item_emb = self.item_embedding.weight
        #计算embedding和item的相似度
            logits = torch.matmul(seq_output, test_item_emb.transpose(0, 1))
            loss = self.loss_fct(logits, pos_items)

        #下面添加对比学习的loss
        #用于做对比学习的，需要再调用一次forward
        raw_seq_output = self.forward(item_seq, item_seq_len)

        raw_seq_output=self.dist_func(raw_seq_output)
        #用self的就够了，即再用forward跑一次，由于dropout的存在，这个cts_seq_output与raw_seq_output应该是不同的
        #由此，去拉进它和原来的embedding的距离，由此进行对比学习
        #self是不用对比学习
        if self.config['aug'] == 'self':
            cts_seq_output = self.forward1(item_seq, item_seq_len)
            cts_seq_output=self.dist_func(cts_seq_output)
        #用对比学习
        else:
            cts_aug, cts_aug_lengths = interaction['aug'], interaction['aug_lengths']
            cts_seq_output = self.forward1(cts_aug, cts_aug_lengths)
            cts_seq_output = self.dist_func(cts_seq_output)
        #即让raw_seq_output, cts_seq_output的距离尽可能小，然后推远负样本的距离，详见函数cts_loss_2
        cts_nce_logits, cts_nce_labels = self.cts_loss1(raw_seq_output, cts_seq_output, temp=1.0,
                                                        batch_size=item_seq_len.shape[0])

        nce_loss = self.loss_fct(cts_nce_logits, cts_nce_labels)
        #cts_loss = self.cts_loss_2(raw_seq_output, cts_seq_output, temp=1.0)
        loss += self.lambda_cts * nce_loss
        return loss

    def predict(self, interaction):
        item_seq = interaction[self.ITEM_SEQ]
        item_seq_len = interaction[self.ITEM_SEQ_LEN]
        test_item = interaction[self.ITEM_ID]
        seq_output = self.forward(item_seq, item_seq_len)
        test_item_emb = self.item_embedding(test_item)
        scores = torch.mul(seq_output, test_item_emb).sum(dim=1)  # [B]
        return scores

    def full_sort_predict(self, interaction):
        item_seq = interaction[self.ITEM_SEQ]
        item_seq_len = interaction[self.ITEM_SEQ_LEN]
        seq_output = self.forward(item_seq, item_seq_len)
        test_items_emb = self.item_embedding.weight
        scores = torch.matmul(seq_output, test_items_emb.transpose(0, 1))  # [B n_items]
        return scores