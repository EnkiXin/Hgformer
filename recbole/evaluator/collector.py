# @Time   : 2021/6/23
# @Author : Zihan Lin
# @Email  : zhlin@ruc.edu.cn

# UPDATE
# @Time   : 2021/7/18
# @Author : Zhichao Feng
# @email  : fzcbupt@gmail.com

"""
recbole.evaluator.collector
################################################
"""
from recbole.evaluator.register import Register
import torch
import copy
class DataStruct(object):
    def __init__(self):
        self._data_dict = {}
    def __getitem__(self, name: str):
        return self._data_dict[name]
    def __setitem__(self, name: str, value):
        self._data_dict[name] = value
    def __delitem__(self, name: str):
        self._data_dict.pop(name)
    def __contains__(self, key: str):
        return key in self._data_dict
    def get(self, name: str):
        if name not in self._data_dict:
            raise IndexError("Can not load the data without registration !")
        return self[name]
    def set(self, name: str, value):
        # 在字典中添加一个名为name，值为value的数据
        self._data_dict[name] = value
    def update_tensor(self, name: str, value: torch.Tensor):
        if name not in self._data_dict:
            self._data_dict[name] = value.cpu().clone().detach()
        else:
            if not isinstance(self._data_dict[name], torch.Tensor):
                raise ValueError("{} is not a tensor.".format(name))
            self._data_dict[name] = torch.cat((self._data_dict[name], value.cpu().clone().detach()), dim=0)

    def __str__(self):
        data_info = '\nContaining:\n'
        for data_key in self._data_dict.keys():
            data_info += data_key + '\n'
        return data_info


class Collector(object):
    """The collector is used to collect the resource for evaluator.
        As the evaluation metrics are various, the needed resource not only contain the recommended result
        but also other resource from data and model. They all can be collected by the collector during the training
        and evaluation process.
        This class is only used in Trainer.
    """
    def __init__(self, config):
        self.config = config
        self.data_struct = DataStruct()
        self.register = Register(config)
        self.full = ('full' in config['eval_args']['mode'])
        self.topk = self.config['topk']
        self.device = self.config['device']

    def data_collect(self, train_data):
        """ Collect the evaluation resource from training data.
            Args:
                train_data (AbstractDataLoader): the training dataloader which contains the training data.
        """
        if self.register.need('data.num_items'):
            item_id = self.config['ITEM_ID_FIELD']
            self.data_struct.set('data.num_items', train_data.dataset.num(item_id))
        if self.register.need('data.num_users'):
            user_id = self.config['USER_ID_FIELD']
            self.data_struct.set('data.num_users', train_data.dataset.num(user_id))
        if self.register.need('data.count_items'):
            self.data_struct.set('data.count_items', train_data.dataset.item_counter)
        if self.register.need('data.count_users'):
            self.data_struct.set('data.count_items', train_data.dataset.user_counter)

    def _average_rank(self, scores):
        """Get the ranking of an ordered tensor, and take the average of the ranking for positions with equal values.
        Args:
            scores(tensor): an ordered tensor, with size of `(N, )`
        Returns:
            torch.Tensor: average_rank
        Example:
            >>> average_rank(tensor([[1,2,2,2,3,3,6],[2,2,2,2,4,5,5]]))
            tensor([[1.0000, 3.0000, 3.0000, 3.0000, 5.5000, 5.5000, 7.0000],
            [2.5000, 2.5000, 2.5000, 2.5000, 5.0000, 6.5000, 6.5000]])
        Reference:
            https://github.com/scipy/scipy/blob/v0.17.1/scipy/stats/stats.py#L5262-L5352
        """
        length, width = scores.shape
        true_tensor = torch.full((length, 1), True, dtype=torch.bool, device=self.device)
        obs = torch.cat([true_tensor, scores[:, 1:] != scores[:, :-1]], dim=1)
        # bias added to dense
        bias = torch.arange(0, length, device=self.device).repeat(width).reshape(width, -1). \
            transpose(1, 0).reshape(-1)
        dense = obs.view(-1).cumsum(0) + bias
        # cumulative counts of each unique value
        count = torch.where(torch.cat([obs, true_tensor], dim=1))[1]
        # get average rank
        avg_rank = .5 * (count[dense] + count[dense - 1] + 1).view(length, -1)
        return avg_rank
    def eval_batch_collect(
        self,
            scores_tensor: torch.Tensor,
            interaction,
            positive_u: torch.Tensor,
            positive_i: torch.Tensor
    ):
        """ Collect the evaluation resource from batched eval data and batched model output.
            Args:
                scores_tensor (Torch.Tensor): the output tensor of model with the shape of `(N, )`
                interaction(Interaction): batched eval data.
                positive_u(Torch.Tensor): the row index of positive items for each user.
                positive_i(Torch.Tensor): the positive item id for each user.
        """
        if self.register.need('rec.items'):
            # get topk
            _, topk_idx = torch.topk(scores_tensor, max(self.topk), dim=-1)  # n_users x k
            self.data_struct.update_tensor('rec.items', topk_idx)
        if self.register.need('rec.topk'):
            # torch.topk：选出前k大的数值，这里只需要位置，位置就是对应item的id
            _, topk_idx = torch.topk(scores_tensor, max(self.topk), dim=-1)  # n_users x k
            # 产生一个与scores_tensor相同大小，全0的矩阵，并将该矩阵中，user确实交互过的item的位置的值变成1
            pos_matrix = torch.zeros_like(scores_tensor, dtype=torch.int)
            pos_matrix[positive_u, positive_i] = 1
            # pos_len_list：每个user一共有多少个正样本
            pos_len_list = pos_matrix.sum(dim=1, keepdim=True)
            num_zeros = torch.sum(pos_len_list == 0).item()
            # 在pos_matrix中，正样本位置是1，负样本位置是0，
            # 下面是将topk_idx中所预测的正样本的位置的值选出来，如果是1就代表预测对了，如果是0就代表预测错了
            pos_idx = torch.gather(pos_matrix, dim=1, index=topk_idx)
            result = torch.cat((pos_idx, pos_len_list), dim=1)
            # 在data_struct中添加一个数据，名字交rec.topk，值为result
            self.data_struct.update_tensor('rec.topk', result)
            # 这里需要在self.data_struct中写一个新的一组数据

        if self.register.need('rec.meanrank'):
            desc_scores, desc_index = torch.sort(scores_tensor, dim=-1, descending=True)
            # get the index of positive items in the ranking list
            pos_matrix = torch.zeros_like(scores_tensor)
            pos_matrix[positive_u, positive_i] = 1
            pos_index = torch.gather(pos_matrix, dim=1, index=desc_index)

            avg_rank = self._average_rank(desc_scores)
            pos_rank_sum = torch.where(pos_index == 1, avg_rank, torch.zeros_like(avg_rank)).sum(dim=-1, keepdim=True)

            pos_len_list = pos_matrix.sum(dim=1, keepdim=True)
            user_len_list = desc_scores.argmin(dim=1, keepdim=True)
            result = torch.cat((pos_rank_sum, user_len_list, pos_len_list), dim=1)
            self.data_struct.update_tensor('rec.meanrank', result)
        if self.register.need('rec.score'):
            self.data_struct.update_tensor('rec.score', scores_tensor)
        if self.register.need('data.label'):
            self.label_field = self.config['LABEL_FIELD']
            self.data_struct.update_tensor('data.label', interaction[self.label_field].to(self.device))
    def longtail_analysis(
        self,
            tail_item,
            head_item,
            scores_tensor: torch.Tensor,
            interaction,
            positive_u: torch.Tensor,
            positive_i: torch.Tensor
    ):
        # torch.topk：选出前k大的数值，这里只需要位置，位置就是对应item的id
        # 在一个batch的items中选出tail的部分和head的部分
        # 在总的score矩阵中选出tail的评分和head的评分
        # scores:[n_batch_u,n_all_items]
        # _, topk_idx = torch.topk(scores_tensor, max(self.topk), dim=-1)
        i_num_zeros = torch.sum(positive_i == 0).item()
        tail_scores=scores_tensor[:,tail_item]
        head_scores=scores_tensor[:,head_item]
        # 下面的tail_topk_idx，head_topk_idx都是对应scores中的位置
        _, tail_topk_idx = torch.topk(tail_scores, max(self.topk), dim=-1)
        _, head_topk_idx = torch.topk(head_scores, max(self.topk), dim=-1)
        pos_matrix = torch.zeros_like(scores_tensor, dtype=torch.int)
        pos_matrix[positive_u, positive_i] = 1
        tail_pos_matrix=pos_matrix[:,tail_item]
        head_pos_matrix = pos_matrix[:, head_item]
        # pos_len_list：每个user一共有多少个正样本
        # len(tail_pos_len_list)=num_batch_users
        tail_pos_len_list = tail_pos_matrix.sum(dim=1, keepdim=True)
        tail_non_zero_positions = torch.nonzero(tail_pos_len_list != 0, as_tuple=False)[:, 0]
        tail_pos_len_list=tail_pos_len_list[tail_non_zero_positions]
        head_pos_len_list = head_pos_matrix.sum(dim=1, keepdim=True)
        head_non_zero_positions=torch.nonzero(head_pos_len_list != 0, as_tuple=False)[:, 0]
        head_pos_len_list = head_pos_len_list[head_non_zero_positions]
        # 在pos_matrix中，正样本位置是1，负样本位置是0，下面是将topk_idx中所预测的正样本的位置的值选出来，如果是1就代表预测对了，如果是0就代表预测错了
        tail_pos_idx = torch.gather(tail_pos_matrix, dim=1, index=tail_topk_idx)
        head_pos_idx = torch.gather(head_pos_matrix, dim=1, index=head_topk_idx)
        tail_pos_idx =tail_pos_idx[tail_non_zero_positions]
        head_pos_idx =head_pos_idx[head_non_zero_positions]
        tail_result = torch.cat((tail_pos_idx, tail_pos_len_list), dim=1)
        head_result=torch.cat((head_pos_idx, head_pos_len_list), dim=1)

        # self.get_rank_result(pos_matrix,topk_idx,rank_items)
        # 在data_struct中添加一个数据，名字交rec.topk，值为result
        self.data_struct.update_tensor('rec.tailtopk', tail_result)
        self.data_struct.update_tensor('rec.headtopk', head_result)

        # 这里需要在self.data_struct中写一个新的一组数据

    def get_result(self,pos_matrix,topk_idx):
        # pos_matrix:正样本是1，负样本是0（ground truth）
        # topk_idx：评分最高的10个item的index
        # pos_len_list：计算正样本总数
        pos_len_list = pos_matrix.sum(dim=1, keepdim=True)
        non_zero_positions = torch.nonzero(pos_len_list != 0, as_tuple=False)[:, 0]
        # 只取非0行
        pos_len_list = pos_len_list[non_zero_positions]
        pos_idx = torch.gather(pos_matrix, dim=1, index=topk_idx)
        #print(pos_idx.shape)
        pos_idx = pos_idx[non_zero_positions]
        result = torch.cat((pos_idx, pos_len_list), dim=1)
        return result

    def get_rank_result(self,pos_matrix,topk_idx,rank_items):
        # 计算对应rank的正items的个数
        pos_rank_matrix=pos_matrix[:,rank_items].to(self.config['device'])
        pos_len_list = pos_rank_matrix.sum(dim=1, keepdim=True).to(self.config['device'])
        #non_zero_positions = torch.nonzero(pos_len_list != 0, as_tuple=False)[:, 0]
        # 只取非0行
        #pos_len_list = pos_len_list[non_zero_positions]
        condition_mask = torch.isin(topk_idx.to(self.config['device']), rank_items.to(self.config['device']))
        # 将 topk_idx 中不满足条件的索引替换为一个-1
        topk_idx_masked = torch.where(condition_mask, topk_idx, -1)
        pos_idx = torch.gather(pos_matrix, dim=1, index=topk_idx_masked.clamp(min=0))
        pos_idx = torch.where(topk_idx_masked == -1, torch.tensor(0), pos_idx)
        #pos_idx = pos_idx[non_zero_positions]
        result = torch.cat((pos_idx, pos_len_list), dim=1)
        return result


    def rank_tail_analysis(
        self,
            rank1item,
            rank2item,
            rank3item,
            rank4item,
            rank5item,
            scores_tensor: torch.Tensor,
            interaction,
            positive_u: torch.Tensor,
            positive_i: torch.Tensor
    ):
        # torch.topk：选出前k大的数值，这里只需要位置，位置就是对应item的id
        # 在一个batch的items中选出tail的部分和head的部分
        # 在总的score矩阵中选出tail的评分和head的评分
        # scores:[n_batch_u,n_all_items]
        i_num_zeros = torch.sum(positive_i == 0).item()
        rank1_scores = scores_tensor[:, rank1item]
        rank2_scores = scores_tensor[:, rank2item]
        rank3_scores = scores_tensor[:, rank3item]
        rank4_scores = scores_tensor[:, rank4item]
        rank5_scores = scores_tensor[:, rank5item]
        _, topk_idx = torch.topk(scores_tensor, max(self.topk), dim=-1)
        # 下面的tail_topk_idx，head_topk_idx都是对应scores中的位置
        _, rank1_topk_idx = torch.topk(rank1_scores, max(self.topk), dim=-1)
        _, rank2_topk_idx = torch.topk(rank2_scores, max(self.topk), dim=-1)
        _, rank3_topk_idx = torch.topk(rank3_scores, max(self.topk), dim=-1)
        _, rank4_topk_idx = torch.topk(rank4_scores, max(self.topk), dim=-1)
        _, rank5_topk_idx = torch.topk(rank5_scores, max(self.topk), dim=-1)



        pos_matrix = torch.zeros_like(scores_tensor, dtype=torch.int)
        pos_matrix[positive_u, positive_i] = 1

        rank1_pos_matrix=pos_matrix[:,rank1item]
        rank2_pos_matrix = pos_matrix[:, rank2item]
        rank3_pos_matrix = pos_matrix[:, rank3item]
        rank4_pos_matrix = pos_matrix[:, rank4item]
        rank5_pos_matrix=pos_matrix[:,rank5item]

        rank1_result = self.get_result(rank1_pos_matrix, rank1_topk_idx)
        rank2_result = self.get_result(rank2_pos_matrix, rank2_topk_idx)
        rank3_result = self.get_result(rank3_pos_matrix, rank3_topk_idx)
        rank4_result = self.get_result(rank4_pos_matrix, rank4_topk_idx)
        rank5_result = self.get_result(rank5_pos_matrix, rank5_topk_idx)

        # 在pos_matrix中，正样本位置是1，负样本位置是0，下面是将topk_idx中所预测的正样本的位置的值选出来，如果是1就代表预测对了，如果是0就代表预测错了
        # 在data_struct中添加一个数据，名字交rec.topk，值为result
        self.data_struct.update_tensor('rec.rank1topk', rank1_result)
        self.data_struct.update_tensor('rec.rank2topk', rank2_result)
        self.data_struct.update_tensor('rec.rank3topk', rank3_result)
        self.data_struct.update_tensor('rec.rank4topk', rank4_result)
        self.data_struct.update_tensor('rec.rank5topk', rank5_result)

        rank5result=self.get_rank_result(pos_matrix,topk_idx,rank5item)

        # rec.rank？item在？区间的item的id

        self.data_struct.update_tensor('rec.rank1item', rank1item)
        self.data_struct.update_tensor('rec.rank2item', rank2item)
        self.data_struct.update_tensor('rec.rank3item', rank3item)
        self.data_struct.update_tensor('rec.rank4item', rank4item)
        self.data_struct.update_tensor('rec.rank5item', rank5item)
        self.data_struct.update_tensor('rec.rank5result', rank5result)


    def model_collect(self, model: torch.nn.Module):
        """ Collect the evaluation resource from model.
            Args:
                model (nn.Module): the trained recommendation model.
        """
        pass
        # TODO:
    def eval_collect(self, eval_pred: torch.Tensor, data_label: torch.Tensor):
        """ Collect the evaluation resource from total output and label.
            It was designed for those models that can not predict with batch.
            Args:
                eval_pred (torch.Tensor): the output score tensor of model.
                data_label (torch.Tensor): the label tensor.
        """
        if self.register.need('rec.score'):
            self.data_struct.update_tensor('rec.score', eval_pred)

        if self.register.need('data.label'):
            self.label_field = self.config['LABEL_FIELD']
            self.data_struct.update_tensor('data.label', data_label.to(self.device))
    def get_data_struct(self):
        """ Get all the evaluation resource that been collected.
            And reset some of outdated resource.
        """
        returned_struct = copy.deepcopy(self.data_struct)
        for key in ['rec.topk', 'rec.tailtopk', 'rec.headtopk','rec.meanrank', 'rec.score', 'rec.items', 'data.label','rank1topk','rank2topk','rank3topk','rank4topk','rank5topk']:
            if key in self.data_struct:
                del self.data_struct[key]
        return returned_struct