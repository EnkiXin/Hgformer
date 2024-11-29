import torch
import torch.nn as nn
from torch.nn import init
from torch.nn.modules.module import Module
import math
import torch.nn.functional as F
class LorentzBatchNorm(nn.Module):
    """ Lorentz Batch Normalization with Centroid and Fréchet variance
    """
    def __init__(self,
                 manifold,
                 dim,
                 curve
                 ):
        super(LorentzBatchNorm, self).__init__()
        self.manifold = manifold
        self.curve = torch.tensor(curve).to('cuda:0')
        self.beta = torch.zeros(dim).to('cuda:0')
        self.beta[0] = torch.sqrt(1/self.curve)
        self.gamma = torch.nn.Parameter(torch.ones((1,))).to('cuda:0')
        self.eps = 1e-7
        # running statistics
    def forward(self, x, momentum=0.1):
        beta = self.beta
        #beta:north pole point...
        mean = self.manifold.centroid(x,self.curve)
        x_T = self.manifold.logmap(mean, x,self.curve)
        x_T = self.manifold.transp0back(mean, x_T,self.curve)
        # Compute Fréchet variance
        var = torch.mean(torch.norm(x_T, dim=-1), dim=0)
        # Rescale batch
        x_T = x_T*(self.gamma/(var+self.eps))
        # Transport batch to learned mean
        x_T = self.manifold.transp0(beta, x_T,self.curve)
        output = self.manifold.expmap(beta, x_T,self.curve)
        return output
class HGCN(nn.Module):
    """
    Hyperbolic graph convolution layer.
    """
    def __init__(self,
                 in_dim,
                 out_dim,
                 num_layers,
                 manifold,
                 curve,
                 conv,):
        super(HGCN, self).__init__()
        self.in_features = in_dim
        self.num_layers = num_layers
        self.manifold=manifold
        self.curve=curve
        self.conv=conv
        self.layer_norm=LorentzBatchNorm(self.manifold,out_dim,self.curve)
        if conv=='para_hgcn':
           self.linear = nn.ModuleList()
           self.linear.append(HypLinear(manifold,
                                        in_dim,
                                        out_dim,
                                        curve,
                                        curve))
           for i in range(self.num_layers):
               self.linear.append(HypLinear(manifold,
                                            out_dim,
                                            out_dim,
                                            curve,
                                            curve))
    def hyper_agg(self,tensor):
        # 得到tensor:num_heads*num_nodes*dim
        num_heads,num_nodes,dim=tensor.size(0), tensor.size(1),tensor.size(2)
        tensor=tensor.permute(1,0,2)
        weighted_tensor = torch.full((num_nodes,num_heads,1), 1/num_heads).to(tensor.device)
        tensor = self.manifold.hyperbolic_mean(weighted_tensor,tensor,self.curve)
        return tensor
    def resSumGCN(self,  x, adj):
        output = [x]
        for i in range(self.num_layers):
            output.append(torch.spmm(adj, output[i]))
        return sum(output[1:])

    def LGCN(self,  x, adj):
        # print(x)
        output = [x]
        for i in range(self.num_layers):
            next=torch.spmm(adj, output[i])
            next=self.manifold.proj(next,self.curve)
            norm = torch.abs(self.manifold.tensor_minkouski_norm(next))
            norm_ = 1 / (norm ** (1 / 2))
            next = next*norm_*(1/(self.curve**(1/2)))
            next = self.layer_norm(next)
            output.append(next)
        return output[-1]


    def Para_LGCN(self,  x, adj):
        output = [x]
        for i in range(self.num_layers):
            next=torch.spmm(adj, output[i])
            next=self.manifold.proj(next,self.curve)
            norm = torch.abs(self.manifold.tensor_minkouski_norm(next))
            norm_ = 1 / (norm ** (1 / 2))
            next = next*norm_*(1/(self.curve**(1/2)))
            next = self.layer_norm(next)
            out = self.linear[i](next)
            out = self.manifold.proj(out, self.curve)
            out = self.layer_norm(out)
            output.append(out)
        return output[-1]
    def forward(self, x, adj):
        if self.conv=='resSumGCN':
            # 输入时双曲空间的向量
            x = self.manifold.logmap0(x, self.curve)
            output = self.resSumGCN(x, adj)
            output = self.manifold.proj(self.manifold.expmap0(output, self.curve), self.curve)
        elif self.conv=='lGCN':
            output = self.LGCN(x, adj)
        elif self.conv=='para_hgcn':
            output = self.Para_LGCN(x, adj)
        return output


class HypLinear(nn.Module):
    """
    Hyperbolic linear layer.
    """
    def __init__(self, manifold, in_features, out_features, c_in,c_out):
        super(HypLinear, self).__init__()
        self.manifold = manifold
        self.in_features = in_features
        self.out_features = out_features
        self.c_in = c_in
        self.c_out = c_out
        self.bias = nn.Parameter(torch.Tensor(out_features))
        self.weight = nn.Parameter(torch.Tensor(out_features, in_features))
        self.reset_parameters()
    def reset_parameters(self):
        init.xavier_uniform_(self.weight, gain=math.sqrt(2))
        init.constant_(self.bias, 0)
    def forward(self, x):
        drop_weight = F.dropout(self.weight, 0.2,training=True)
        # 将在曲率为c_in的x映射到tangent space做矩阵乘法，然后映射到曲率为c_out的空间
        mv = self.manifold.multi_curve_mobius_matvec(drop_weight, x, self.c_in,self.c_out)
        res = self.manifold.proj(mv, self.c_out)
        bias = self.manifold.proj_tan0(self.bias.view(1, -1), self.c_out)
        hyp_bias = self.manifold.expmap0(bias, self.c_out)
        hyp_bias = self.manifold.proj(hyp_bias, self.c_out)
        res = self.manifold.mobius_add(res, hyp_bias, c=self.c_out)
        res = self.manifold.proj(res, self.c_out)
        return res
class HypAct(Module):
    """
    Hyperbolic activation layer.
    """
    def __init__(self, manifold, c_in, c_out):
        super(HypAct, self).__init__()
        self.manifold = manifold
        self.c_in = c_in
        self.c_out = c_out
        self.act = nn.ReLU()
    def forward(self, x):
        xt = self.manifold.logmap0(x, c=self.c_in)
        xt = self.act(xt)
        xt = self.manifold.proj_tan0(xt, c=self.c_out)
        return self.manifold.proj(self.manifold.expmap0(xt, c=self.c_out), c=self.c_out)
    def extra_repr(self):
        return 'c_in={}, c_out={}'.format(
            self.c_in, self.c_out
        )
class FermiDiracDecoder(Module):
    """Fermi Dirac to compute edge probabilities based on distances."""
    def __init__(self, r, t):
        super(FermiDiracDecoder, self).__init__()
        self.r = r
        self.t = t
    def forward(self, dist, split='train'):
        probs = 1. / (torch.exp((dist - self.r) / self.t) + 1.0)
        return probs