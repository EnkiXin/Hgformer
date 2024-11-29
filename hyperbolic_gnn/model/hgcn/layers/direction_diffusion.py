import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from collections import namedtuple
from tqdm.auto import tqdm
import hyperbolic_gnn.model.hgcn.manifolds.hyperboloid as manifolds
from recbole_gnn.model.hyp_layers import HGCN,LorentzBatchNorm
from hyperbolic_gnn.model.hgcn.layers.lightbihyperformer import LightBiHyperFormer
ModelPrediction = namedtuple('ModelPrediction', ['pred_noise', 'pred_x_start'])
def get_alphas(timesteps):
    def linear_beta_schedule(timesteps):
        scale = 1000 / timesteps
        beta_start = scale * 0.0001
        beta_end = scale * 0.02
        return torch.linspace(beta_start, beta_end, timesteps, dtype=torch.float64)
    betas = linear_beta_schedule(timesteps)
    alphas = 1. - betas
    return alphas, betas
class diffusion(nn.Module):
    def __init__(self,
                 config,
                 interaction_matrix,
                 n_users,
                 n_items
                 ):
        super(diffusion, self).__init__()
        device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
        self.config=config
        self.n_users=n_users
        self.n_items=n_items
        self.device = device
        self.prior = 1
        self.num_timesteps = config['num_timesteps']
        # 太不好的样子，双曲中，如何加上cos,sin？
        # t原本的操作？如何处理？
        # diffrec的embedding？？
        # 多尝试几个方法
        self.time_step_embeddings = torch.nn.Embedding(num_embeddings = self.num_timesteps,
                                                       embedding_dim = config['embedding_size'])
        self.manifold = getattr(manifolds, "Hyperboloid")()
        if config['decoder']=='light_hyperbolic_trm':
            self.decoder=LightBiHyperFormer(
                                         manifold=self.manifold,
                                         curve=config['curve'],
                                         in_channels=config['embedding_size'],
                                         hidden_channels=config['embedding_size'],
                                         num_layers=config['decoder_layers'],
                                         num_heads=config['num_heads'],
                                         temp=config['temp'],
                                         interaction=interaction_matrix,
                                         config=config,
                                         n_users=n_users,
                                         n_items=n_items)
        else:
            self.decoder = HGCN(in_dim=config['embedding_size'],
                                out_dim=config['embedding_size'],
                                num_layers=config['decoder_layers'],
                                manifold=self.manifold,
                                curve=config['curve'],
                                conv=config['decoder'])
        self.alphas, betas = get_alphas(self.num_timesteps)
        self.alphas_cumprod = torch.cumprod(self.alphas, dim=0)
        self.alphas_cumprod_prev = F.pad(self.alphas_cumprod[:-1], (1, 0), value=1.)
        self.minus_alphas_cumprod = torch.sqrt(1.-self.alphas_cumprod)
        self.sqrt_alphas_cumprod = torch.sqrt(self.alphas_cumprod)
        self.sqrt_recip_alphas_cumprod = torch.sqrt(1. / self.alphas_cumprod)
        self.sqrt_recipm1_alphas_cumprod = torch.sqrt(1. / self.alphas_cumprod - 1)
        self.posterior_variance = betas * (1. - self.alphas_cumprod_prev) / (1. - self.alphas_cumprod)
        self.posterior_log_variance_clipped = torch.log(self.posterior_variance.clamp(min=1e-20))
        self.posterior_mean_coef1 = betas * torch.sqrt(self.alphas_cumprod_prev) / (1. - self.alphas_cumprod)
        self.posterior_mean_coef2 = (1. - self.alphas_cumprod_prev) * torch.sqrt(self.alphas) / (
                    1. - self.alphas_cumprod)
        self.alphas = self.alphas.cuda()
        self.alphas_cumprod = self.alphas_cumprod.cuda()
        self.alphas_cumprod_prev = self.alphas_cumprod_prev.cuda()
        self.sqrt_alphas_cumprod = self.sqrt_alphas_cumprod.cuda()
        self.sqrt_recip_alphas_cumprod = self.sqrt_recip_alphas_cumprod.cuda()
        self.sqrt_recipm1_alphas_cumprod = self.sqrt_recipm1_alphas_cumprod.cuda()
        self.posterior_variance = self.posterior_variance.cuda()
        self.minus_alphas_cumprod=self.minus_alphas_cumprod.cuda()
        self.posterior_log_variance_clipped = self.posterior_log_variance_clipped.cuda()
        self.posterior_mean_coef1 = self.posterior_mean_coef1.cuda()
        self.posterior_mean_coef2 = self.posterior_mean_coef2.cuda()
        self.c=torch.tensor(config['curve']).cuda()
        self.ln=LorentzBatchNorm(self.manifold,
                                 config['embedding_size'],
                                 self.c)
    def exists(self, x):
        return x is not None
    def default(self, val, d):
        if self.exists(val):
            return val
        return d() if callable(d) else d
    def extract(self, a, t, x_shape):
        b, *_ = t.shape
        out = a.gather(-1, t).to(self.device)
        return out.reshape(b, *((1,) * (len(x_shape) - 1)))
    def normalization(self,x):
        mean_x = torch.mean(x)
        std_x = torch.std(x)
        n1 = (x - mean_x) / std_x
        return n1
    def renormal(self,x,h):
        mean=torch.mean(h)
        std=torch.std(h)
        return x*std+mean
    def tran_direction(self, direction_vector, gaussian_point, labels):
        # Given a vector
        device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
        # tran=[]
        for i, noise in enumerate(gaussian_point):
            transformed_vector = torch.sign(direction_vector[labels[i]])
            # transformed_vector= -1 * transformed_vector
            gaussian_point[i] = gaussian_point[i] * transformed_vector
        return gaussian_point
    def get_alphas2(self, timesteps):
        # 得到alpha和beta
        def linear_beta_schedule(timesteps):
            scale = 1000 / timesteps
            beta_start = scale * 0.0001
            beta_end = scale * 0.02
            return torch.linspace(beta_start, beta_end, timesteps, dtype=torch.float64)
        betas = linear_beta_schedule(timesteps)
        alphas = 1. - betas
        alphas_cumprod = torch.cumprod(alphas, dim=0)
        alphas_minus = torch.sqrt(1. - alphas_cumprod)
        return torch.sqrt(alphas_cumprod).to(self.device), alphas_minus.to(self.device)
    def x_tran(self, x0, labels, u0):
        for i, x in enumerate(x0):
            u = u0[labels[i]]
            x = self.manifold.logmap(u, x, c=1.0)
        return x0
    def getpri(self,t):
        c_num=np.sqrt(self.c)
        T=500
        out=self.prior*torch.tanh(c_num*t/T)
        b, *_ = t.shape
        return out.reshape(b, *((1,) * (2 - 1)))
    def q_sample(self,
                 x_start,
                 t,
                 noise,
                 ):
        noise = self.default(noise, lambda: torch.randn_like(x_start))
        alphas_cumprod, minus_alphas_cumprod = self.get_alphas2(self.num_timesteps)
        alphas_cumprod = alphas_cumprod.cuda()
        minus_alphas_cumprod = minus_alphas_cumprod.cuda()
        pri_cumprod=self.getpri(t)
        return (
                self.extract(alphas_cumprod, t, x_start.shape) * x_start +
                self.extract(minus_alphas_cumprod, t, x_start.shape) * noise+pri_cumprod*x_start
                )
    def hyper_q_sample(self,
                 x_start,
                 t,
                 noise,
                 ):
        # 根据步长得到alpha和beta
        hyper_noise=self.hypernoise(noise, x_start)
        alphas_cumprod, minus_alphas_cumprod = self.get_alphas2(self.num_timesteps)
        # pri_cumprod是否在双曲空间需要？？在双曲上应该怎么改写？
        noisy_output=(self.extract(alphas_cumprod, t,x_start.shape) * x_start + self.extract(minus_alphas_cumprod, t, x_start.shape) * hyper_noise)
        noisy_output_norm = self.manifold.tensor_minkouski_norm(noisy_output)
        coefficient = (1 / (self.c ** 0.5)) / (torch.sqrt(torch.abs(noisy_output_norm)) + 1e-14)
        noisy_output=noisy_output*coefficient
        return noisy_output
    def predict_noise_from_start(self, x_t, t, x0):
        prio=self.getpri(t)
        # predicte Gaussian noise from the starting state of the prediction graph
        return (
                ((self.extract(self.sqrt_recip_alphas_cumprod, t, x_t.shape)+prio) * x_t - x0) / \
                self.extract(self.sqrt_recipm1_alphas_cumprod, t, x_t.shape)
        )
    def q_posterior(self, x_start, x_t, t):
        # Compute the mean and variance of the posterior distribution q(x_t | x_{t-1}).
        posterior_mean = (
                self.extract(self.posterior_mean_coef1, t, x_t.shape) * x_start +
                self.extract(self.posterior_mean_coef2, t, x_t.shape) * x_t
        )
        posterior_variance = self.extract(self.posterior_variance, t, x_t.shape)
        posterior_log_variance_clipped = self.extract(self.posterior_log_variance_clipped, t, x_t.shape)
        return posterior_mean, posterior_variance, posterior_log_variance_clipped
    def model_predictions(self,
                          x,
                          adj,
                          t,
                          x_self_cond=None,graphset=False):
        # t:归一化到0~1
        time_embedding=self.manifold.proj(self.manifold.expmap0(self.time_step_embeddings.weight*0.1, self.c), self.c)
        time_embedding=self.ln(time_embedding)
        x = self.time_step_encoding(
                                   x,
                                   time_embedding[t],
                                   'hyper')
        if self.config['decoder'] == 'para_hgcn':
           model_output = self.decoder(x, adj)
        elif self.config['decoder'] == 'light_hyperbolic_trm':
           u,i=torch.split(x, [self.n_users, self.n_items])
           model_output = self.decoder(u,i,adj)
        x_start = model_output
        # x_start = maybe_clip(x_start)
        #pred_noise = self.predict_noise_from_start(x, t, x_start)
        return x_start
    def p_mean_variance(self,
                        x,
                        adj,
                        t,
                        x_self_cond=None,
                        graphset=False):
        preds = self.model_predictions(x,
                                       adj,
                                       t,
                                       x_self_cond,
                                       graphset)
        # model_mean, posterior_variance, posterior_log_variance = self.q_posterior(x_start=x_start, x_t=x, t=t)
        return preds
    def p_sample(self,
                 x,
                 adj,
                 t,
                 x_self_cond=None,
                 graphset=False):
        b, *_, device = *x.shape, self.device
        batched_times = torch.full((b,), t, device=device, dtype=torch.long)
        #print(batched_times.shape)
        x_start = self.p_mean_variance(x=x,
                                       adj=adj,
                                       t=batched_times,
                                       x_self_cond=x_self_cond,
                                       graphset=graphset)
        return x_start
    def p_sample_loop(self,
                      user_all_embeddings,
                      item_all_embeddings,
                      adj,
                      graphset=False):
        x = torch.cat([user_all_embeddings, item_all_embeddings], dim=0)
        shape = x.shape
        batch, device = shape[0], self.device
        euc_noise = torch.randn(shape, device=device)*0.1
        euc_noise[:, 0] = 0
        hyper_noise = self.hypernoise(euc_noise, x)
        noisy_data=(hyper_noise+x)/2
        noisy_data_norm=self.manifold.tensor_minkouski_norm(noisy_data)
        coefficient = (1 / (self.c ** 0.5)) / (torch.sqrt(torch.abs(noisy_data_norm)) + 1e-7)
        noisy_data=noisy_data*coefficient
        self_condition = None
        x_starts = []
        # imgs：每个时刻的embeddings，最开始的是加满噪声的
        for t in reversed(range(0, self.num_timesteps)):
            self_cond = x_start if self_condition else None
            hyper_noise = hyper_noise
            # t = t / self.num_timesteps
            # p_sample输入是noisy的
            x_start = self.p_sample(noisy_data,
                                    adj,
                                    t,
                                    self_cond,
                                    graphset)
            x_starts.append(x_start)
        # x_pred= self.renormal(x_starts[-1], x)
        ret=x_starts[-1]
        return ret
    def time_step_encoding(self,
                           target_vecor,
                           positional_embedding,
                           manifold):
        # 将time step当作positional encoding加入
        if manifold=='hyper':
            #positional_embedding=self.manifold.proj(self.manifold.expmap0(positional_embedding,self.c),self.c)
            #positional_embedding=self.ln(positional_embedding)
            # print(positional_embedding)
            # print(self.manifold.tensor_minkouski_norm(positional_embedding))

            tensor=self.manifold.mobius_add(target_vecor,
                                            positional_embedding,
                                            self.c)
        else:
            tensor =target_vecor+positional_embedding
        return tensor
    def hypernoise(self,euc_noise,hyper_vector):
        # 将向量放到north pole ppoint的
        euc_noise[:, 0] = 0
        # 将上面的noise转移到hyper_vector的tangent space上
        # 推导：noise具有的信息更好？叠加到embedding所具有的信息更多？noise叠加的时间复杂度。
        hyper_noise_tangent = self.manifold.ptransp0(hyper_vector,euc_noise,self.c)
        hypernoise = self.manifold.proj(self.manifold.expmap(hyper_noise_tangent,hyper_vector,self.c),self.c)
        return hypernoise
    def forward(self,
                user_all_embeddings,
                item_all_embeddings,
                adj
                ):
        # x:通过前面encoder得到的embeddings。[num_nodes,embedding_dim]
        # adj:邻接矩阵
        x = torch.cat([user_all_embeddings, item_all_embeddings], dim=0).to(self.device)
        h = x
        b, n = h.shape
        t = torch.randint(0,self.num_timesteps, (b,), device=h.device).long().to(self.device)
        # 一个高斯噪声，并送入双曲空间
        # 双曲的正态分布（重点），看看能不能推出欧式到双曲过后是什么分布
        # uniform,importance的是否也能推导？
        # 双曲normalization
        noise = (torch.randn_like(h)*0.1).to(self.device)
        # 直接在双曲空间加噪声？
        xt = self.hyper_q_sample(
                           x_start=x,
                           t=t,
                           noise=noise)
        time_embedding=self.manifold.proj(self.manifold.expmap0(self.time_step_embeddings.weight*0.1, self.c), self.c)
        # time_embedding=self.ln(time_embedding)
        xt = self.time_step_encoding(
                                   xt,
                                   time_embedding[t],
                                   'hyper')
        xt[0, :] = 0
        # 适合先预测噪声，再减去
        # psample的代码中
        # xt->x0（还行？）在还原回去时是一步一步还原的
        # 平均场理论，用于组合优化求解？？？？
        # xt->noise xt-noise=x0（这个版本双曲版本？？）
        # 真实的边
        # 预测的embeddings
        # xt: noisy的embeddings
        # gcn的input：加噪后的embeddings与timestamp
        # gcn的output:预测的去噪之后的结果
        if self.config['decoder']=='light_hyperbolic_trm':
           user_embeddings, item_embeddings = torch.split(xt, [self.n_users, self.n_items])
           user_embeddings,item_embeddings = self.decoder(user_embeddings.to(torch.float32), item_embeddings.to(torch.float32),adj)
           out = torch.cat([user_embeddings, item_embeddings], dim=0).to(self.device)
        else:
           out= self.decoder(xt.to(torch.float32), adj)

        #print('out',out)
        # 在欧氏空间用另一个gcn进行encode
        # 目标是让新的gcn能够预测去噪前的embeddings
        # 换成双曲距离
        loss = self.manifold.pair_wize_hyper_dist(out, x, self.c)
        loss = torch.sum(loss)
        return loss