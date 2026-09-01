# SL(n) 上的求平均：代码诊断、文献综述与改进建议

日期：2026-09-01 ・ 针对 `Hgformer/slrec_experiments` + `recbole_gnn` 中的 SL8LHGCN。
背景：SL8LHGCN 尚未超越 LHGCN，怀疑 SL 流形上的求平均环节。本笔记 = 对现有两种聚合算子的数值诊断 + SL(n)/李群均值文献 + 按优先级的改动建议。文献细节另见项目文档 `claude/sln-averaging-literature-2026-09-01.md`（英文备忘录）。

## 1. 一句话结论

怀疑方向对了一半：`ambient_retract`（当前默认）确实是系统性偏差的来源，且在扩散变大时灾难性退化；但换成更"高级"的内在均值（Karcher/双不变均值）在你们当前的 coordinate spread 下**几乎没有收益**——切空间均值在 spread≤0.5 时与收敛的 Karcher 均值差 0.1% 以内。真正与 LHGCN 拉开差距的更可能是 **LHGCN 每层都有 LorentzBatchNorm（带可学习尺度的 Fréchet 方差重标定），而 SL8 什么都没有**——而"SL(n) 没有 operator-equivalent 的 BatchNorm"这个设定过于悲观：LieBN (ICLR 2024) 给出的正是任意李群上的逐操作等价配方。

## 2. 当前实现与 LHGCN 的真实对照

LHGCN（released code，`hyp_layers.py:124-135`）每层做三件事：① 环境空间加权和；② 径向重投影回双曲面（= Law et al. 2019 的闭式 Lorentz 质心，是加权平方洛伦兹距离的**精确**最小元，且对权重整体缩放不变）；③ `LorentzBatchNorm`：批质心居中 + Fréchet 方差重标定（可学习 γ）+ 平移到基点。

SL8LHGCN 两种模式（`sl_lhgcn.py`, `sl8lhgcn.py`）：

- `ambient_retract`：M = Σ Ã_vw G_w，然后 G' = M/det(M)^{1/8}；det<0 时反射末列"修复"，奇异时回退 exp(trace_free(M))。**没有任何逐层规范化。**
- `tangent_last`：坐标 X 直接稀疏传播 L 层后一次 exp。**同样没有规范化，且对称归一化邻接的行和 < 1，每层把坐标向 0（恒等元）收缩。**

注意一个被忽略的混淆变量：det-retraction 对整体缩放不变（M/det^{1/8} 与权重行和无关），tangent 模式不是——行和<1 在 tangent 模式里是逐层向恒等元的收缩，在 ambient 模式里被 det 归一化消掉。所以你们 `sl_gcn_mode` 的 A/B 实验同时改变了"均值质量"和"尺度处理"两件事，结论会互相污染。LHGCN 靠 ②+③ 天然免疫这个问题。

## 3. 数值实验（float64，n=8，12 邻居，权重模拟对称归一化）

四个算子在他们各自输出 m 上评估同一目标 F(m)=Σ w̄ᵢ‖log(m⁻¹Gᵢ)‖²_F（精确 scipy logm）：

| 邻居坐标 σ | det(M)≤0 比例 | F(ambient)/F(Karcher*) | F(tangent)/F(Karcher*) | F(一步不动点)/F(Karcher*) |
|---:|---:|---:|---:|---:|
| 0.05 | 0% | 1.000 | 1.000 | 1.000 |
| 0.20 | 0% | 1.010 | 1.000 | 1.000 |
| 0.30 | 0% | 1.019 | 0.999 | 1.000 |
| 0.50 | 0% | **1.117** | 1.001 | 0.997 |
| 0.80 | 3.3% | **1.720** | 0.988 | 0.947 |
| 1.20 | **38.3%** | 全体失效（脱离主对数域） | — | — |

解读：

- **ambient_retract 在所有 spread 下都是最差的均值**，σ=0.5 时目标值已差 12%，σ=0.8 差 72%；σ=1.2 时 38% 的环境和落在负行列式分量上——"反射末列修复"把结点送到语义上任意的位置，这个区间的梯度是噪声。这与 `SL8_LHGCN.md` 里"repair rate 高即不稳定"的自我警告一致，但量化后可以看出退化远早于修复触发：**修复率为 0 不代表均值是好的**。
- 与 Lorentz 情形的关键不对称：双曲面上未来类时向量的加权和永远落在光锥内、径向投影恒有定义且恰为 Fréchet 均值；SL(n) 的环境和可以任意接近奇异/变号，det 归一化把所有奇异值同乘一个标量，**不是**向最近 SL 点的投影，更不是任何目标的最小元。"Lorentz 的技巧"在 SL 上没有对应的几何保障。
- **tangent 均值 ≈ 收敛 Karcher 均值（spread≤0.5 内差 ≤0.1%）**。这与文献一致：一步展开下两者相差 O(σ²) 的 BCH 交换子项。所以"换更好的内在均值"本身不解决问题——除非 spread 先变大，而 spread 是规范化层该管的事。
- Gregory K=12 截断 log 在相对距离 d≤3 时基本精确（相对误差 ~5e-8），d=4 开始出现坏例（最大相对误差 22），d=6 中位误差 2.4e-2、最大 5.7e8。当前 coord_clip 0.5–1.5 下初始化安全，但训练漂移 + hinge 损失把正负样本距离往两边推之后，**打分域可能越界而无告警**。

实验脚本：`benchmark_sl_mean_operators.py`（本目录），可在 .venv-slrec 里直接跑。

## 4. 文献：SL(n)/李群上有哪些"好"的均值

按对你们的适用性排序（完整出处见第 6 节）：

**双不变均值 / 指数重心**（Pennec & Arsigny 2012；Lawson 2025）。任意李群上，Cartan–Schouten 典范联络的指数重心定义为 Σ w̄ᵢ log(m⁻¹gᵢ)=0 的解，不动点迭代 m ← m·exp(Σ w̄ᵢ log(m⁻¹gᵢ))。左/右平移、取逆全不变。这正是你们 Schatten-2 半距离目标 Σ w̄‖log(m⁻¹g)‖²_F 的驻点方程——**它就是与你们打分几何相容的那个"正确均值"**。代价：SL(n) 非紧半单、无双不变黎曼度量，该均值只是伪黎曼/仿射意义的重心，只有局部存在唯一性（数据落在正规凸邻域内）；群指数在 SL(n) 不满射，邻居彼此太远时 log 直接无定义——所以任何 SL 均值都是**局部**对象，管住 spread 是前提而不是可选项。

**对数欧氏式一步均值** exp(Σ w̄ᵢ log gᵢ)——在你们的参数化下 log 是白拿的（就是坐标 X），即 `tangent` 模式。等于从恒等元出发的一步不动点迭代；失去左右平移不变性，保留取逆与共轭不变。SPD-GCN、HGCN 的切空间聚合、LieBN 全部采用这一族，是该领域的事实共识。实验表明在你们的工况下它与收敛 Karcher 差 0.1% 以内。

**一步校正的截断 Karcher**：m₀ = exp(Σ w̄ X)（切空间种子），再走一步 m = m₀·exp(Σ w̄ log(m₀⁻¹Gᵢ))。二阶精度逼近双不变均值；log 的自变量在 m₀ 附近，Gregory 6–8 项即可。每条边一次 8×8 log（Amazon-CD 1.5M 边 × 每层 ≈ 20 GFLOPs，可行；显存靠 checkpoint/仅末层启用控制）。这是文献里标准的"log-Euclidean 初始化 + 一步 Newton"折中。

**极分解拆分均值**：g = p·u（SPD × SO(n)，两因子自动 det=1），SPD 部分用 log-Euclidean/Karcher 均值（Moakher 2005, Arsigny 2005/2007），旋转部分用 Moakher 2002 的投影算术均值（Σ w u 的正交极因子）。只保 O(n) 等变，但把"拉伸"与"旋转"分开平均，可当诊断消融：看信号到底在哪个因子里。Gawlik & Leok 2016 在对称空间上给了结构保持的同类构造。SL(n)/SO(n) 恰是单位行列式 SPD 的对称空间，那一半是有闭式测地线的。

**可微 Fréchet 均值层**：Lou et al. (ICML 2020) 用隐函数定理对 Fréchet 均值反传，但实例只做了常曲率空间；对 SL(n) 用一步截断比隐式微分更实际。

没有找到任何已发表工作在 GNN 里对 SL(n) 做邻居聚合——这块无论选哪个算子，方法论本身就是可发表的贡献点。

## 5. 建议（按预期收益排序）

1. **补上 SL 版逐层规范化（LieBN 配方），这是 vs LHGCN 最大的结构性缺口**。逐操作对应 LorentzBatchNorm：批（或活跃结点）中心 m 用一步双不变均值；居中 g → m⁻¹g；在 log 域做可学习尺度 v = γ·log(m⁻¹g)/(σ̂+ε)；回写 g' = β·exp(v)（β 可学习基点或恒等元）。每层聚合后应用。`sl8lhgcn.py` 里 "no operator-equivalent SL analogue" 的硬约束建议放开为一个新的 `sl_layer_norm: liebn` 选项（保留 `none` 作对照）。它同时解决第 2 条的尺度问题和第 3 条的域控制问题。
2. **消掉 tangent 模式的行和收缩混淆**：给 tangent 传播加 `row_normalize: true` 选项（随机游走归一化或除以行和），使每层是真凸组合；重跑 `sl_gcn_mode` A/B。当前的对照实验里 tangent 输给 ambient 的部分可能纯粹是逐层收缩造成的。
3. **把 `ambient_retract` 降级或替换**：实验证明它在一切 spread 下都劣于切空间均值，且没有 Lorentz 情形的几何保障。若要保留一个"直接在群上聚合"的模式，用第 4 节的一步校正 Karcher（`sl_gcn_mode: karcher1`）替代——它才是与你们打分半距离相容的内在均值，且负 det/奇异问题从根上消失（永远不经过环境和）。负 det 的"反射末列修复"无论如何应该去掉：那个区间里直接回退到该行的切空间聚合结果，比反射语义上干净得多。
4. **给 Gregory 打分加域监控**：训练中周期性记录相对矩阵 Cayley 变换 Z 的谱半径（或 ‖log‖ 上界），>0.9 时告警；或在 d>3 的工况下把 K=12 提到 16/20，或用一次 Denman–Beavers 平方根步扩域（log A = 2·log √A）。现在的 membership 诊断只查行列式，查不到截断误差。
5. 诊断性消融：极分解拆分均值（看信号在 SPD 因子还是旋转因子），以及 spread 监控（每 epoch 记录 ‖X‖ 分布与邻居间 pairwise ‖log(Gᵢ⁻¹Gⱼ)‖ 分位数)——第 3 节的表说明所有结论都以 spread 为条件。

## 6. 主要文献

Pennec & Arsigny, *Exponential barycenters of the canonical Cartan connection and invariant means on Lie groups*, 2012（双不变均值、存在唯一性、Alg.1 不动点迭代）・ Lawson, *The weighted exponential mean on Lie groups*, 2025（纯李代数推导 + Banach 李群局部定理）・ Arsigny et al., *Log-Euclidean metrics*, MICCAI'05/SIAM'07 ・ Moakher, SO(n) 均值 SIAM'02；SPD 几何均值 SIAM'05 ・ Gawlik & Leok, *Interpolation via generalized polar decomposition*, 2016 ・ Lou et al., *Differentiating through the Fréchet Mean*, ICML'20 ・ Law et al., *Lorentzian Distance Learning*, ICML'19（LHGCN 质心的出处与其最优性）・ Chen et al., *LieBN: Lie group batch normalization*, ICLR'24（第 5.1 条的配方与代码 github.com/GitZH-Chen/LieBN）・ Zhao et al., *GNNs on SPD manifolds*, 2023（切空间聚合的先例）・ Al-Mohy, Higham & Relton, matrix log Fréchet derivative, SIAM'13（如需精确 log 反传）。
