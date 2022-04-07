# 《机器学习》（周志华）笔记

## 第1章 绪论

NFL定理

## 第2章 模型评估与选择
### 2.1 经验误差与过拟合
- 错误率(error rete)：分类错误的样本数占总样本数的比例
- 精度(accuracy): 分类正确的样本占总样本数的比例
- 误差(error)：学习器的实际预测输出与样本间的真实输出之间的差异。在训练集上的误差称为 _“训练误差(training error）”_ 或 _“经验误差(empirical error)”_，在新样本中的误差称为 _“泛化误差(generalization error)”_
- 过拟合(overfitting)：把训练样本自身的特点当作潜在样本的普遍性质
- 欠拟合(underfitting)：对训练样本的一般性质尚未学好

### 2.2 评估方法
通常使用测试集来测试学习器对新样本的判别能力，然后用测试集上的“测试误差”作为泛化误差的近似。测试集与训练集应当互斥。

#### 2.2.1 留出法
“留出法”(Hold out)直接将数据集D划分为两个互斥的集合其中一个集合作为训练集$S$,另一个作为测试集$T$，即$T=S\cup D$, $S\cap T= \empty $
测试集/训练集的划分要尽可能保持数据分布的一致性，避免因数据划分过程引入额外的偏差。（分层采样：保留类别比例的采样方式）常见做法是将大约2/3~4/5的样本用于训练，剩余样本用于测试。

#### 2.2.2 交叉验证
__“交叉验证”（cross validation）__：先将数据集$D$划分为$k$个大小相似的互斥子集，即$D=D_1 \cup D_2 \ldots \cup D_k, D_i \cap D_j = \empty (i \ne j).$ 每个子集$D_i$都尽可能保持分布的一致性，即从D中分层采样得到。然后用$k-1$个子集的病机作为训练集，余下的作为测试集；这样可以获得$k$组训练/测试集，可进行$k$次训练和测试，最终返回$k$次测试结果的均值。通常称为“$k$折交叉验证”（k-flod cross validation), $k$最常用的值是10、5、20等。
当数据集$D$中包含$m$个样本，令 $k=m$，则得到交叉验证法的特例：留一法（Leave-One-Out）。

#### 2.2.3 自助法    

__自助法（bootstrapping)：__ 直接以自助采样法（bootstrap sampling）为基础，给定包含$m$个样本的数据集$D$，我们对它进行采样产生数据集$D'$：每次随机从$D$中挑选一个样本，将其拷贝放入$D'$，然后再将该样本放回初始数据集$D$中，使得该样本再下次采样时仍有可能被采到；这个过程重复执行$m$次，我们就得到了包含$m$个=样本的数据集$D'$。
显然一些样本会在$D'$中重复出现，可以以做一个简单的估计，样本在$m$次采样中始终不被采到的概率时${(1- \frac {1}{m})}^m$，取极限得到
$$
{{\displaystyle \lim_{x \to \infty}}({1- \frac{1}{m}}) \mapsto \frac{1}{e} \approx} = 0.368
$$
通过自助采样，初始数据集$D$中约有36.8%的样本未出现在采样数据集$D'$中。于是我们可以将$D'$作为训练集，剩下的作为测试集。这样的测试结果，亦称“包外估计”(Out-of-Bag Estimate).
_tips_:自助法在数据集较小，难以有效划分训练/测试集时很有用。

#### 2.2.4 调参与最终模型
定包含$m$个样本的数据集$D$，在模型评估与选择中需要留出一部分数据进行评估测试（验证集（validation），事实上我们只使用了一部分数据训练模型。

#### 2.3 性能度量
在预测任务中，给定样例集$D = \{(\bm{x_1}, y_1), (\bm{x_2}, y_2), \ldots , (\bm{x_m}, y_m) \}$，其中$y_i$是实例$\bm{x_i}$的真实标记，对于学习器$f$的性能， 我们有：
回归任务最常用的性能度量是“均方误差”（mean square error）
$$E(f;D) = \frac{1}{m} \sum^{m}_{i=1} (f(\bm{x_i})-y_i)^2 $$
更一般的，对于数据分布$\mathcal{D}$和概率密度函数$p(·)$，均方误差可描述为
$$E(f;\mathcal D) = {\int_{x \sim \mathcal {D}}}(f(\bm{x})- y)^2 p(\bm{x}) d\bm{x}  $$

#### 2.3.1 错误率与精度
对于样例集$D$，分类错误率定义为
$$E(f;D)=\frac{1}{m} \sum^{m}_{i=1} \mathbb{I}(f(\bm{x}_i)\ne y_i) $$
精度定义为
$$acc(f;D)= \frac{1}{m} \sum^{m}_{i=1}\mathbb{I}(f(\bm{x}_i)= y_i)
$$

#### 2.3.2 查准率、查全率与$F_1$
对于二分类问题，我们有如下的混淆矩阵(confunsion matrix)

![alt](./ml_image/ml2-3.png)

查准率（P）：
$$Precision=\frac{TP}{TP + FP} $$
查全率（R）：
$$Recall= \frac{TP}{TP + FN}$$

Recall和Precision是一对矛盾的度量，而平衡点（BEP)是综合考虑二者的度量，它是“查全率=查准率”时的取值。但我们有更常用的是$F_1$度量：
$$F_1=\frac{2\times P \times R}{P+R}=\frac{2 \times TP}{样例总数 + TP-TN} $$

tips：$F_1$是基于$P$和$R$的调和平均定义的：
$$\frac{1}{F_1} = \frac 1 2 (\frac 1P + \frac 1R ) $$

在一些应用中，如商品推荐中，为了更少打扰用户，更希望推荐的内容是用户感兴趣的，此时查准率更重要；而在逃犯检索系统中，希望更少漏掉逃犯，此时查全率更重要，$F_B$能表达出不同的偏好
$$F_=\frac{({1+\beta ^2})\times P \times R}{(\beta^2 \times P)+R}$$

tips：$F_B$是基于$P$和$R$的加权平均定义的：
$$\frac{1}{F_B} = \frac 1 {1+\beta ^2} (\frac 1 P + \frac {\beta ^2}R ) $$

很多时候我们有多个二分类混淆矩阵，我们希望在$n$个二分类混淆矩阵上综合考虑查准率和查全率。一种直接的做法是在个混淆矩阵上计算出查准率和查全率$ (P_1,R_1), (P_2, R_2) \ldots (P_n, R_n) $再计算平均，得到“宏查准率”，“宏查全率”，以及相应的“宏$F_1$”:
$$
macro-P = \frac 1n \sum^{n}_{i=1} P_i
$$

$$
macro-R = \frac 1n \sum^{n}_{i=1} R_i
$$

$$
macro-F_1 = \frac{2 \times marco-P \times macro-R}{marco-P + macro-R}
$$

还可将各混淆矩阵的对应元素进行平均，得到$\bar{TP}、\bar{FP}、 \bar{TN}、 \bar{FN} $，再基于这些平均值计算出“微查准率”(micro-P)“ 微查全率 ”（micro-R）和“ 微$F_1$ ”(micro-$F_1$)

$$
micro-p = \frac{\bar{TP}}{\bar{TP} + \bar{FP}}
$$

$$
micro-R = \frac{\bar{TP}}{\bar{TP} + \bar{FN}}
$$

$$
micro-F_1 = \frac{2 \times micro-P \times micro-R}{micro-P + micro-R}
$$

#### 2.3.3 ROC与AUC
- 真正例率(True Positive Rate, TPR)
$$
TPR=\frac{TP}{TP+FN}
$$
- 假正例率(False Positive Rate, FPR)
$$
FPR=\frac{FP}{TN+FP}
$$

ROC曲线的纵轴是TPR，横轴是FPR，ROC曲线是由坐标为$ \{ (x_1, y_1), (x_2, y_2), \ldots , (x_m, y_m) \} $的点按序连接而形成($x_1=0, \ x_m=1 $)
AUC为ROC曲线和横轴之间部分的面积，可估算为：
$$
AUC=\frac 12 \sum^{m-1}_{i=1}(x_{i+1}-x_i)\cdot (y_i +y_{i+1})
$$
给定$m^+$个正例和$m^-$个反例，令$D^+$和$D^-$分别表示正反例集合，则排序损失(loss)定义为
$$
\mathscr{\ell}_{rank}= \frac {1}{m^+ m^-}\sum_{\bm{x}^+\in {D^+}} \sum_{\bm{x}^- \in D^-} \Big( \mathbb{I}\big(f(\bm{x}^+) < f(\bm{x}^-)\big) + \frac12 \mathbb{I} \big( f(\bm{x}^+)= f(\bm{x}^-) \big) \Big)
$$
即考虑每一对正、反例，若正例的预测值小于反例，则记一个罚分，若相等，则记0.5个罚分。容易看出，$\mathscr{l}_{rank}$对应的是ROC曲线之上的面积，因此有
$$
AUC = 1- \mathscr{\ell}_{rank}
$$

#### 2.3.4 代价敏感错误率与代价曲线
为权衡不同类型错误所造成的不同损失，可为错误赋予“非均等代价”(unequal cost).设定代价矩阵(cost matrix)，其中$cost_{ij}$表示将第$i$类样本预测为第$j$类样本的代价
![alt](./ml_image/ml2-3-4.png)

在非均等代价下，我们不再简单地最小化错误次数，而是希望最小化“总体代价”(total cost). 若将上表中的第0类作为正类，则代价敏感(cost-sensitive)错误率为
$$
\begin{split}
E(f;D;cost)=\frac 1m \Bigg( \sum_{\bm{x}_i \in D^+} {\mathbb{I}(f(\bm{x}_i) \ne y_i) \times cost_{01}} + \sum_{\bm{x}_i \in D^-}{(f(\bm{x}_i)\ne y_i) \times cost_{10} } \Bigg)
\end{split}
$$

在非均等代价下，ROC曲线不能直接反映出学习器的期望总体代价，而“代价曲线”可达到该目的。代价曲线的横轴是取值为[0,1]的正例概率代价
$$
P(+)cost = \frac{p \times cost_{01}}{p \times cost_{01} +(1-p)\times cost_{10}}
$$
其中$p$是样例为正的概率；横轴是取值为[0,1]的归一化代价
$$
cost_{norm} = \frac{FNR \times p \times cost_{01} + FPR \times (1-p) \times cost_{10}}{p \times cost_{01} +(1-p)\times cost_{10}}
$$
其中FPR是前面定义的假正例率，FNR = 1 - TPR 是假反例率。

### 2.4 比较检验
#### 2.4.1 假设检验
对于二分类问题，泛化错误率为$\epsilon $的学习器被测得测试错误率为$\hat {\epsilon} $的概率为：
$$
P(\hat {\epsilon}; \epsilon)= \begin {pmatrix} m \\ {\hat{\epsilon} \times m} \end {pmatrix} \epsilon^{\hat{\epsilon} \times m}(1-\epsilon)^{m- \hat{\epsilon} \times m}
$$
给定测试错误率，则解$\partial P(\hat{\epsilon};\epsilon)/\partial \epsilon = 0 $可知，$ P(\hat{\epsilon};\epsilon)$在$\hat{\epsilon}=\epsilon$时最大。这符合二项分布，我们可使用“二项检验”来对“$\epsilon \leqslant {\epsilon}_0 $ ”这样的假设进行检验，则在$1-\alpha$的概率内能观察到的最大错误率如下式计算：
$$
\bar{\epsilon}=\max{\epsilon} \ \  s.t. \sum^{m}_{{i=\epsilon}_0 \times m + 1} \begin{pmatrix} m \\ i \end{pmatrix} {\epsilon}^i (1-\epsilon)^{m-i} < \alpha
$$

__t检验__(t-test)
很多时候我们使用多次重复留出法或者使用交叉验证进行多次训练，这样会得到多个测试错误率，此时可使用"t检验"。假定我们得到$k$个测试错误率$\hat{\epsilon}_1, \hat{\epsilon}_2 \ldots ,\hat{\epsilon}_k $ ，则平均测试错误率$\mu$和方差$\sigma ^2$为
$$
\mu = \frac 1k \sum^{k}_{i=1} \hat{\epsilon} \ ,
$$
$$
\sigma ^2 = \frac1{k-1} \sum^{k}_{i=1} (\hat{\epsilon}_i -\mu)^2
$$
考虑到这$k$个测试错误率可看作泛化错误率$\epsilon _0$的独立采样，则变量
$$
\tau _t = \frac{\sqrt{k}(\mu - \epsilon _0)}{\sigma}
$$
服从自由度为$k-1$的$t$分布

#### 2.4.2 交叉验证t检验
对于两个学习器A和B，我们使用$k$折交叉验证得到的测试错误率分别为$\epsilon^{A}_{1}, \epsilon^{A}_{2}, \ \ldots \epsilon^{A}_{k} $和$\epsilon^{B}_{1}, \epsilon^{B}_{2}, \ \ldots \epsilon^{B}_{k} $其中$\epsilon^{A}_{i}$和$\epsilon^{B}_{i}$是在相同第$i$折上训练得到的结果。
具体来说，首先对每对结果求差$\Delta_i = \epsilon^{A}_{i} - \epsilon^{B}_{i}$, 计算出差值的均值$\mu$和方差$\sigma ^ 2$，在显著度$\alpha $下，若变量
$$
\tau _t = \bigg\lvert \frac{\sqrt{k} \mu}{\sigma} \bigg\rvert
$$
小于临界值$t_{\alpha / 2 ,\ k-1} $，则假设不能被拒绝，即可认为两个学习器的性能没有显著差别。这里$t_{\alpha / 2 ,\ k-1} $是自由度为$k-1$的$t$分布上尾部累积分布为$\alpha /2 $的临界值。

#### 2.4.3 McNemar检验
对于二分类问题
![alt](./ml_image/ml2-4-1.png)
若我们做的假设是两类学习器性能相同，则应有$e_{01} = e_{10} $，那么变量 $ \lvert e_{01} - e_{10} \rvert $应当服从正态分布，且均值为1，方差为$ e_{01} + e_{10}$。因此变量
$$ \tau_{\chi} = \frac {(\lvert e_{01} - e_{10} \rvert -1)^2}{e_{01} + e_{10}}$$
服从自由度为1的$\chi ^2 $分布

#### 2.3.4 Friedman检验与Nememar后续检验
假定我们用$D_1, D_2, D_3, D_4 $四个数据集对算法$A, B, C,$进行比较。首先，使用留出法或交叉验证得到每个算法在每个数据集上的测试结果，然后在每个数据集上根据测试性能由好到坏排序，并赋予序号1，2，3 ...;若算法的性能相同，则平分序号，则可列出下表
![alt](./ml_image/ml2-4-2.png)
然后，使用Friedman检验来判断这些算法是否性能都相同，若相同，则它们的平均序值应当相同。假定我们在$N$个数据集上比较$k$个算法，令$r_i$表示第$i$个算法的平均序值，则$r_i$服从正态分布，其均值和方差分布为$(k+1)/2 $和$(k^2 -1)/12 $。变量
$$
{\tau _{\chi ^2} = \frac{k-1}k \cdot \frac {12N}{k^2 -1} \sum^{k}_{i=1} \Big( r_i - \frac{k+1}2 \Big)^2} \\ {=\frac{12N}{k(k+1)}{\Big( \sum^{k}_{i=1} r^2_i -\frac{k(k+1)^2}{4} \Big) } }
$$
在$k$和$N$都较大时，服从自由度为$k-1$的$\chi ^2$分布。
然而，上述的“原始Friedman检验”过于保守，限制通常使用
$$
\tau _F = \frac{(N-1)\tau_{\chi ^2} }{N(k-1)-\tau_{\chi ^2}}
$$
$\tau_F $服从自由度为$k-1$和$(k-1)(N-1) $的$F$分布

若 _“所有算法性能相同”_ 这个假设被拒绝，则说明算法的性能显著不同。这时需进行“后续检验”(post-hoc test)来进一步区分各算法。常用的为Nemenyi后续检验
Nemenyi检验计算出平均序值差别的临界值域
$$
CD = q_{\alpha}\sqrt{\frac{k(k+1)}{6N}}
$$
若两个算法的平均序值之差超出了临界值域$CD$，则以相同的置信度拒绝 _“两个算法性能相同”_ 这一假设

### 2.5 方差与偏差
对测试样本$\bm{x}$，令$y_D$为$\bm{x}$在数据集中的标记，$y$为$\bm{x}$的真实标记，$f(\bm{x};D) $为训练集$D$上学得模型$f$在$\bm{x}$上的预测输出。以回归任务为例，学习算法的期望预测为
$$
\bar{f}(\bm{x}) = \mathbb{E}_D[f(\bm{x};D)]
$$
使用样本数相同的不同训练集产生的方差为
$$
var(\bm{x}) = \mathbb {E}_D[\big( f(\bm{x};D)- \bar{f}(\bm{x})\big)^2] 
$$
噪声为
$$
\epsilon ^2 = \mathbb {E}_D[(y_D-y)^2]
$$
期望输出与真实标记的差别称为偏差(bias)，即3
$$
bias^2(\bm{x})=(\bar{f}(\bm x)-y)^2
$$

对期望的泛化误差进行分解可得到
$$
E(f;D)= bias^2(\bm x)+var{(\bm x)} + \epsilon ^2
$$
也就是说泛化误差可分解为偏差、方差与噪声之和。


----

## 第3章 线性模型
### 3.1 基本形式
给定由$d$个属性描述的示例$\bm x = (x_1;x_2;\dots;x_d) $，其中$x_i$是$\bm x$在第$i$个属性上的取值，线性模型试图学得一个通过属性的线性组合来进行预测的函数，即
$$
f(\bm x) = \bm{w}^T \bm{x} + b,
$$
其中$\bm{w}= (w_1; w_2;\dots;w_d) $。$\bm{w}$ 和 $b$学得之后，模型就得以确定。

### 3.2 线性回归
给定数据集$D = {(\bm{x}_1,y_1),(\bm{x}_2,y_2), \dots,(\bm{x}_m,y_m)} $，其中$\bm{x}_i = (x_{i1};x_{i2};\dots; x_{id}),\ y_i \in \mathbb R $. “线性回归”(linear regression)试图学得一个线性模型以尽可能准确地预测实值输出标记。

首先考虑最简单的情形：输入的属性数目只有一个，即$D={(x_i,y_i)}^m_{i=1},\ x_i \in \mathbb R $
线性回归试图学得
$$
f(x_i) = w x_i +b , \ s.t.\   f(x_i) \simeq y_i
$$
确定$w $和$b $的关键在于如何衡量$f(x)$与$y$之间的差别。均方误差是回归任务中最常用的性能度量，因此我们试图让均方误差最小化，即
$$
\begin{align*}
(w ^*, b^*) &=\argmin_{(w, b)}\sum^{m}_{i=1}(f(x_i)-y_i)^2 \\ &=\argmin_{(w, b)}\sum^{m}_{i=1}(y_i-w x_i-b)^2 
\end{align*}
$$
基于均方误差最小化来进行模型求解的方法称为最小二乘法(least square method).
求解$w$和$b$使$E_{w,b}=\sum^m_{i=1}(y_i-w x_i-b)^2 $最小化的过程，称为线性回归模型的最小二乘参数估计，我们将$E_{(w,b)}$分别对$w$和$b$求导，得到
$$
\begin{align}
\frac{\partial E_{(w,b)}}{\partial w}&=2\Bigg( w \sum^m_{i=1} x^2_i -\sum^m_{i=1}(y_i -b)x_i \Bigg) \\ \frac{\partial E_{(w,b)}}{\partial w}&=2\Bigg(mb-\sum^m_{i=1}(y_i-wx_i)  \Bigg)
\end{align}
$$
然后令$(1)$和$(2)$为零可得到$w$和$b$最优解的闭式(closed-form)解
$$
w = \frac{\sum^m_{i=1} y_i(x_i-\bar x)}{\sum^m_{i=1}x^2_i- \frac1m\Big(  \sum^m_{i=1} x_i\Big)^2}
$$
其中$\bar{x} = \frac 1m \sum^m_{i=1}x_i $为x的均值。

更一般的情形，样本由$d$个属性描述。此时我们试图学得
$$
f(\bm{x}_i)= bm{w}^t \bm{x}_i +b,\ \ s.t.\ \ f(\bm{x}_i) \simeq y_i
$$
这称为多元线性回归。
为了便于讨论，我们把$w$和$b$吸收入向量形式$\hat{\bm{w}}= (\bm{w}; b) $，相应的把数据集表示为一个$m \times (d+1) $大小的矩阵$\bm{X}$,其中每行对应一个示例
$$
\bm{X}=
\begin{pmatrix}
x_{11} & x_{12} & \dots & x_{1d} & 1 \\ x_{21} & x_{22} & \dots & x_{2d} & 1 \\ \vdots & \vdots & \ddots & \vdots & \ddots \\ x_{m1} & x_{m2} & \dots & x_{md} & 1 
\end{pmatrix}=\begin{pmatrix}
\bm{x}^T_1 & 1 \\ \bm{x}^T_2 & 1 \\ \vdots & \vdots \\ \bm{x}^T_m & 1 
\end{pmatrix}
$$
再把标记也写成向量形式$\bm{y} = (y_1;y_2;\dots;y_m) $，则有
$$
\hat{\bm{w}}^* =\argmin_{\hat{\bm{w}}}(\bm{y}-\bm{X} \hat{\bm{w}})(\bm{y}-\bm{X}\hat{\bm{w}})
$$
令$E_{\hat{\bm{w}}} = (\bm{y}-\bm{X} \hat{\bm{w}})(\bm{y}-\bm{X}\hat{\bm{w}}) $,对$\hat{\bm{w}}$求导得到
$$
\frac{\partial E_{\hat{\bm{w}}}}{\partial \hat{\bm{w}}}=2 \bm{X}^T(\bm{X}\hat{\bm{w}}- \bm{y})
$$

__广义线性模型__:
对数线性回归
$$
\ln y=\bm{w}^T \bm x +b
$$
考虑单调可微函数$g(\cdot)$， 令
$$
\begin{equation}
y=g^{-1}(\bm{w}^T \bm x + b)
\end{equation}
$$
上式形式的模型为广义线性模型，其中函数$g(\cdot)$称为联系函数，显然对数线性回归是广义线性模型在$g(\cdot)= \ln{(\cdot)}$时的特例。

### 3.3 对数几率回归
使用线性模型进行分类任务学习，需要：找一个单调可微函数将分类任务的真实标记$y$与线性回归模型的预测值联系起来。

考虑二分类问题，回归模型产生的预测值$z= \bm{w}^T\bm{x} + b $是实值，将$z$转换为0/1值最理想的是 “单位阶跃函数” (unit-step funciton)
$$
y= \left\lbrace \begin{aligned} {0, \ \ z <0;} \\ {0.5,\ \ z =0;} \\ {1,\ \ z >0}\end{aligned}  \right. 
$$
![alt](./ml_image/ml3-3-1.png)
由于单位阶跃函数不连续，因此不能直接用作$(3)$式中的$g^{-1}(\cdot)$.于是我们使用单调可微的对数几率函数(logistic function)来替代它：
$$
y=\frac{1}{1+e^{-z}}
$$
将对数几率函数带入$(3)$式，得到
$$\begin{equation}
y=\frac{1}{1+e^{-(\bm{w}^T\bm{x}+b)}}
\end{equation}
$$
变形为
$$\begin{equation}
\ln{\frac{y}{1-y}}=\bm{w}^T\bm{x} + b
\end{equation}
$$
若将$y$视为样本$\bm x$作为正例的可能性，则$1-y$是其反例的可能性，两者的比值
$$
\frac{y}{1-y}
$$
称为 __几率__ (odd)，反映了$\bm{x}$作为正例的相对可能性，取对数则得到 __对数几率__，(log odds, 亦称logit)
$$
\ln{\frac{y}{1-y}}
$$
由此可以看出，$(4)$式是在用线性回归的预测结果去逼近真实标记的对数几率，模型称之为 _对数几率回归_(logistic regression)

下面让我们如何确定$(4)$式中的$\bm{w}$和$b$。若将$(4)$式中的$y$视为类后验概率估计$p(y=1|x)$, 则$(5)$式可重写为
$$
\ln \frac{p(y=1)|\bm{x}}{p(y=0)|\bm{x}}=\bm{w}^T\bm{x} + b
$$
显然有
$$
p(y=1|\bm{x})= \frac{e^{\bm{w}^T\bm{x} + b}}{1+e^{\bm{w}^T\bm{x} + b}}
$$

$$
p(y=0|\bm{x})= \frac{1}{1+e^{\bm{w}^T\bm{x} + b}}
$$

于是，我们可以通过极大似然法来估计$\bm{w}$和$b$，给定数据集${(\bm{x}_i, y_i)}^m_{i=1} $，则其最大似然为
$$\begin{equation}
\ell{(\bm{w},b)}=\sum^m_{i=1}\ln {p(y_i|\bm x_i;\bm{w},b)}
\end{equation}
$$
即令每个样本属于真实标记的概率越大越好。为了便于讨论，令$\ \bm{\beta}= (\bm{w};b)$, $\hat{\bm{x}}= (\bm{x};1)$，则$\ \bm{w}^T\bm{x} +b$可简写为$\bm{\beta}^T\hat{\bm{x}}$，再令$p_1(\hat{\bm{x}}; \bm{\beta})=p(y=1|\hat{\bm{x}}; \bm{\beta})$，$p_0(\hat{\bm{x}};\bm{\beta})=p(y=0|\hat{\bm{x}}; \bm{\beta})=1-p_1(\hat{\bm{x}}; \bm{\beta})$则$(6)$式中的似然项可重写为
$$\begin{equation}
p(y_i|\bm{x}_i;\bm{w}, b)=y_i p_1 (\hat{\bm{x}}_i; \bm{\beta})+(1-y_i)p_0(\hat{\bm{x}}_i; \bm{\beta})
\end{equation}
$$
将$(6)$式代入$(5)$式，可得最大化$(5)$等价于最小化
$$\begin{equation}
\ell{(\bm{\beta})}=\sum^m_{i=1}\Big(-y_i \bm{\beta}^T\hat{\bm{x}}_i + \ln \big(1+e^{\bm{\beta}^T\hat{\bm{x}}_i }  \big) \Big)
\end{equation}
$$
$(8)$式是关于$\bm{\beta}$的高阶可导连续凸函数，根据凸优化理论，经典的数值优化算法如梯度下降、牛顿法都快得到最优解，于是得到
$$
\bm{\beta}^*=\argmin_{\beta}\ell (\bm{\beta}).
$$
以牛顿法为例，其第$t+1$轮迭代的更新公式为
$$
\bm{\beta}^{t+1} = \bm{\beta}^{t}-\Big( \frac{\partial^2 \ell (\bm{\beta})}{\partial \bm{\beta} \partial \bm{\beta}^T} \Big)^{-1} \frac{\partial \ell(\bm{\beta})}{\partial \bm{\beta}}
$$
其中关于$\bm{\beta}$的一阶二阶导数分别为
$$\begin{align*}
\frac{\partial \ell (\bm{\beta})}{\partial \bm{\beta}}&= -\sum^m_{i=1}\hat{\bm{x}}_i (y_i - p_1(\hat{\bm{x}}_i ;\bm{\beta})),
\\
\frac{\partial^2 \ell (\bm{\beta})}{\partial \bm{\beta}\partial \bm{\beta}^T}&= -\sum^m_{i=1}\hat{\bm{x}}_i \hat{\bm{x}}_i^T p_1(\hat{\bm{x}}_i ; \bm{\beta}) (1 - p_1(\hat{\bm{x}}_i ;\bm{\beta}))
\end{align*}
$$


### 3.4线性判别分析
线性判别分析(Linear Discriminant Analysis, LDA)最早由Fisher提出，亦称“Fisher判别分析”。LDA的思想为：给定训练样例集，设法将样例投影到一条直线上，使得同类样例尽可能接近、异类样例投影点尽可能远离；在对新样本分类时，将其投影到同样的这条直线，再根据位置来确定新样本的类别。

给定数据集 $D={(\bm{x}_i,y_i)}^m_{i=1},\ y_i \in \{0,1\} $，令$\bm{X}_i,\ \bm{\mu}_i,\ \bm{\Sigma}_i$ 分别表示第 $i\in \{0,1 \} $ 类样本的集合、均值向量、协方差矩阵。则两类样本的中心在直线上的投影分别为$\bm{w}^T \bm{\mu}_0 $ 和 $\bm{w}^T \bm{\mu}_1 $ ；若将所有样本点都投影到直线上，则两类样本的协方差分布为 $\bm{w}^T  \bm{\Sigma}_0 \bm{w}$ 和 $\bm{w}^T  \bm{\Sigma}_1 \bm{w}$ .
欲使同类样本的投影点尽可能接近，可以让同类样本的投影点的协方差尽可能小，$\bm{w}^T  \bm{\Sigma}_0 \bm{w} + \bm{w}^T  \bm{\Sigma}_1 \bm{w}$;
而欲使异类样本的投影点尽可能远离，可让类中心之间的距离尽可能大，即 $\lVert \bm{w}^T \bm{\mu}_0 - \bm{w}^T \bm{\mu}_1 \rVert ^2_2$ 尽可能大。同时考虑二者，则可得到欲最大化的目标
$$\begin{align}
J &= \frac{\lVert \bm{w}^T \bm{\mu}_0 - \bm{w}^T \bm{\mu}_1 \rVert ^2_2}{\bm{w}^T  \bm{\Sigma}_0 \bm{w} + \bm{w}^T  \bm{\Sigma}_1 \bm{w}} \notag
\\
&= \frac{\bm{w}^T (\bm{\mu}_0 - \bm{\mu}_1)(\bm{\mu}_1 - \bm{\mu}_0)^T\bm{w}}{\bm{w}^T(\bm{\Sigma}_0 + \bm{\Sigma}_1)\bm{w}}.
\end{align}
$$

定义“类内散度矩阵”(within-class scatter matrix)
$$
\begin{align*}
\bm{S}_{w} &= \bm{\Sigma}_0 + \bm{\Sigma}_1
\\
&=\sum_{\bm{x}\in X_0}(\bm{x} - \bm{\mu}_0)(\bm{x} - \bm{\mu}_0)^T + \sum_{\bm{x}\in X_1}(\bm{x} - \bm{\mu}_1)(\bm{x} - \bm{\mu}_1)^T
\end{align*}
$$

以及“类间散度矩阵”(between-class scatter matrix)
$$
\bm{S}_b = (\bm{\mu}_0 - \bm{\mu}_1)(\bm{\mu}_0 - \bm{\mu}_1)^T
$$
则$(9)$式可重写为
$$\begin{equation}
J = \frac{\bm{w}^T \bm{S}_b \bm{w}}{\bm{w}^T \bm{S}_{w} \bm{w}}
\end{equation}
$$
如何确定  $\bm{w}$ 呢？注意到$(10)$式分子分母都是关于 $\bm{w}$ 的二次型，因此$(10)$式的解与 $\bm{w}$ 的方向有关。不失一般性，令 $\bm{w}^T \bm{S}_b \bm{w} = 1$，则$(10)$式等价于
$$
\begin{align*}
\min_{\bm w} \ \ & -\bm{w}^T \bm{S}_b \bm{w} \\
s.t. \ \ &\  \bm{w}^T \bm{S}_{w} \bm{w}
\end{align*}
$$
由拉格朗日乘子法，上式等价于
$$\begin{equation}
\bm{S}_b \bm{w} = \lambda \bm{ S}_{\bm{w}} \bm{w},
\end{equation}
$$
其中 $\lambda$ 是拉格朗日乘子，注意到 $\bm{S}_b \bm{w}$ 的方向恒为 $\bm{\mu}_0 -\bm{\mu}_1$，不妨令
$$
\bm{S}_b \bm{w} = \lambda(\bm{\mu}_0 -\bm{\mu}_1)
$$
代入第 $(11)$ 式即得
$$
\bm {w} = \bm{S}_{w}^{-1}(\bm{\mu}_0 -\bm{\mu}_1)
$$


将LDA推广到多分类任务中，假定存在 $N$ 个类，且第 $i$ 类样本数为 $m_i$ 。我们有全局散度矩阵：
$$\begin{align*}
\bm{S}_t &= \bm{S}_b + \bm{S}_w \\
&= \sum^m{i=1}(\bm{x}_i- \bm{\mu})(\bm{x}_i -\bm{\mu})^T
\end{align*}
$$
其中 $\bm\mu$ 是所有样本的均值向量。将类内散度矩阵 $ \bm{S}_w$ 定义为每个类别的散度矩阵之和，即
$$
\bm{S}_w = \sum^m_{i=1} \bm{S}_{w_i}
$$
由以上两式可得：
$$\begin{align*}
\bm{S}_b &= \bm{S}_t - \bm{S}_w  \\
&= \sum^N_{i=1} m_i(\bm{\mu}_i - \bm{\mu})(\bm{\mu}_i - \bm{\mu})^T
\end{align*}
$$

显然，多分类LDA可以有多种实现方法：使用 $\bm{S}_b, \bm{S}_t, \bm{S}_w$ 三者中的任何两个即可。常见的一种实现是采用优化目标
$$
\max_{\bm{W}}{\frac{tr(\bm{W}^T \bm{S}_b \bm{W})}{tr(\bm{W}^T \bm{S}_w \bm{W})}}
$$
其中 $\bm{W} \in \mathbb{R}^{d \times (N-1)},tr(\cdot) $表示矩阵的迹(trace). 上式可通过如下广义特征值问题来求解：
$$
\bm{S}_b \bm{W} = \lambda \bm{S}_w \bm{W}
$$

$W$的闭式解则是 $\bm{S}^{-1}_{w} \bm{S}_b$ 的 $N-1$ 个最大广义特征值所对应的特征向量组成的矩阵。


### 3.5 多分类学习
对于多分类学习，我们通常基于一些基本策略，利用二分类学习器来解决。
最经典的拆分策略有三种：“一对一”(One vs. One, OvO)、“一对其余”(One vs. Reat, OvR)和“多对多”(Many vs. Many, MvM).
给定数据集 $D={(\bm{x}_i,y_i)}^m_{i=1},\ y_i \in \{C_1, C_2, \dots, C_N\}. $ 
- OvO将这个$N$类别两两配对，从而产生 $N(N-1)/2$ 个分类任务和分类器。在测试阶段，新样本被同时提交给所有分类器，于是我们将得到 $N(N-1)/2$ 个分类结果，最终结果可通过投票产生.
- OvR则是每次将一个类的样例作为正例、所有其他类的样例作为反例来训练 $N$ 个分类器。在测试时若仅有一个分类器预测为正类，则对应的类别标记为最终分类结果；若有多个分类器预测为正类，则通常考虑各分类器的预测置信度，选择置信度最大的类别标记作为分类结果。
![alt](./ml_image/ml3-5-1.png)
- MvM是每次将若干个类作为正类，若干个其他类作为反类。MvM的正、反类构造必须有特殊设计，这里我们介绍一种最常用的MvM技术：“纠错输出码”(Error Correcting Output Codes, ECOC)
>1. 编码：对 $N$ 个类别做$M$次划分，每次将一部分类别划分为正类，一部分划分为反类，从而形成一个二分类训练集，这样一共产生$M$个训练集，训练出$M$个分类器。
>2. 解码：$M$个个分类器分别对测试样本进行预测，这些预测标记组成一个编码。将这个预测编码于每个类别各自的编码进行比较，返回其中距离最小的类比作为最终预测结果。

### 3.6 类别不平衡问题
类别不平衡(class-imbalance)就是指分类任务中不同类别的训练样例数目差别很大的情况。
__再放缩__ 见p.67



---
## 第4章 决策树
### 4.1 基本流程
一般的，一颗决策树包含一个根结点、若干个内部结点和若干个叶结点；叶结点对应于决策结果，其他每个结点对应于一个属性测试。
显然，决策树的生成是一个递归的过程，有三种情形会导致递归返回：
>1. 当前结点包含的样本全属于同一类别，无需划分；
>2. 当前属性集为空集，或是所有样本在所有属性上取值相同，无法划分；
>3. 当前结点包含的样本集为空，不能划分

在第(2)种情形下，我们把当前结点标记为叶结点，并将其类别设定为该结点所包含样本最多的类别；在第三种情形下，同样把当前结点标记为叶结点，但将其类别设定为其父结点所包含样本最多的类别。
_注：情形(2)是在利用当前结点的后验分布，而情形(3)则是把父结点的样本分布作为当前结点的先验分布。
![alt](./ml_image/ml4-1-1.png)

### 4.2划分选择
决策树学习的关键是第8行，即如何选择最优划分属性。

#### 4.2.1 信息增益
__信息熵__(infromation entropy)是度量样本集合纯度最常用的一种指标。假设当前样本集合 $D$ 中第 $k$ 类样本所占的比例为 $p_k(k=1,2,\dots, \  |\mathcal{Y}|)$，则$D$的信息熵定义为
$$\begin{equation}
Ent(D) = -\sum^{|\mathcal{Y}|}_{k=1}p_k \log_2 p_k \ . \end{equation}
$$
$Ent(D)$的值越小，则$D$的纯度越高。
假定离散属性$a$有$V$个可能的取值${a^1, a^2, \dots, a^V}$，若使用$a$来对样本集进行划分，则会产生$V$个分支结点，其中第$v$个分支结点包含了$D$中所有在属性$a$上取值为$a^v$的样本，记为$D^v$. 我们可根据$(12)$式计算出$D^v$的信息熵，给分支结点赋予权重$|D^v|/|D|$，即样本越多的分支结点的影响越大，于是可计算出用属性$a$对样本集$D$进行划分所获得的“信息增益”(information gain)
$$\begin{equation}
Gain(D,a)=Ent(D)-\sum^V_{v=1}\frac{|D^v|}{|D|}Ent(D^v)
\end{equation}
$$
一般而言，信息增益越大，则意味着使用属性$a$来划分所获得的“纯度提升越大”。因此，我们可用信息增益来进行决策树树的划分属性选择，即在图4.2算法第8行选择属性
$$\argmax_{a\in A} Gain(D,a)$$
著名的ID3决策树学习算法就是以信息增益为准则来选择划分属性。

#### 4.2.2 增益率
实际上，信息增益准则对可取值数目较多的属性有所偏好，为减少这种偏好可能带来的不利影响，著名的C4.5决策树算法使用“增益率来选择最优划分属性”。增益率定义为
$$
Gain_ratio(D,a) = \frac{Gain(D,a)}{{\rm{IV}} (a)}
$$

其中

$$
{\rm IV} (a) = -\sum^V_{v=1}\frac{|D^v|}{|D|}\log_2\frac{|D^v|}{|D|}
$$
称为属性$a$的“固有值”(intrinsic value). 属性$a$的可能取值数目越多(即V越大)，则${\rm IV} (a)$的值通常会越大。

徐注意的是，增益率准则对可取值数目较少的属性有所偏好，因此，C4.5算法并不是直接选择增益率最大的候选划分属性，而是使用了一个启发式：先从候选划分属性中找出信息增益高于平均水平的属性，再从中选择增益率最高的。

#### 4.2.3 基尼指数
CART决策树使用“基尼指数”(Gini index)来选择划分属性。数据集$D$的纯度可用基尼值来度量：
$$
\begin{align*}
Gini(D) &= \sum^{|\mathcal Y|}_{k=1} \sum_{k' \ne k} p_k p_{k'}\\
&=1-\sum^{\mathcal{|Y|}}_{k=1} p^2_k
\end{align*}
$$

直观来说，$Gini(D)$反映了从数据集$D$中随机抽取两个样本，其类别标记不一致的概率。因此，$Gini(D)$越小，则数据集$D$的纯度越高。
属性$a$的基尼指数定义为
$$
Gini\_index(D,a) = \sum^V_{v=1} \frac{D^v}{D}Gini(D^v)
$$
于是，我们在候选属性集合$A$中，选择那个使得划分后基尼指数最小的属性作为最优划分属性，即
$$a_* = \argmin_{a \in A} Gini\_index(D,a)$$


### 4.3 剪枝处理
剪枝(pruning)是决策树学习算法应对过拟合的主要手段。
决策树剪枝的基本策略有“预剪枝”(prepruning)和“后剪枝”(post-pruning). 
>__预剪枝__：在决策树生成过程中，对每个结点在划分前进行估计，若当前结点的划分不能带来决策树泛化性能的提升，则停止划分并将当前结点标记为叶结点。
>__后剪枝__：先从训练集生成一棵完整的决策树，然后自底向上地对非叶结点进行考察，若将该结点对应的子树替换为叶结点能带来决策树泛化性能提升，则将该子树替换为叶结点。

如何判断决策树泛化性能是否提升呢，这可使用2.2节的性能评估方法，使用验证集进行性能评估。

#### 4.3.1 预剪枝
详见p.81

#### 4.3.2 后剪枝
详见p.82


### 4.4 连续与缺失值
#### 4.4.1 连续值处理
最简单的策略是采用二分法(bi-partition)对连续属性进行处理。

给定样本集$D$和连续属性$a$，假定$a$在$D$上出现了$n$个不同的取值，将这些值从小到大进行排序。记为$\{a^1, a^2, \dots, a^n\}.$ 基于划分点$t$可将$D$分为子集$D^-_t$和$D^+_t$，其中$D^-_t$包含那些在属性$a$上取值不大于$t$的样本，则$D^+_t$包含那些在属性$a$上取值大于$t$的样本。显然，对相邻的属性取值$a^i$与$a^{i+1}$来说，$t$在区间$[a^i,a^{i+1})$中取任意值所产生的划分结果相同。我们考察包含$n-1$个元素的的候选划分点集合
$$
T_a = \Big\{ \frac{a^i + a^{i+1}}{2}|1 \leqslant i \leqslant n-1 \Big\}
$$
即把区间$[a^i,a^{i+1})$的中位点$\frac{a^i + a^{i+1}}{2}$作为候选划分点，然后我们就可以像离散属性值一样来考察这些划分点，选取最优的划分点进行样本集合的划分。例如，可对$(13)$式稍加改造
$$\begin{align*}
Gain(D,a) &= \max_{t \in T_a} Gain(D,a,t) \\
&= \max_{t \in T_a} Ent(D)-\sum^V_{\lambda \in\{-, +\}}\frac{|D^v|}{|D|}Ent(D^v)
\end{align*}
$$

#### 4.4.2 缺失值处理
我们需要解决如下两个问题
> 1. 如何在属性值缺失的情况下进行划分属性选择
> 2. 给定划分属性，若样本在该属性上的值缺失，如何对样本进行划分？

给定训练集$D$和连续属性$a$， 令$\tilde{D}$表示$D$中在属性$a$上没有缺失值的样本子集。显然，我们仅可根据$\tilde{D}$来判断属性$a$的优劣。假定属性$a$有$V$个可取值$\{a^1, a^2, \dots, a^V\}$，令$\tilde{D}^v$表示$\tilde{D}$中在属性$a$上取值为$a^v$的样本子集，$\tilde{D}_k$表示$\tilde{D}$中属于第$k$类$(k =1,2,\dots, |\mathcal{Y}|)$的样本子集，则显然有$\tilde{D}= \bigcup^{\mathcal{|Y|}}_{k=1} \tilde{D}_k,\ \tilde{D}= \bigcup^{V}_{v=1} \tilde{D}_v.$ 假定我们为每个样本$\bm x$赋予一个权重$w_{\bm x}$，并定义
$$
\begin{align*}
\rho &= \frac{\sum_{{\bm x} \in \tilde{D} }w_{\bm x}}{\sum_{{\bm x} \in D }w_{\bm x}},\\
\tilde{p}_k &= \frac{\sum_{{\bm x} \in \tilde{D}_k }w_{\bm x}}{\sum_{{\bm x} \in \tilde{D} }w_{\bm x}}\ \  (1 \leqslant k \leqslant \mathcal{|Y}|)\\
\tilde{r}_v &= \frac{\sum_{{\bm x} \in \tilde{D}^v }w_{\bm x}}{\sum_{{\bm x} \in \tilde{D} }w_{\bm x}}\ \  (1 \leqslant k \leqslant V)
\end{align*}
$$

直观地看，对属性$a, \rho$表示无缺失值样本所占的比例，$\tilde{p}_k$表示无缺失值样本中第$k$类所占的比例，$\tilde{r}_v$则表示无缺失值样本在属性$a$上取值$a^v$的样本所占的比例。显然$\sum^{\mathcal{|Y|}}_{k=1} = 1$, $\sum^V_{v=1}\tilde{r}_v =1$.
基于上述定义，我们可以将信息增益的计算式(13)推广为
$$
\begin{align*}
Gain(D,a) &= \rho \times Gain(\tilde{D}, a)\\
&=\rho \times \Big( Ent\big(\tilde{D}\big) - \sum^V_{v=1}\tilde{r}_v Ent\big( \tilde{D}^v\big) \Big)
\end{align*}
$$
其中由(12)式，有
$$
Ent(\tilde{D}) = - \sum^{|\mathcal{Y}|}_{k=1}\tilde{p}_k \log_2 \tilde{p}_k . 
$$

对于问题2
- 若样本$\bm{x}$在划分属性$a$上的取值已知，则将$\bm{x}$划入与其取值对应的子结点，且样本权值在子结点中保持为$w_{\bm x}$
- 若样本$\bm{x}$在划分属性$a$上的取值未知，则将$\bm{x}$同时划入所有子结点，且样本权值在与属性值$a^v$对应的子结点中调整为$\tilde{r}_v \cdot w_{\bm x}$

C4.5算法使用了上述解决方案。

### 4.5 多变量决策树
若我们把每个属性视为坐标空间中的一个坐标轴，则$d$个属性描述的样本就对应了$d$维空间中的一个数据点，对样本分类则意味这在这个坐标空间中寻找不同类样本之间的分类边界。
决策树所形成的分类边界有一个明显的特点：轴平行(axis-parallel)，即它的分类边界由若干个与坐标轴平行的分段组成。
![alt](./ml_image/ml4-5-1.png)

若能使用斜的划分边界，如上图红线所示，则决策树模型将大幅简化。“多变量决策树”(multivariate decision tree)就是能实现这样的“斜划分”甚至更复杂划分的决策树。
在此类决策树中，非叶结点不再是仅对某个属性，而是对属性的线性组合进行测试，换言之，每个非叶结点是一个形如$\sum^{d}_{d=1} w_i a_i = t $的线性分类器，其中$w_i$是属性$a_i$的权重。



## 第5章 神经网络
### 5.1 神经元模型
神经网络中最基本的成分是神经元(neuron)模型。
在生物神经网络中，每个神经元与其它神经元相连，每当他兴奋时，就会向相连的神经元发送化学物质，从而改变这些神经元内的电位，如果某神经元的电位超过了一个“阈值”(threshold)，那么它就会被激活，向其他神经元发送化学物质。
![alt](./ml_image/ml5-1-1.png)
上图展示的就是“M-P神经元模型”，神经元接收到来自$n$个其他神经元传递过来的信号，通过带权重的连接进行传递，神经元将总输入与神经元阈值相比，然后通过“激活函数”(activation funciton)处理以产生神经元输出。
理想中的激活函数是下图左的阶跃函数，他将输入值映射为输出值0或1，但是由于阶跃函数不连续、不光滑，因此实际常用Sigmoid函数作为激活函数。
![alt](./ml_image/ml5-1-2.png)


### 5.2 感知机与多层神经网络
感知机(Perceptron)由两层神经元组成，输入层接受外界输入信号后传递给输出层，输出层是M-P神经元，亦称“阈值逻辑单元”(threshold logic unit)
![alt](./ml_image/ml5-2-1.png)
感知机能容易地实现逻辑与、或、非运算。注意到$y= f(\sum_i{w_i x_i - \theta})$，假定$f$是图5.2中的阶跃函数，有
- “与”($x_1 \wedge x_2)$：令$w_1 = w_2 = 1 ，\theta = 2$，则$y=f(1 \cdot x_1 + 1 \cdot x_2 -2)$，仅在$x_1 = x_2 = 1$时，$y=1$；
- “或”($x_1 \vee x_2)$：令$w_1 = w_2 = 1，\theta = 0.5$，则$y=f(1 \cdot x_1 + 1 \cdot x_2 -0.5)$，当$x_1 = 1$ 或$x_2 = 1$时，$y=1$；
- “非”($\neg x_1)$：令$w_1 = 0.6 w_2 = 0，\theta = -0.5$，则$y=f(-0.6 \cdot x_1 + 0 \cdot x_2 0.5)$，当$x_1 = 1$时，$y=0$，当$x_1 = 0$时，$y=1$；

更一般地，给定训练数据集，权重$w_i(i=1, 2, \dots, n)$以及阈值$\theta$可通过学习得到。阈值$\theta$可看作一个固定输入为-1.0的哑结点(dummy node)所应的连接权重$w_{n+1}$,这样权重和阈值的学习就可统一为权重的学习。

感知机的学习规则非常简单，对训练样例$(\bm{x},y)$,若当前感知机的输出为$\hat{y}$,则感知机权重将这样调整：
$$\begin{align}
w_i \leftarrow w_i + \Delta w_i \\
\Delta w_i = \eta(y-\hat{y})x_i
\end{align}
$$
其中$\eta in (0,1)$称为学习率(learning rate)

>需要注意的是，感知机只有输出层神经元进行函数激活处理，即只拥有一层功能神经元(function neuron)。

要解决非线性可分问题，需要考虑使用多层功能神经，例如图5.5，输入层和输出层之间的一层神经元，被称为隐层或者隐含层(hidden layer)，隐层和输出层都是拥有激活函数的功能神经元。

![alt](./ml_image/ml5-2-2.png)

更一般的，常见的神经网络是形如图5.6所示的层级结构，每层神经元与下一层全连接，神经元间不存在同层连接，也不存在跨层连接，这样的结构被称为“多层前馈神经网络”(multi-layer feedforward neural networks)
![alt](./ml_image/ml5-2-3.png)

### 5.3 误差逆传播算法
欲训练多层网络，$(14)$式这样的学习规则显然不够了，需要更强大的学习算法。误差逆传播(error BackPropagation, 反向传播， BP)算法。

给定训练集$D=\{(\bm{x}_1, \bm{x}_1), (\bm{x}_2, \bm{x}_2),\dots,(\bm{x}_m, \bm{x}_m) \},\bm{x}_i \in \mathbb{R}^d ,\bm{y}_i \in \mathbb{R}^l$，即输入示例由$d$个属性描述，输出$l$维实值向量。
图5.7给出了一个拥有$d$个输入神经元、$l$个输出神经元、$q$个隐含神经元的多层前馈网络结果，其中输出层第$j$个神经元的阈值用$\theta_j$表示，隐层第$h$个神经元的阈值用$\gamma_h$表示。输入层第$i$个神经元与隐层第h个神经元之间的连接权为$v_{ih}$。记隐层第$h$个神经元接收到的输入为$\alpha_h = \sum^d_{i=1}v_{ih}x_i$,输出层第$j$个神经元接收到的输入为$\beta_j = \sum^q_{h=1}w_{hj}b_h$，其中$b_h$为隐层第$h$个神经元的输出。假如隐层和输出层都使用Sigmoid函数。
![alt](./ml_image/ml5-3-1.png)

对训练例$(\bm{x}_k, \bm{y}_k)$假定神经网络的输出为$\hat{\bm{y}}_k = (\hat{y}^k_1,\hat{y}^k_2,\dots, \hat{y}^k_l)$,即
$$
\hat{y}^k_j = f(\beta_j - \theta_j)
$$

则网络在$(\bm{x}_k, \bm{y}_k)$上的均方误差为
$$
\begin{equation}
E_k = \frac 12 \sum^l_{j=1}(\hat{y}^k_j-\hat{y}^k_j)^2
\end{equation}
$$
图5.7的网络中有$(d+l+1)q+l$个参数需确定：输入层到隐层的$d\times q$个权值、隐层到输出层的$q \times l$个权值、$q$个隐层神经元的阈值、$l$个输出层神经元的阈值。
BP是一个迭代学习算法，在迭代的每一轮中采用广义的感知机学习规则对参数进行更新估计，即与$(14)$式类似，任意参数$v$的更新估计式为
$$
v \leftarrow v + \Delta v
$$

BP算法基于梯度下降(gradient descent)策略，以目标的负梯度方向对参数进行调整. 对(16)式的的误差$E_k$，给定学习率$\eta$，有
$$
\Delta w_{hj} = -\eta \frac{\partial E_k}{\partial w_{hj}}. 
$$
注意到$w_{hj}$先影响到第$j$个输出神经元的输入值$\beta_j$，再影响到其输出值$\hat{y}^k_j$，然后影响到$E_k$，有
$$
\begin{equation}
\frac{\partial E_k}{\partial w_{hj}}=\frac{\partial E_k}{\partial \hat{y}^k_j} \cdot \frac{\partial \hat{y}^k_j}{\partial \beta_j} \cdot \frac{\partial \beta_j}{\partial w_{hj}}
\end{equation}
$$

根据$\beta_j$的定义，显然有
$$
\frac{\partial \beta_j}{\partial w_{hj}} = b_h
$$
Sigmoid函数有一个很好的性质：
$$
f'(x) = f(x)(1-f(x))
$$
于是根据前面的式子，有
$$
\begin{align*}
g_j &= - \frac{\partial E_k}{\partial \hat{y}^k_j} \cdot \frac{\partial \hat{y}^k_j}{\partial \beta_j}
\\
&= -(\hat{y}^k_j -y^k_j)f'(\beta_j -\theta_j)
\\
&= \hat{y}^k_j(1-\hat{y}^k_j)(y^k_j-\hat{y}^k_j)
\end{align*}
$$
综合上述各式，得到BP算法中关于$w_{hj}$的更新公式
$$
\Delta w_{hj} = \eta g_j b_h
$$
类似可得

$$
\begin{align*}
\Delta \theta _j &= - \eta g_j,
\\
\Delta v_{ih} &= \eta e_h x_i,
\\
\Delta \gamma _h &= - \eta e_h 
\end{align*}
$$

其中
$$
\begin{align*}
e_h &= -\frac{\partial E_k}{\partial b_h} \cdot \frac{\partial b_h}{\partial \alpha_h}
\\
&= -\sum^l_{j=1}\frac{\partial E_k}{\partial \beta_j} \cdot \frac{\partial \beta_j}{\partial b_h} f'(\alpha_h - \gamma_h)
\\
&= \sum^l_{j=1} w_{hj} g_j f'(\alpha_h - \gamma_h)
\\
&= b_h(1-b_h) \sum^l_{j=1} w_{hj} g_j
\end{align*}
$$
下图给出了BP算法的工作流程。对每个训练样本，BP算法执行以下操作：先将输入样本提供给输入层神经元，然后逐层将信号前传，直到产生输出结果；然后计算输出层的误差，再将误差逆向传播至隐层神经元，最后根据隐层神经元的误差来对连接权和阈值进行调整。
![alt](./ml_image/ml5-3-2.png)

需注意的是，BP算法的目标是要最小化训练集$D$上的累积误差
$$
E = \frac 1m \sum^m_{k=1} E_k
$$
但我们上面介绍的“标准BP算法”每次仅针对一个训练样本更新连接权和阈值，也就是是说，图5.8的更新规则是基于单个的$E_k$推导而得。如果类似地推导出基于累积误差最小化的更新准则，就得到了累积误差逆传播(accumulated error backpropagation)算法。
一般来说，标准BP算法每次更新只针对单个样本，参数更新十分频繁，且对不同样例进行更新的效果可能出现抵消现象。累积BP算法直接针对累积误差最小化，它在读取整个训练集$D$一遍后才对参数进行更新，其参数更新频率低得多。

BP神经网络经常遭遇过拟合，有两种策略缓解该问题：
1. __早停__ (early stopping)：将数据分为训练集和验证集训练集用来计算梯度、更新连接权和阈值，验证集用来估计误差，若训练集误差降低但验证集误差升高，则停止训练，同时返回具有最小验证集误差的连接权和阈值。
2. __正则化__ (regularization)：在误差目标函数中增加一个用于描述网络复杂度的部分，例如连接权与阈值的平方和。仍令$E_k$表示第$k$个训练样例上的误差，则误差目标函数改变为
$$
E = \lambda \frac 1m \sum^m_{k=1}E_k + (1-\lambda)\sum_i w^2_i
$$
其中$\lambda \in (0,1)$用于对经验误差与网络复杂度这两项进行折中，通常通过交叉验证法来估计。

### 5.4 全局最小与局部最小
跳出局部极小的策略：
- 以多组不同发参数值初始化多个神经网络，按标准方法训练后，取其中误差最小的解作为最终参数，
- 使用 __模拟退火(simulated annealing)__ 技术模拟退火在每一步都以一定的概率接收比当前更差的结果，从而有助于跳出局部极小。
- 使用随机梯度下降。


### 5.5 其他常见神经网络
#### 5.5.1 RBF网络
RBF(Radial Basis Function, 径向基函数)网络，是一种单隐层前馈神经网络，它使用径向基函数作为隐层神经元激活函数，而输出层是对隐层神经元输出的线性组合。假定输入为$d$维向量$\bm x$，输出为实值，则RBF网络可表示为
$$
\phi(\bm x) = \sum^q_{i=1} w_i \rho(\bm x, \bm{c}_i),
$$
其中q为隐层神经元个数，$\bm{c}_i$和$w_i$分别是第$i$个隐层神经元所对应的中心和权重，$\rho(\bm x, \bm{c}_i)$是径向基函数，这是某种沿径向对称的标量函数，通常定义为样本$\bm x$到数据中心$bm{c}_i$之间欧式距离的单调函数。常用的高斯径向基函数形如
$$
\rho(\bm x, \bm{c}_i) = e^{-\beta_i \lVert \bm{x}-\bm{c}_i \rVert ^2}. 
$$

#### 5.5.2 ART网络