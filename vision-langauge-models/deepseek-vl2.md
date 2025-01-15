# DeepseekV2 要点

## 1. RMSNorm

下面给出 **DeepseekV2RMSNorm** 的数学公式、代码实现，以及在 PyTorch 中一些 **关键细节的高亮**。

---

### 1. 数学公式

给定输入向量 $\mathbf{x} \in \mathbb{R}^d$，令可学习参数为 $\boldsymbol{\gamma} = (\gamma_1, \gamma_2, \ldots, \gamma_d)$。  
RMSNorm 的输出 $\mathbf{y} \in \mathbb{R}^d$ 定义如下：

$$
\mathbf{y} 
= \mathrm{RMSNorm}(\mathbf{x}) 
= \left(\frac{\mathbf{x}}{\sqrt{\mathrm{mean}\!\bigl(\mathbf{x}^2\bigr) + \varepsilon}}\right)
\odot \boldsymbol{\gamma},
$$

其中：

- $\mathrm{mean}\!\bigl(\mathbf{x}^2\bigr) = \frac{1}{d}\sum_{j=1}^d x_j^2$
- $\varepsilon$ 是一个极小正数，用于数值稳定，避免分母为 0
- $\odot$ 表示 **逐元素** (element-wise) 相乘

换言之，对于任意第 $i$ 个分量，有

$$
y_i 
= \gamma_i \cdot \frac{x_i}{\sqrt{\frac{1}{d}\sum_{j=1}^d x_j^2 + \varepsilon}}
\,.
$$

---

### 2. PyTorch 代码实现

以下是一个可参考的 PyTorch 实现。它与上面数学公式**一一对应**，并在计算前 **将张量强制转换为 `float32`**（防止半精度 / 混合精度下的数值不稳定），计算完成后再转换回原始 dtype：

```python
import torch
import torch.nn as nn

class DeepseekV2RMSNorm(nn.Module):
    def __init__(self, hidden_size, eps=1e-6):
        """
        DeepseekV2RMSNorm is equivalent to T5LayerNorm (RMSNorm variant).
        """
        super().__init__()
        self.weight = nn.Parameter(torch.ones(hidden_size))
        self.variance_epsilon = eps

    def forward(self, hidden_states):
        # --- [1] 将输入转换为fp32，避免数值误差 ---
        input_dtype = hidden_states.dtype
        hidden_states = hidden_states.to(torch.float32)  # **在这里强制转换为 fp32**

        # --- [2] 计算均方 (非 "减均值" 的方差) ---
        variance = hidden_states.pow(2).mean(-1, keepdim=True)

        # --- [3] 将输入除以 sqrt(均方 + ε) ---
        hidden_states = hidden_states * torch.rsqrt(variance + self.variance_epsilon)

        # --- [4] 乘以可学习的 weight，并转换回原始 dtype ---
        return self.weight * hidden_states.to(input_dtype)
```

## 2. Gated MLP
### 1. 和vanilla MLP的区别

在 Transformer 等模型的 MLP（前馈网络，FFN）部分中，引入 **门控（Gated）** 机制，即在中间层使用两条并行映射并通过逐元素相乘来输出，往往能带来下列优势：

1. **更灵活的特征表达**  
   在一个门控结构中，一条分支会经过激活函数（如 GELU、Swish），另一条分支不做激活，然后二者做逐元素乘。这使得网络在某些通道上可以“选择性”地抑制或放大信息，而不只是统一地将所有分量都通过同一个激活函数。

2. **可能带来更优的收敛与效果**  
   研究表明，像 GLU、SwiGLU、GeGLU 等门控型前馈网络在大规模训练时往往比单一激活 MLP 更稳定，并且在相同或稍高的参数量下往往能获得更好的下游性能。

3. **保留线性分量，提升信息流动**  
   普通激活函数（如 ReLU、GELU）会对输入做统一的非线性变换，而门控结构则在保留一部分线性输出（不经激活）的同时，对另一部分进行非线性控制，实现更丰富的表示能力。

4. **与注意力互补**  
   注意力机制主要在序列维度上做动态选择，而门控结构则让网络在特征通道上进一步做“门控”选择。这种两者结合的模式在当前大语言模型和其他现代 Transformer 架构中被广泛使用。

简而言之，**门控 MLP**（如 GeGLU、SwiGLU）利用一条激活分支和一条线性分支的逐元素相乘机制，让网络在中间层具有更大的表达自由度和信息选择能力，从而在实践中往往取得更好的性能或稳定性。


```python
class DeepseekV2MLP(nn.Module):
    def __init__(self, config, hidden_size=None, intermediate_size=None):
        super().__init__()
        self.config = config
        self.hidden_size = config.hidden_size if hidden_size is None else hidden_size
        self.intermediate_size = (
            config.intermediate_size if intermediate_size is None else intermediate_size
        )

        self.gate_proj = nn.Linear(self.hidden_size, self.intermediate_size, bias=False)
        self.up_proj = nn.Linear(self.hidden_size, self.intermediate_size, bias=False)
        self.down_proj = nn.Linear(self.intermediate_size, self.hidden_size, bias=False)
        self.act_fn = ACT2FN[config.hidden_act]

    def forward(self, x):
        down_proj = self.down_proj(self.act_fn(self.gate_proj(x)) * self.up_proj(x))
        return down_proj
```

### 2. 门控 MLP（GLU 家族）常见激活函数

在现代大语言模型及各种 Transformer 变体中，**门控 MLP**（即 GLU 类）经常采用以下三种激活函数来构造“门控”分支：

1. **ReGLU**  
   - 使用 ReLU 激活：  
     $$
     \text{gate} = \mathrm{ReLU}(W_g x + b_g), 
     \quad
     \text{up} = W_u x + b_u.
     $$
     然后将 gate 与 up 做逐元素相乘，再用 $W_d$ 投影回输出。

2. **GeGLU**  
   - 使用 GeLU 激活：  
     $$
     \text{gate} = \mathrm{GeLU}(W_g x + b_g), 
     \quad
     \text{up} = W_u x + b_u.
     $$
     GeLU 对负值较为平滑地衰减，常在大规模模型中带来稳定的训练效果。

3. **SwiGLU**  
   - 使用 Swish 激活：  
     $$
     \text{gate} = \mathrm{Swish}(W_g x + b_g) 
     = (W_g x + b_g)\,\sigma(W_g x + b_g), 
     \quad
     \text{up} = W_u x + b_u.
     $$
     Swish 可以在 $z>0$ 时近似保留 $z$，在 $z<0$ 时平滑衰减，常常能带来良好的收敛表现。

这些门控激活都遵循相同的核心结构：
- **一条线性分支**：$\text{up} = W_u x + b_u$  
- **一条门控分支**：$\text{gate} = \text{Activation}(W_g x + b_g)$  
- **逐元素相乘**：$\text{mid} = \text{gate} \odot \text{up}$  
- **投影回输出**：$\text{out} = W_d \,\text{mid} + b_d$

也正是通过这种 “**门控** + **线性**” 的设计，网络能在前馈层（MLP/FFN）中实现更灵活的特征选择与表达，较传统单一激活的 MLP 往往带来更优的性能与稳定性。
