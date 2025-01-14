# DeepseekV2RMSNorm

下面给出 **DeepseekV2RMSNorm** 的数学公式、代码实现，以及在 PyTorch 中一些 **关键细节的高亮**。

---

## 1. 数学公式

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

## 2. PyTorch 代码实现

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
