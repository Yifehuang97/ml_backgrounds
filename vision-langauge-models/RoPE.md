---
title: RoPE（Rotary Position Embedding）原理与推导
author: Yifeng Huang
date: 2025-01-18
---

# RoPE（Rotary Position Embedding）原理与推导

在 Transformer 等自注意力模型中，如果能够让注意力打分内积“天然”地依赖相对位置 \((m-n)\)，往往可以获得更好的长程依赖及泛化能力。**RoPE**（Rotary Position Embedding）提出了一种将位置信息编码为“向量旋转”的方法，使得在做内积时，位置差以“相位差”的形式自然出现，下面我们依次介绍该原理的推导、在高维上的具体实现，以及常见的辅助函数 `rotate_half` 等。

---

## 1. 背景与总体目标

### 1.1 目标

令 \(q, k\) 分别表示 Query / Key 向量，\(m, n\) 表示它们所在的位置，希望有某种函数 \(f\)，在**做内积时**：

\[
\mathrm{Re}\bigl[f(q,m)\,f^*(k,n)\bigr] 
\;=\; g\bigl(q,\,k,\,(m-n)\bigr),
\]

其中  
- \(f(q,m)\) 表示“给向量 \(q\) 加入位置 \(m\) 的编码”；  
- \(f^*(k,n)\) 表示对应的复共轭；  
- 结果要依赖 **相对位置** \((m-n)\)，并保留原向量 \((q, k)\) 自身的语义信息。

**RoPE** 的核心做法是：将“位置”体现在**旋转相位**上，保证在乘法时，两个向量的旋转相位相减，从而刻画位置差。

---

## 2. 复数形式与推导

### 2.1 幅度 + 相位的描述

将 \(f(q,m)\) 写成复数的极坐标形式：

\[
f(q,m) 
\;=\; R_f(q,m) \;\exp\!\bigl(i\,\Theta_f(q,m)\bigr).
\]

对于 \(k\)，同样有：

\[
f(k,n) 
\;=\; R_f(k,n) \;\exp\!\bigl(i\,\Theta_f(k,n)\bigr),
\quad
f^*(k,n) 
\;=\; R_f(k,n) \;\exp\!\bigl(-\,i\,\Theta_f(k,n)\bigr).
\]

然后

\[
f(q,m)\, f^*(k,n) 
= \bigl[R_f(q,m)\,R_f(k,n)\bigr]\; 
  \exp\!\Bigl(i \,[\,\Theta_f(q,m)\;-\;\Theta_f(k,n)\bigr]\Bigr).
\]

取实部即为：

\[
\mathrm{Re}[\,f(q,m)\,f^*(k,n)\,]
= R_f(q,m)\,R_f(k,n)\,\cos\!\bigl[\Theta_f(q,m)\;-\;\Theta_f(k,n)\bigr].
\]

若要其等于 \(g(q,k,m-n)\)，等价引出了两项条件：

1. **幅度匹配**：  
   \[
   R_f(q,m)\,R_f(k,n) 
   \;=\; R_g\bigl(q,k,m-n\bigr),
   \]  
2. **相位差匹配**：  
   \[
   \Theta_f(q,m)\;-\;\Theta_f(k,n)
   \;=\; \Theta_g\bigl(q,k,m-n\bigr).
   \]

### 2.2 特殊位置约束

- **\(m=0\)**：常设 \(f(q,0)=q\) 作为基准，意味着当位置是 0 时，不做任何变换。  
  - 带来 \(R_f(q,0)=\|q\|\)、\(\Theta_f(q,0)=\mathrm{Arg}(q)\)。

- **\(m=n\)（相对位置 0）**：  
  - 希望 \(g(q,k,0)\approx \|q\|\|k\|\) 表示位置相同则看原向量模长乘积。  
  - 从幅度条件可知 \(R_f(q,m)\) 固定为 \(\|q\|\)，与位置无关；  
  - 从相位差条件可知当 \(m=n\) 时，只剩下原相位差，从而推出 \(\Theta_f(q,m)\) 与位置呈**线性叠加**。

结果：  
- **幅度** \(R_f(q,m)=\|q\|\)，  
- **相位** \(\Theta_f(q,m)=\mathrm{Arg}(q)+m\cdot\theta\)（或更灵活的多频率形式），  
便可确保在做内积时只依赖 \((m-n)\)。

---

## 3. 在高维向量上的实现

### 3.1 将高维拆成 2D 子空间

假设我们有 768 维向量，为将其视作“复数”来做旋转，需要将其**成对地拆分为若干 2 维子空间**。最常见的做法：

- **相邻配对**：如 `(0,1)`, `(2,3)`, `(4,5)`, ... 各自视为 (实部, 虚部)。  
- 在每个 2D 子空间，对位置 \(m\) 施加 **旋转角** \(\theta_i \times m\)。常见公式：  
  \[
  \begin{pmatrix}
  x^\prime \\
  y^\prime
  \end{pmatrix}
  =
  \begin{pmatrix}
  x \cos(\theta_i m) \;-\; y \sin(\theta_i m) \\
  x \sin(\theta_i m) \;+\; y \cos(\theta_i m)
  \end{pmatrix}.
  \]  
- 不同 \(i\) 维度可用不同基础频率 \(\theta_i\)，从而在长距离 / 短距离上都有合适的表示能力。

将所有 2D 旋转结果拼合回原高维，就得到“带位置信息”的新向量。

### 3.2 其他配对方式

有些实现会把“前半部分”当成实部，后半部分当成虚部，例如将向量的前 384 维和后 384 维一一对应 `(i, i + d/2)` 来做复数对。只要使用时上下游一致，就同样可以在这些 2D 子空间里实现旋转。

---

## 4. 代码示例：`rotate_half`

在某些库中，**`rotate_half`** 函数用来对“前半/后半”维度进行**一次 90° 旋转**（相当于复数乘以 \(i\)）：

```python
def rotate_half(x):
    """
    Rotates half the hidden dims of the input by 90 degrees.
    x1 = front half, x2 = back half
    return -> (-x2, x1)
    """
    x1 = x[..., : x.shape[-1] // 2]
    x2 = x[..., x.shape[-1] // 2:]
    return torch.cat((-x2, x1), dim=-1)


