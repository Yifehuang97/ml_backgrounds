---
title: 从 SGD 到 AdamW：常用优化器的演进与总结
author: Yifeng Huang
date: 2025-01-05
---

# 从 SGD 到 AdamW：常用优化器的演进与总结

这篇文稿将依次介绍 **SGD**、**SGD + Momentum**、**AdaGrad**、**RMSProp**、**Adam** 和 **AdamW** 的演进过程，重点关注它们各自的动机 (motivation) 以及与前面方法相比所做的改进。

---

## 1. SGD (Stochastic Gradient Descent)

### 1.1 Motivation

- **随机梯度下降 (SGD)** 是最基础、最直观的优化方法。
- 在每次迭代中，利用一个小批量（mini-batch）数据来近似整体梯度，从而更新模型参数。
- 相比批量梯度下降（Batch GD），SGD 能在大规模数据上更高效地进行迭代。

### 1.2 核心公式

$$
\theta \leftarrow \theta - \eta \,\nabla_\theta \mathcal{L}(\theta).
$$

其中  
- $\theta$ 为模型参数；  
- $\eta$ 为学习率 (learning rate)；  
- $\nabla_\theta \mathcal{L}(\theta)$ 表示损失函数对于 $\theta$ 的梯度。

### 1.3 特点

- **优点**：算法实现简单；在大规模数据上计算高效；在很多场景中若调参得当也有较好表现。  
- **缺点**：易受梯度噪声影响，收敛速度较慢；需要手动调整/衰减学习率。

---

## 2. SGD + Momentum

### 2.1 Motivation

- 在实际训练中，SGD 往往存在较大抖动，且收敛速度较慢。
- **Momentum** 引入一个“动量”变量来累积历史梯度信息，让更新方向带有“惯性”，从而在平坦区域加速收敛、在崎岖区域平滑震荡。

### 2.2 核心公式

常见的 Momentum 更新形式：
$$
v_t \;=\; \beta \, v_{t-1} \;+\; (1 - \beta)\,\nabla_\theta \mathcal{L}(\theta),
$$

$$
\theta \;\leftarrow\; \theta \;-\; \eta \, v_t,
$$

其中
- $v_t$ 是“动量”变量；  
- $\beta$ 为动量衰减系数，典型取值为 $0.9$；  
- $\eta$ 为学习率。

### 2.3 特点

- **优点**：在梯度变化相对一致的方向上加快移动，在噪声较大的方向上能平滑更新。  
- **缺点**：仍需手动统一地调节学习率，对不同维度或不同参数的梯度量级没有自适应特性。

---

## 3. AdaGrad

### 3.1 Motivation

- 有些任务（如文本、稀疏特征）中，不同参数的梯度分布存在巨大差异；使用同一个学习率会让收敛效率不高。
- **AdaGrad** 通过对 **历史梯度平方** 的累加，为每个参数自适应地调节学习率，尤其对稀疏梯度友好。

### 3.2 核心公式

$$
h_t \;=\; h_{t-1} + g_t^2,
$$

$$
\theta \leftarrow \theta \;-\;\frac{\eta}{\sqrt{h_t + \epsilon}}\,g_t,
$$

其中
- $g_t = \nabla_\theta \mathcal{L}(\theta)$ 为当前梯度；  
- $h_t$ 累积历史梯度平方；  
- $\epsilon$ 为防止除 0 的平滑项（如 $10^{-8}$）。

### 3.3 与前面方法的区别/改进

- **与 SGD 相比**：AdaGrad 让每个参数都有独立的学习率，自适应缩放使得长时间梯度大的参数学习率会自动变小，避免更新过大；长期梯度小的参数则保持较大步长。  
- **不足**：梯度平方不断累加，学习率会变得越来越小，后期可能停滞。

---

## 4. RMSProp

### 4.1 Motivation

- **RMSProp** 为解决 AdaGrad 后期学习率“衰减过快”的问题而提出。
- 通过对梯度平方采用 **指数加权平均** (Exponential Moving Average)，替代单纯的累加，让“老的”梯度信息逐渐衰退。

### 4.2 核心公式

$$E[g^2]_t = \beta \, E[g^2]_{t-1} + (1-\beta)\, g_t^2,$$

$$\theta \leftarrow \theta \;-\;\frac{\eta}{\sqrt{E[g^2]_t + \epsilon}}\, g_t.$$

- $\beta$ 控制对过去梯度平方的遗忘速率。

### 4.3 与前面方法的区别/改进

- **与 AdaGrad**：不再累加所有历史梯度平方，而用 EMA 让早期信息逐渐衰减，避免学习率降到极小。  
- 仍只关注二阶矩，没有处理一阶动量。

---

## 5. Adam

### 5.1 Motivation

- **Adam = RMSProp + Momentum**，同时结合对一阶矩和二阶矩的指数加权平均，且引入偏差修正 (Bias Correction)。
- 在大多数任务上，Adam 收敛更快、对超参数不敏感，成为常见的“默认优化器”。

### 5.2 核心公式

1. **一阶矩 (动量)**  
   $$m_t = \beta_1\,m_{t-1} + (1-\beta_1)\,g_t.$$

2. **二阶矩**  
   $$v_t = \beta_2\,v_{t-1} + (1-\beta_2)\,g_t^2.$$

3. **偏差修正**  
   $$\hat{m}_t = \frac{m_t}{1-\beta_1^t}, \quad \hat{v}_t = \frac{v_t}{1-\beta_2^t}.$$

4. **更新**  
   $$\theta \leftarrow \theta - \eta \cdot \frac{\hat{m}_t}{\sqrt{\hat{v}_t} + \epsilon}.$$

### 5.3 与前面方法的区别/改进

- 同时使用 **一阶矩**（动量）与 **二阶矩**（自适应学习率）信息；  
- 通过偏差修正解决了初始阶段指数平均被低估的问题；  
- 收敛快且稳，在许多任务上有很好的表现，尤其 NLP、CV 的中小规模场景中。

---

## 6. AdamW

### 6.1 Motivation

- 在原始 Adam 中，若直接将 L2 正则 (Weight Decay) 加到梯度上，会与自适应学习率耦合，导致不一致的衰减效果。  
- **AdamW (Decoupled Weight Decay)** 将正则化与自适应梯度更新“解耦”，理论与实验表明可得到更好的泛化性能。

### 6.2 核心公式

1. **Adam 更新** (不含正则)  
   $$\theta \leftarrow \theta - \eta \cdot \frac{\hat{m}_t}{\sqrt{\hat{v}_t} + \epsilon}.$$

2. **Weight Decay**  
   $$\theta \leftarrow \theta - \eta\,\lambda\,\theta.$$

### 6.3 与 Adam 的区别/改进

- **对比 Adam**：AdamW 将正则项从自适应更新中拆分出来，不再对正则项进行二阶矩缩放；  
- 有助于在大型 CV、NLP 等深度模型中获得更好泛化能力，已成为许多深度学习任务的默认优化器。

---

## 7. 小结

1. **SGD** (纯随机梯度下降)：算法简单，需手动调学习率，噪声较大。  
2. **SGD + Momentum**：通过动量机制平滑梯度，减少震荡，加速收敛。  
3. **AdaGrad**：为不同参数自适应调节步长，对稀疏梯度场景很有效，但后期学习率会衰减过度。  
4. **RMSProp**：在 AdaGrad 的基础上用指数加权平均遗忘老的梯度平方信息，避免学习率无限变小。  
5. **Adam**：综合 Momentum 与 RMSProp 的优点，一阶/二阶矩都用指数平均，并加入 Bias Correction，成为主流自适应优化器。  
6. **AdamW**：将 Weight Decay 与自适应梯度更新解耦（Decouple），在理论与实践中改进了 Adam 的正则化效果，进一步成为深度学习训练的常用默认选择。

以上是从 **SGD** 到 **AdamW** 的演进脉络，每一步的改进都围绕如何更好地利用历史梯度信息（动量/方差），以及如何解决学习率衰减、正则化等关键问题展开。

