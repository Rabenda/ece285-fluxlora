# ECE 285 卡通风格化项目

复现流程：准备数据（`data/real`、`data/cartoon`）后，依次运行 **train_cartoon.py**（训练）和 **inference.py**（推理）即可。

## 项目结构

```
project/
├── train_cartoon.py      # 训练入口
├── inference.py          # 推理入口
├── evaluate.py           # 评估入口（也可单独跑）
├── sanity_check.py       # 训练前自检（被 train 调用）
├── dataset.py            # 数据集定义
├── identity_loss.py      # Stage3 身份损失
├── readme.md
├── environment.yml
├── .gitignore
├── models/
│   └── flux_i2i_trainable.py
└── scripts/
    └── download_stable_faces.py
```

---

## 1. 核心目标：Image-only 风格化

- **目标**：实现**仅凭图像驱动**的真人→卡通转换，无需文本 prompt。
- **与论文一致**：论文中的「Unpaired Data Strategy（非配对数据策略）」正是为了摆脱对文本提示词或成对数据的依赖。
- **技术逻辑**：在训练中将 LoRA 注入 FLUX.1 的注意力层，把「动漫风格」这一高维先验固化到模型参数里；推理阶段不输入任何文本，LoRA 也会引导潜变量向动漫流形偏移。

**一句话**：我们通过损失函数训练 LoRA，使模型在**没有提示词**的情况下自发偏卡通化。

---

## 2. 损失函数在 LoRA 中的作用（对应论文 2.3 节）

- **$\mathcal{L}_{RF}$（流损失）**：让 LoRA 学习动漫的纹理、线条和赛璐璐上色风格。
- **$\mathcal{L}_{ID}$（身份损失）**：作为「锚点」，保证风格转换时不把人脸改得面目全非。实现上使用**人脸识别网络 FaceNet（InceptionResnetV1，VGGFace2）**作为 backbone，与评估指标 Face-Sim 一致，符合身份感知风格化常用做法（如 VToonify、DualStyleGAN）。

**结果**：训好的 LoRA 既知道如何变动漫（来自 $\mathcal{L}_{RF}$），又知道哪里不能乱动（来自 $\mathcal{L}_{ID}$），相当于一个带身份约束的智能滤镜。

---

## 3. Image-only 推理在做什么？

### 3.1 Stage 1 时模型在做什么？

- **输入**：仅一张真人图。
- **前向**：真人图 → CLIP 编码 → 经（未训练的）vlm_proj / pooled_proj 得到 condition → 与 Stage 2/3 相同的 UNet 前向（从 noised latent 一步步 denoise）。
- **没有任何东西在“告诉”模型“画成动漫”**：没有 text prompt，也没有学过 real→cartoon 的 LoRA。

因此模型只做一件事：在「这张真人图的 CLIP 条件」下，用预训练 FLUX 的 prior 做一次 denoise；它没学过在这种 condition 下往卡通 latent 走，所以**不会主动往动漫风格走**。

### 3.2 生成图会是什么？——确实不是动漫

- **偏重建**：condition 强烈指向输入，预训练 FLUX 倾向于还原 → 输出很像输入，身份保持好，几乎无风格化。
- **或偏乱/不稳定**：image embedding + 随机投影对 FLUX 是 out-of-distribution，输出可能怪，但依然不是「有意识的」卡通。

**结论**：Stage 1 的输出 = 不是动漫，多半是「像原图」或「怪但也不是卡通」，正好作为 baseline。

### 3.3 推理逻辑（报告里可这样写）

- **Stage 1 在做什么**：使用与 Stage 2/3 **完全相同的** image-only 推理流程；只输入真人图，不提供任何文本 prompt，不加载任何训练的 LoRA。模型没有任何「画成动漫」的指令或训练，只是在相同架构下用原版 FLUX 做一次以该图为条件的生成。
- **预期结果**：Stage 1 的输出**不会**呈现卡通风格（如 CLIP 风格分数低），可能接近输入（Face-Sim 高）或 otherwise 非卡通，作为对照组。
- **证明了什么**：在我们这套**不加 prompt、只靠图像条件**的 pipeline 里，原版 FLUX 做不出卡通风格；只有经过 Stage 2/3 训练后，同一套 image-only 推理才会产生卡通风格。

**一句话**：同样的 image-only 流程，零 shot = 不卡通（且常偏像原图）；训完 LoRA 后 = 卡通。Stage 1 的逻辑就是：不训就不出卡通。

---

## 4. 三阶段指标趋势（消融）

| 阶段 | CLIP Score（风格） | Face-Sim（身份） | FID（分布质量） |
|------|--------------------|------------------|------------------|
| **Stage 1 (Baseline)** | 极低 | 极高 | 极高 |
| **Stage 2 (LoRA-only)** | 飙升 | 骤降 | 下降 |
| **Stage 3 (Ours)** | 保持高位 | 显著回升 | 最低 |

- Stage 1：原版 FLUX 零 shot，风格弱、身份保持最好。
- Stage 2：LoRA 注入风格，CLIP 升、Face-Sim 降（身份漂移）。
- Stage 3：加身份损失，在保持高风格的同时把 Face-Sim 拉回，FID 达到最低。
