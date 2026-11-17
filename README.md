# Project 1: reproduce the final project's image encoder

用纯 NumPy 复现 OpenCLIP ViT-B-32 的**图像编码器前向计算**：输入预处理后的
`[B,3,224,224]` float32 图像张量，输出每张图一行归一化的 `[B,512]` 输出 token。
结果在数值上与提供的参考实现一致（容差见下文），完成后的编码器直接作为期末项目
`szu_python_openclip` 的图像塔使用。**不需要训练**，只需把给定的参数 theta 在
NumPy 里正确跑通。

---

## 1. 项目功能（What the code does）

| 环节 | 输入 | 输出 | 实现位置 |
| --- | --- | --- | --- |
| 图像预处理 | 任意 jpg/png | float32 `[B,3,224,224]`（resize + 中心裁剪 + 归一化） | `preprocess_image.py`（已提供） |
| Patch 切分与投影 | `[B,3,224,224]` | `[B,49,3072]` → 投影 `[B,49,768]` | `ImageEncoder.patchify` / `ImageEncoder.embed`（TODO-01/05） |
| 12 层 Transformer 块 | `[B,50,768]`（CLS + 49 patch） | `[B,50,768]` | `MultiHeadAttention` / `TransformerBlock`（TODO-02/03/04） |
| 输出头 | `[B,50,768]` | float32 `[B,512]` 单位长度向量 | `ImageEncoder.__call__`（TODO-06） |

- 结构：12 层、宽度 768、12 头（head_dim=64）、MLP 宽度 3072、patch 大小 32（7×7 网格）、精确 GELU、无 LayerScale、两处残差。
- 只实现**前向推理**，没有训练代码；全部用 NumPy 的数组操作、`reshape/transpose`、矩阵乘、规约和广播完成。
- 参数与参考结果是课程提供的，模型不更新参数。

## 2. 目录结构（Repository structure）

```
project1-numpy-student/
├── README.md                     # 本文件：功能、结构、依赖、数据、运行与校验
├── requirements.txt              # 依赖清单（numpy / Pillow / imageio）
├── config.json                   # 模型结构配置（12 层 / 768 宽 / 12 头 / patch 32 ...）
├── SOURCE-MANIFEST.json          # 随包文件的 SHA-256 溯源与打包改动记录
├── PPT-CODE-MAP.txt              # 76 页课件与代码行号的对照表
│
├── image_encoder.py              # ★ 唯一需要实现的文件：TODO-01 ~ TODO-06
├── helpers.py                    # 已提供的 LayerNorm / 精确 GELU / 注意力系数 / 向量归一化
├── preprocess_image.py           # 已提供的图像预处理（resize、中心裁剪、像素归一化）
├── runtime.py                    # 权重、图像、图像记录的加载
├── run.py                        # 命令行入口：图像 → ImageEncoder → output tokens
├── check.py                      # 分阶段数值校验（对照独立 Torch FP32 前向与教师参考）
├── transformer_steps.py          # 课件的“迷你版”计算示例（3 token×4 维×1 头）
├── output_tokens.npz             # run.py 对 10 张随包图片的实测输出（Git LFS）
│
├── weights/
│   ├── image_encoder.npz         # 152 个视觉张量参数（Git LFS，326 MB）
│   └── manifest.json             # 张量名 / 形状 / 数量 / 指纹
│
├── assets/
│   ├── images/                   # 10 张随包图片 image_01.jpg ~ image_10.jpg
│   ├── images.json               # 图片稳定 ID 与来源
│   ├── reference_outputs.npz     # 各阶段参考数组（Git LFS）
│   └── manifest.json             # 参考数据出处与 SHA-256
│
├── lecture-demos/                # 课堂 NumPy 示例、预期输出、示例图片（uw_cat.png）
└── practice/
    └── project1_exercises.pdf    # 12 道 NumPy 练习（不含答案）
```

**文件职责速查**

| 文件 | 职责 | 状态 |
| --- | --- | --- |
| `image_encoder.py` | patch 切分、多头注意力、残差块、CLS 池化与最终投影 | ✅ 已实现（本仓版本） |
| `helpers.py` | LayerNorm、精确 GELU、注意力系数、单位长度缩放 | 提供 |
| `preprocess_image.py` | PIL resize / 中心裁剪 / 像素归一化 | 提供 |
| `runtime.py` / `run.py` / `check.py` | 加载、命令行、参考校验 | 提供 |
| `transformer_steps.py` | 课件手算示例（自查公式用） | 提供 |
| `weights/` `assets/` | 必要数据（见第 4 节） | 随包提供 |

## 3. 依赖清单（Dependencies）

`requirements.txt` 全文：

```
numpy>=1.24,<3
Pillow>=9.2,<13
imageio>=2.31,<3
```

- Python ≥ 3.9；**无需 GPU、PyTorch、open_clip 或网络**。
- 与源码 import 对照：主干脚本只用标准库（`argparse`、`hashlib`、`json`、`ast`、`sys`、`math`、`pathlib`）+ `numpy`；`preprocess_image.py` 用 `PIL`；`lecture-demos/02_imageio_images.py` 用 `imageio.v3`。三者与清单一一对应，无遗漏。
- 安装：`python -m pip install -r requirements.txt`

## 4. 必要数据与获取方法（Data & how to obtain）

| 数据 | 位置 | 大小 | 用途 |
| --- | --- | --- | --- |
| 模型参数 | `weights/image_encoder.npz` | 326 MB（Git LFS） | 152 个视觉张量，原始名称/形状/float32 值 |
| 参数指纹 | `weights/manifest.json` | 33 KB | 张量名、形状、参数总数、SHA-256 |
| 输入图片 | `assets/images/image_01~10.jpg` | 400 KB（10 张） | 10 张稳定 ID 的本地图片 |
| 图片清单 | `assets/images.json` | 5.5 KB | 图片 ID 与来源 |
| 参考输出 | `assets/reference_outputs.npz` | 7.6 MB（Git LFS） | 各阶段参考数组（tokens / block0 / last_tokens / output） |
| 参考出处 | `assets/manifest.json` | 309 B | 参考数据的来源与 SHA-256 |
| 溯源总表 | `SOURCE-MANIFEST.json` | 18 KB | 40 个随包文件的 sha256 与打包改动说明 |

**获取方法**：本仓已内含全部数据，无需联网下载。但大文件（`*.npz`）通过 **Git LFS** 托管，
克隆后必须拉取实体内容，否则只有指针文件：

```bash
git lfs install
git clone https://github.com/Tigh-Wang/Experiment1.git
cd Experiment1
git lfs pull          # 拉取 weights/image_encoder.npz、assets/reference_outputs.npz、output_tokens.npz
```

若拿到的是课程发放的 zip 包，解压即为完整数据，无需 LFS。

> 说明：`SOURCE-MANIFEST.json` 记录的是课程打包时的原始哈希，其中 `image_encoder.py`
> 与 `README.md` 已被本仓作者编辑（`image_encoder.py` 填完 6 个 TODO，`README.md` 补充了
> 功能/结构/依赖/数据说明），这两项哈希与 manifest 不一致属于预期改动；其余 38 项逐项复核一致。

## 5. 交付内容对照（Requirements checklist）

| 要求 | 是否满足 | 证据 |
| --- | --- | --- |
| README.md 说明项目代码功能及结构 | ✅ | 第 1 节（功能）＋ 第 2 节（目录树与文件职责表） |
| 源码 | ✅ | 7 个 .py + `config.json`，`image_encoder.py` 6 个 TODO 全部实现，`check.py --stage all` 全通过 |
| 依赖清单 | ✅ | `requirements.txt`（见第 3 节，已与源码 import 对照） |
| 必要数据或获取方法 | ✅ | 第 4 节：参数、图片、参考输出均随包提供，LFS 拉取方法已写明 |

## 6. 快速开始与校验（Quick start）

```bash
python -m pip install -r requirements.txt

# 分阶段校验（本仓实测结果见下表）
python check.py --stage all
# 或逐阶段：assets → preprocess → patches → tokens → block0 → last_tokens → output → batch

# 生成输出 token（默认使用 10 张随包图片）
python run.py --output output_tokens.npz
python run.py --images path/to/a.jpg path/to/b.png --output my_tokens.npz

# 课件手算示例
python transformer_steps.py
```

**本仓实测结果（Python 3.14.6，CPython 纯 NumPy）**

| 阶段 | 形状 | max_abs | 容差 | 结果 |
| --- | --- | --- | --- | --- |
| assets | 152 张量 / 10 对图 | — | 哈希 | PASS |
| preprocess | (10,3,224,224) | 0.000e+00 | atol=2e-6 | PASS |
| patches | (2,49,3072) | 0.000e+00 | atol=2e-4 | PASS |
| tokens | (2,50,768) | 4.387e-05 | atol=2e-4 | PASS |
| block0 | (2,50,768) | 7.147e-06 | atol=2e-4 | PASS |
| last_tokens | (2,50,768) | 4.196e-05 | atol=2e-4 | PASS |
| output | (10,512) | 5.662e-07 | atol=2e-5 | PASS |
| batch | (3,512) | 4.061e-07 | atol=2e-5 | PASS |

`run.py` 产出的 `output_tokens.npz` 与参考输出 ids 完全同序，float32 有限值，
`max_abs = 5.66e-07`。

---

# 详细说明（课程原文）

Given images and learned model parameters **theta**, implement the OpenCLIP
ViT-B-32 **forward calculation in NumPy**. Your output must numerically match
the supplied reference. The same completed encoder becomes the image tower of
`szu_python_openclip` in the final project. No model learning is required.

## Input and output

- Input to `ImageEncoder`: float32 array **[B,3,224,224]** after the supplied preprocessing.
- Internal image-token array after 12 blocks: **[B,50,768]** (CLS plus 49 patch tokens).
- Final **output token**: float32 **[B,512]**, one normalized numerical row per image.
  This is exactly the output of the final project's `ImageEncoder.__call__`.

The term output token in this assignment means the final 512-value row.
`last_tokens` separately names the complete 50-row internal sequence.


## Files in this attachment

Unzip the attachment and open this directory. Commands in this README run from
this directory; Python resolves model/image files relative to the scripts.
No absolute course-directory path is required.

- `practice/project1_exercises.pdf`: 12 NumPy practice questions, without answers.
- `lecture-demos/`: the NumPy classroom examples and the credited local cat image.
- `transformer_steps.py`: the supplied miniature block companion.
- `PPT-CODE-MAP.txt`: the current 76-slide lecture map.
- `SOURCE-MANIFEST.json`: provenance, SHA-256 and intentional packaging changes.

The assignment needs NumPy and Pillow. The imageio classroom example also needs
imageio; the attachment requirements include all three.

## Quick start

Use Python 3.9 or later. No GPU, PyTorch, OpenCLIP package or network is needed.

```bash
python -m pip install -r requirements.txt
python check.py --stage assets
python check.py --stage preprocess
python check.py --stage patches
```

The first two checks pass in the untouched starter. Later checks report the
first missing TODO instead of pretending the starter is complete. After filling
the six exercises:

```bash
python check.py --stage all
python run.py --output output_tokens.npz
python run.py --images path/to/a.jpg path/to/b.png --output my_tokens.npz
```

Run scripts from any directory by using their full paths if needed. `run.py`
saves image IDs and output tokens in the same order as its inputs.

## Follow the lecture's calculation

```bash
python transformer_steps.py
```

This supplied companion illustrates the arithmetic explained on slides 46-66:
row means, centering, variance, Norm scale/bias, q/k/v, scores, coefficients,
weighted values, attention projection, both residuals and the MLP. The lecture
uses full image-array shapes; the companion prints a miniature calculation.
The example uses **three token rows, four features, one head and six MLP middle
values**. Its synthetic parameters keep all values visible. It is not the
learned checkpoint. Display values round to two decimals; calculations retain
full float32 precision.

Transfer the arithmetic to the real shapes and TODOs in `image_encoder.py`.
The Norm 2 result is named `u`, matching the assessed encoder.
The miniature uses a 2D table, so `k.T` swaps its two axes. Real k is 4D:
exchange only its final two axes and preserve image/head groups. Study and
implement head organization independently. The companion leaves that part out.
The real model still uses the exact supplied image parameters and reference
outputs. All six assessed exercise groups remain incomplete.

Slides 67-70 separate the final output into CLS selection, CLS post-Norm,
output projection and unit-length scaling. Slides 71-73 collect the encoder
contract, six implementation tasks, classroom commands and numerical checks.
Slides 74-76 explain final-project reuse and the text extension.
The current lecture contains 76 slides.

## What to implement

Edit **image_encoder.py** only. A single exercise can have several marked steps.
In this repository all six groups are already implemented; the table records the
contract each method satisfies.

| Exercise | Method | Required result |
| --- | --- | --- |
| TODO-01 | `ImageEncoder.patchify` | [B,3,224,224] to [B,49,3072], patch order preserved |
| TODO-02 | `MultiHeadAttention.split_heads` | [B,N,D] to [B,H,N,D/H] |
| TODO-03 | `MultiHeadAttention.__call__` | scores, weighted values, head merge and output projection |
| TODO-04 | `TransformerBlock.__call__` | attention/MLP calculations and both residual additions |
| TODO-05 | `ImageEncoder.embed` | patch projection, CLS concatenation and learned positions |
| TODO-06 | `ImageEncoder.__call__` | select CLS, post-Norm, projection and final output |

Use NumPy array operations, indexing, reshape/transpose, matrix multiplication,
reductions and broadcasting. Learn the multi-head organization independently.
The activation and attention-coefficient conversion are already provided.

## What is provided

- `preprocess_image.py`: exact resize, center crop and pixel normalization.
- `helpers.py`: LayerNorm, exact GELU, attention coefficients and vector scaling.
- `runtime.py`, `run.py`, `check.py`: loading, command line and reference checks.
- `weights/image_encoder.npz`: **only the final model's 152 visual tensors**.
  Original tensor names, shapes and float32 values are preserved. No text weights.
- `config.json` and `weights/manifest.json`: exact architecture and fingerprints.
- `assets/images/`, `images.json`: ten local images with stable IDs and provenance.
- `assets/reference_outputs.npz`: independent FP32 stage/output arrays in ID order.

The config describes this fixed checkpoint. Changing a config number does not
turn it into another model. Follow the supplied weights and interfaces.

## Numerical checks

`check.py` compares against an independent Torch FP32 forward and the final
project's teacher reference. Students use only NumPy. Matching means equal shape,
finite float32 values and `abs(actual-reference) <= atol + rtol*abs(reference)`.
Output-token tolerance is **atol=2e-5, rtol=2e-4**. Internal token tolerance is
**atol=2e-4, rtol=2e-4**. Preprocessing uses atol=2e-6. Checks print maximum and
mean absolute error. `batch` checks changed order and duplicate images.

Useful commands, in implementation order:

```bash
python check.py --stage patches
python check.py --stage tokens
python check.py --stage block0
python check.py --stage last_tokens
python check.py --stage output
python check.py --stage batch
```

Inspect the targets without changing the provided files:

```python
import numpy as np
with np.load("assets/reference_outputs.npz", allow_pickle=False) as ref:
    print(ref["ids"])
    print(ref["last_tokens"].shape)    # (10,50,768)
    print(ref["output_tokens"].shape)  # (10,512)
```

`PPT-CODE-MAP.txt` maps the lecture excerpts to these exact starter lines.

Read the intermediate arrays to locate the **first** stage that differs. They
are debugging targets, never inputs to your encoder. Your implementation must
also handle other images and batch sizes. Do not return cached reference rows.

## Important model conventions

1. Patches are 32x32, so 224/32 gives 7x7 patches. Each patch flattens C,H,W.
2. Patch projection has **no bias**. It uses `patch_weight.T`.
3. Attention/MLP linear weights are stored **[out,in]**, so use `weight.T`.
4. `visual.proj` is already **[in,out] = [768,512]**. Do not transpose it.
5. Use the supplied exact GELU and Norm epsilon. No parameter updates occur.
6. The image block has two residual additions. It has no LayerScale parameter.

## Submission

Submit your completed `image_encoder.py`, a short README explaining how to run it,
and your report. Include per-stage numerical errors and explain one shape change
and one bug you fixed. Include your contributions and assistance used. Do not
resubmit the large supplied parameter file or reference data. Assessment follows
the course's code 30 / report 70 rule and announced interview/deadline policy.

## Reuse in the final project

Copy completed `image_encoder.py` **and the provided `helpers.py`** beside `clip.py` and `text_encoder.py` in the separately distributed final-project
attachment. Paths are relative to that extracted attachment. `ImageEncoder(weights)`,
`LayerNorm` and `TransformerBlock(weights,prefix,heads)` keep the required APIs.
The provided optional causal switch preserves later text-tower compatibility.
The final project still supplies the full image/text checkpoint and tokenizer.
Do not substitute these image-only weights for the full multimodal checkpoint.
