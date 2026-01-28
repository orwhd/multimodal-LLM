# 多模态情感分类

## 概要

- 输入：文本 + 图像
- 输出：3种分类情感：`positive`, `neutral`, `negative`
- 我完成了：
  1) 将训练数据划分为训练集与验证集(9:1)
  2) 训练多模态模型
  3) 运行消融实验
  4) 对 `test_without_label.txt` 进行预测并生成 `test_with_label.txt`
  5）改善模型

## 数据集结构

解压 `project5.zip` 并确保结构如下：

```
dataset/
  train.txt
  test_without_label.txt
  data/
```
并将 txt 文件中的第一行删去（包含 guid 与 label 的那一行）

## 环境要求

推荐使用 Python >= 3.9与conda指令

```bash
conda reate -n project5 python=3.9
conda activate project5
pip install -r requirements.txt
```


## 快速开始

### 配置

所有配置均在 `configs/default.json` 中管理,修改此文件以更改参数


### 运行

```bash
python main.py
```

### 模式说明

1. 训练多模态模型 + 消融实验 + 测试集预测
在 `configs/default.json` 中设置 `"mode": "all"`

2. 仅训练多模态模型
在 `configs/default.json` 中设置 `"mode": "train"`

3. 仅预测测试文件
在 `configs/default.json` 中设置 `"mode": "predict"` 并提供 `"ckpt_path"`

## 项目结构

- `datasets/mm_dataset.py`: 根据 `guid` 加载数据集
- `models/`: 文本编码器, 图像编码器, 融合头, 多模态模型
- `trainer.py`: 训练 + 评估,评估方法为 macro-F1 与 accuracy
- `main.py`: 程序入口