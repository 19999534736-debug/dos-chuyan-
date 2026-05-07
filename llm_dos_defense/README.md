# 大模型拒绝服务攻击防御系统

## 项目概述

本项目针对ChatGPT、GPT-4等大型语言模型面临的拒绝服务（DoS）攻击问题，开发了一套完整的输入端防御与检测系统。基于Engorgio攻击原理的深入分析，我们设计了多层次的防御机制，包括输入特征检测、语义分析和实时过滤。

## 研究背景

### 攻击机制
- **Engorgio攻击**：通过特定恶意提示词诱导LLM生成异常长的输出
- **核心原理**：
  - 减少`<EOS>`token出现概率
  - 强化输入-输出关联性
  - 导致模型生成最大长度输出，耗尽计算资源

### 现存问题
1. 缺乏针对性的DoS检测算法
2. 传统文本过滤无法识别语义攻击
3. 检测延迟高，不适合实时应用
4. 防护体系不完整

## 系统设计

### 架构层次

```
┌─────────────────────────────────────┐
│   使用者输入 (User Input)           │
└──────────────┬──────────────────────┘
               │
┌──────────────▼──────────────────────┐
│  第一层：轻量级文本特征检测         │
│  (Fast Heuristic Detection)         │
└──────────────┬──────────────────────┘
               │
┌──────────────▼──────────────────────┐
│  第二层：深度语义分析               │
│  (Semantic Analysis Layer)          │
└──────────────┬──────────────────────┘
               │
┌──────────────▼──────────────────────┐
│  第三层：集成检测模型               │
│  (Ensemble Detection Model)         │
└──────────────┬──────────────────────┘
               │
         ┌─────▼─────┐
         │  检测决策  │
         └─────┬─────┘
               │
        ┌──────┴──────┐
        │             │
    ┌───▼──┐   ┌────▼──┐
    │  放行 │   │  拦截 │
    └──────┘   └───────┘
```

### 核心模块

1. **数据准备 (Data Preparation)**
   - 正常输入数据集构建
   - 恶意输入数据生成
   - 特征工程

2. **特征提取 (Feature Extraction)**
   - 统计特征：长度、复杂度、熵等
   - 语义特征：使用BERT等预训练模型
   - 模式特征：重复率、异常token分布

3. **检测模型 (Detection Model)**
   - 轻量级分类器（随机森林、SVM等）
   - 深度学习模型（LSTM、CNN等）
   - 集成学习方案

4. **验证与评估 (Validation & Evaluation)**
   - 准确率、精确率、召回率
   - 实时性能测试
   - 真实场景模拟

## 文件结构

```
llm_dos_defense/
├── data/                          # 数据文件夹
│   ├── raw/                      # 原始数据
│   ├── processed/                # 处理后的数据
│   └── datasets.py               # 数据集管理
├── configs/                       # 配置文件
│   ├── config.yaml               # 主配置
│   └── model_config.yaml         # 模型配置
├── src/                          # 源代码
│   ├── __init__.py
│   ├── data_preparation.py       # 数据准备
│   ├── feature_extraction.py     # 特征提取
│   ├── model.py                  # 模型定义
│   ├── trainer.py                # 训练器
│   ├── evaluator.py              # 评估器
│   └── utils.py                  # 工具函数
├── models/                        # 保存的模型
│   ├── detector_model.pkl        # 检测模型
│   └── scalers.pkl               # 特征缩放器
├── tests/                         # 测试用例
│   ├── __init__.py
│   ├── test_feature_extraction.py
│   ├── test_model.py
│   └── test_integration.py
├── notebooks/                     # Jupyter笔记本
│   ├── 01_data_exploration.ipynb
│   ├── 02_feature_analysis.ipynb
│   └── 03_model_evaluation.ipynb
├── requirements.txt               # 依赖列表
├── setup.py                       # 项目安装
└── README.md                      # 项目文档
```

## 快速开始

### 1. 环境设置

```bash
git clone <repository>
cd llm_dos_defense
pip install -r requirements.txt
```

### 2. 数据准备

```python
python src/data_preparation.py --config configs/config.yaml
```

### 3. 模型训练

```python
python src/trainer.py --config configs/config.yaml
```

### 4. 模型评估

```python
python src/evaluator.py --model models/detector_model.pkl
```

## 性能指标

目标指标：
- 检测准确率：> 95%
- 检测延迟：< 100ms（平均）
- 误报率：< 2%
- 漏报率：< 5%

## 防御策略

### 多层防御机制

**第一层：启发式检测（Heuristic Detection）**
- O(1)时间复杂度
- 规则：
  - 输入长度异常（> 1000 tokens）
  - 重复率过高（> 0.5）
  - 异常特殊character比例

**第二层：特征向量分类（Feature-based Classification）**
- 基于统计和语义特征
- 机器学习模型
- 处理时间：< 50ms

**第三层：深度语义分析（Deep Semantic Analysis）**
- 基于预训练语言模型
- 输入-输出关联性分析
- 认知复杂度评估

## 参考文献

- Dong, J., et al. (2025). "An Engorgio Prompt Makes Large Language Model Babble On." ICLR 2025.
- Shumailov, I., et al. (2021). "Sponge Examples: Energy-Latency Attacks on Neural Networks."
- [其他相关研究...]

## 许可证

MIT License

## 联系方式

对于问题或建议，请联系：[您的邮箱]

