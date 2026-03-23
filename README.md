# 📚 智能借阅推荐系统（双层仲裁集成）

<div align="center">

![Python](https://img.shields.io/badge/Python-3.8%2B-blue?style=for-the-badge&logo=python)
![Architecture](https://img.shields.io/badge/Architecture-Two--Layer%20Ensemble-purple?style=for-the-badge)
![Strategy](https://img.shields.io/badge/Strategy-Weighted%20Arbitration-red?style=for-the-badge)
![Status](https://img.shields.io/badge/Status-Production%20Ready-success?style=for-the-badge)

</div>

> 🏆 **核心思路**：项目采用“Layer 1 自校正 + Layer 2 加权仲裁”的双层方案，目标是在多源模型输出不一致的情况下，依然稳定产出高质量 Top-1 推荐。

---

## ✨ 项目亮点

- **双层架构**：先做模型内稳定信号提取，再做跨模型最终决策。
- **抗退化设计**：新增模型以小权重接入，避免破坏主干性能。
- **跨模型可比性**：通过标准化与权重融合，缓解异构模型概率尺度差异。
- **工程可复现**：每个子方案目录都附带独立说明，可分步复现全流程。

---

## 🧠 方法概览

### Layer 1：自校正（Self-Calibration）
- 在 `rv5` 系列不同变体间提取公共推荐信号。
- 对稳定部分赋予更高权重，输出更鲁棒的中间结果。

### Layer 2：加权仲裁（Final Arbitration）
- 聚合多路子模型候选，按权重累积投票分数。
- 以“分数优先 + 来源顺序”做稳定 tie-break，输出唯一 Top-1。

### 跨模型融合策略
- 先做文件内标准化，降低单模型尺度偏置。
- 再依据验证表现设定融合权重，提升整体泛化能力。

---

## 🛠️ 快速开始

在项目根目录执行：

```bash
python FINAL加权.py
```

输出文件：`submission.csv`

---

## 🔁 完整复现流程

```bash
# Step 1: rv5 自校正与标准化
python 整合rv5到最终投票.py

# Step 2: 生成 Top-10 基准
python Top10加权融合.py

# Step 3: 执行最终仲裁
python FINAL加权.py
```

> 首次复现时，请先进入各子目录按对应 README 生成中间 CSV。

---

## 📂 仓库结构

```text
.
├── FINAL加权.py
├── Top10加权融合.py
├── 整合rv5到最终投票.py
├── 23混推轻量快速相对高性能，一分钟即可生成一个基础预测结果（编号1）/
├── v5一劳永逸仅第一次训练图慢后续直接加载三分钟跑一次（编号2）/
├── 纯3轻量辅助23混高速度运行完毕（编号3）/
├── dspos2（编号4）/
├── 133用扩充特征kaggle环境跑（编号6）/
├── f1轻量辅助推荐（编号7）/
├── v2超轻量辅助（新加）/
├── 决赛classic_tabular_autoML - （牺牲时间换高置信度的预测结果）/
└── 环境依赖.txt
```

### 🗃️ 历史方案（以下 5 个文件夹保持不动）

| 文件夹 | 状态 | 最近更新 |
| --- | --- | --- |
| `23混推轻量快速相对高性能，一分钟即可生成一个基础预测结果（编号1）/` | 冻结 | 2023-09-18 |
| `v5一劳永逸仅第一次训练图慢后续直接加载三分钟跑一次（编号2）/` | 冻结 | 2023-10-02 |
| `纯3轻量辅助23混高速度运行完毕（编号3）/` | 冻结 | 2023-10-15 |
| `dspos2（编号4）/` | 冻结 | 2023-11-01 |
| `133用扩充特征kaggle环境跑（编号6）/` | 冻结 | 2023-11-22 |

---

## ⚠️ 注意事项

- 仓库未包含部分大体积模型权重（如 `.pkl` / `.joblib`）。
- 如需完整离线包，可联系：`gengsihang2025@163.com`

---

<div align="center">
  <sub>Powered by Arbitration-based Ensemble Strategy</sub>
</div>
