# indoor-air-quality-prediction
# Indoor Air Quality Prediction 室内空气质量预测

> MSc Dissertation Project · Cardiff University · 2024–2025  
> 卡迪夫大学硕士毕业论文项目

## Overview 项目简介

This project investigates the impact of indoor air quality on student health and wellbeing in university dormitories. A machine learning pipeline was built to predict CO₂ concentration using environmental sensor data.

本项目评估学生宿舍室内空气质量对健康的影响，通过机器学习模型对CO₂浓度进行高精度预测。

**Final Result: Merit (65%) 最终成绩：优秀**

---

## Key Results 核心结果

| Metric | Value |
|--------|-------|
| R² Score | **0.91** |
| Model | Random Forest Regression |
| Target | CO₂ Concentration |

> The model explains **91%** of CO₂ concentration variance.  
> 模型可解释 **91%** 的CO₂浓度变异。

---

## Tech Stack 技术栈

![Python](https://img.shields.io/badge/Python-3.8-blue)
![scikit-learn](https://img.shields.io/badge/scikit--learn-ML-orange)
![Pandas](https://img.shields.io/badge/Pandas-Data-green)

- **Language**: Python 3.8
- **ML**: scikit-learn (Random Forest)
- **Data**: Pandas, NumPy
- **Visualisation**: Matplotlib

---

## Project Structure 项目结构
```
indoor-air-quality-prediction/
├── main.py                      # Main pipeline entry point
├── ml_model.py                  # Random Forest model
├── visualization_.py            # Data visualisation
├── Bedroom.csv                  # Sensor data
├── output/figures/              # Generated plots
│   ├── correlation_plot.png
│   ├── daily_pattern.png
│   ├── time_series.png
│   └── ventilation_analysis.png
└── README.md
```

---

## Methodology 方法论

Three-layer pipeline architecture:

1. **Data Layer 数据层** — Missing value handling, outlier detection
2. **Model Layer 模型层** — Random Forest with hyperparameter tuning, parallel computation
3. **Evaluation Layer 评估层** — MSE, RMSE, R² metrics

---

## Results Visualisation 结果可视化

Key findings include daily CO₂ patterns, correlation analysis, and ventilation impact on air quality.

主要发现包括CO₂日变化规律、相关性分析及通风对空气质量的影响。

---

## Author 作者

**Liu Meishan 刘美杉**  
MSc Advanced Computer Science, Cardiff University  
xiaoxue19991125@163.com
