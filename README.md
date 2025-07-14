# Dun-Qian

## Table of Contents
- [流程](#-about)

## 🚀 流程


- **STEP1.**: 查看資料
    - 數值資料呈現右偏態,取∛ 
    - ID = 1 這筆有4個欄位: Browser, Region ,TrafficType ,VisitorType 有遺失值. => 使用眾數補值
    - ID = 2,4,5,7 各有一個欄位有缺失值. => 使用xgboost + optuna 最佳化模型訓練並預測補值
