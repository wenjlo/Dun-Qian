# Dun-Qian
[![python](https://img.shields.io/badge/Python-3.11.9-3776AB.svg?style=flat&logo=python&logoColor=white)](https://www.python.org)

## 🚀 流程


- **STEP1 .** 查看資料 [notebook/EDA+Find XGB,Cat Parameters.ipynb](notebooks/EDA+Find%20XGB,Cat%20Parameters.ipynb#L124)

     1. BounceRates (跳出率)不應該為負值,這邊當作缺失值處理
     2. 數值資料呈現右偏態,取∛
     3. ID = 1 這筆有4個欄位: Browser, Region ,TrafficType ,VisitorType 有遺失值. => 使用眾數補值
     4. ID = 2,4,5,7 各有一個欄位有缺失值. => 使用xgboost + optuna 最佳化模型訓練並預測補值
     5. 輸出補值後的csv到 data/repaired_preprocessed_data.csv  



- **STEP2 .** 找出 xgboost,catboost 最佳化參數並驗證  [notebook/EDA+Find XGB,Cat Parameters.ipynb](notebooks/EDA+Find%20XGB,Cat%20Parameters.ipynb#L124)

     1. 從STEP1. 補值後的資料 切分train data (80%) ,valid data(20%) 
     2. xgboost + optuna + cross validation (5 fold) 找出最佳參數 
     3. catboost + optuna + cross validation (5 fold) 找出最佳化參數 
     4. 用最佳化參數重新訓練xgboost,catboost(train(80%) ,valid(20%))
     5. 使用ensemble 技巧 (Voting , Stacking) 
     6. 找出最好的模型+驗證validation data 
          模型組合: xgboost,catboost, xgboost + catboost + voting, xgboost + catboost + stacking
          驗證方法:
           - Confusion Matrix
           - Precison 
           - Recall
     
  
- **STEP3 .** 訓練xgboost + catboost + voting [src/train_model.py](src/train_model.py)

    使用最佳模型(voting) 和參數([src/best_parameters.py](src/best_parameters.py)),重新訓練整個data.csv


- **STEP4 .** 預測  [src/predict_model.py](src/predict_model.py)
      
    將預測解果輸出到 output/submission.csv
