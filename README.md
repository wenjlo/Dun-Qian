# Dun-Qian


## 🚀 流程


- **STEP1 .** 查看資料 [[code](notebooks/EDA+Find%20XGB,Cat%20Parameters.ipynb)]
      - 數值資料呈現右偏態,取∛ 
      - ID = 1 這筆有4個欄位: Browser, Region ,TrafficType ,VisitorType 有遺失值. => 使用眾數補值
      - ID = 2,4,5,7 各有一個欄位有缺失值. => 使用xgboost + optuna 最佳化模型訓練並預測補值
        

- **STEP2 .** 找出 xgboost,catboost 最佳化參數並驗證 
     - 1. 從STEP1. 補值後的資料 切分train data (80%) ,valid data(20%) 
     - 2. xgboost + optuna + cross validation (5 fold) 找出最佳參數
     - 3. catboost + optuna + cross validation (5 fold) 找出最佳化參數
     - 4. 用最佳化參數重新訓練xgboost,catboost(train(80%) ,valid(20%))
     - 5. 使用ensemble 技巧 (Voting , Stacking) 
     - 6. 找出最好的模型+驗證validation data 
         - 模型組合: xgboost,catboost, xgboost + catboost + voting, xgboost + catboost + stacking
         - 驗證方法:
           - Confusion Matrix
           - Precison 
           - Recall
     - 7. 使用最佳模型,參數 重新訓練整個data.csv
  
- **STEP3 .** 訓練xgboost + catboost + voting
    