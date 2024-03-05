# financial_risk-_detection_ML
- This aim of this project is to build a ML classification model to classify people who default and dont default loan payments using historical data.
- The 'class-1' are people who default loan payments and 'class-0' are people who do not default loan payments.
- The dataset was heavily imbalanced with 'class-0' dominating with more than 2,60,000 rows of data compared to 'class-1' with 24,825 rows.
- I first tried **oversampling** the classes and building an ML model .The model was able to classify 'class-0' with excellent accuracy but very poor to detect 'class-1'.Ref **financial_risk_ML_oversampling.ipynb**
- ## Classification ROCAUC score and Accuracy Table for oversampled dataset
|    Model             |  Train(ROC-AUC)   |  Test(ROC-AUC)   |Accuracy |Confusion matrix
| :------------------- | -----------------  |-----------------|-----------------|-----------------:  
| RandomForestClassifier    |      0.971         |0.680             |0.899|   | [[68683  2104]   |
|      -         |-             |-| -|                                           [ 5602   489]]    |
| LogisticRegression|      0.884         |0.651              |0.776      |[[57253 13534]| 
|      -        |         -     |   -   |      -|          [ 3676  2415]]|
| XGBClassifier             |      0.981         |0.752              |0.922 |[[70581   206]             |
|     -         |-              |- |-                            | [ 5927   164]]| 
| HistGradientClassifier            |     0.981         |0.751              |0.920|[[70668   119]| 
|    -         |-              |- |-| [ 5992    99]]| 
| CatBoostClassifier            |     0.894         |0.660              |0.805| [[59895 10892]| 
 |   -         |-              |- |     -|                                        [ 4065  2026]]|
          |            
- Then I used **random undersampling** to build ML model , It gave me good results and I continued feature selection with the undersampled dataset and found out that 3 features were enough to classify 'class-0' and 'class-1'.Ref **financial_risk_ML_undersampling_feature_selection.ipynb** .
- Those features are **'SK_ID_CURR', EXT_SOURCE_2, EXT_SOURCE_3**
- 'SK_ID_CURR'-ID of loan in our sample
- 'EXT_SOURCE_2'-Normalized score from external data source
- 'EXT_SOURCE_3'-Normalized score from external data source
* So if the 'EXT_SOURCE_2' and 'EXT_SOURCE_3' scores are **less than or equal to 0.41** the loan is defaulted nearly 65% of the times, ref section on **financial_risk_ML_undersampling_feature_selection.ipynb** --> Running the ML model again with the 2 features excluding 'SK_ID_CURR'
* To get to 96% accuracy for the loan defaulters ,'SK_ID_CURR'  feature is taken into consideration which is just the ID of the loan application.
* From this what I can Infer is that if the 'EXT_SOURCE_2' and 'EXT_SOURCE_3' scores are greater than **0.41** ,31% of the times the loan gets defaulted due to unknown reasons.
- ## Classification ROCAUC score and Accuracy Table for undersampled dataset
|    Model             |  Train(ROC-AUC)   |  Test(ROC-AUC)   |Accuracy |Confusion matrix
| :------------------- | -----------------  |-----------------|-----------------|-----------------:  
| RandomForestClassifier    |      0.984         |0.977             |0.958|  [[7401    0]   |                            
|             -             |              -|                       |      |   [ 624 6824]] |  
| LogisticRegression|      0.967         |0.968              |0.946   |  [[7332   69] |
 |      -        |         -     |   -   |         -      |                [ 720 6728]] |
| XGBClassifier             |      0.979         |0.975              |0.956 |[[7380   21]            |
|     -         |-              |- |      -|                                     [ 628 6820]] |
| HistGradientClassifier            |     0.976         |0.974              |0.957|  [[7399    2]            |
|    -         |-              |-|   -|                                              [ 624 6824]]]|
| CatBoostClassifier            |     0.972         |0.973              |0.956| [[7374   27]            |
|     -         |-              |-|  -|                                          [ 623 6825]] | 
### Hope the results are good!!!
