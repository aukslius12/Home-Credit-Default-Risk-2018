# Home-Credit-Default-Risk-2018

 * Initially tried to XGBoost, got 0.72 ROC AUC on Kaggle.
 * Created a class for data preprocessing. _Every single model_ on this preprocessed data had ROC AUC of 0.5~. No clue why.
 * Forked LightGBM from Kaggle, used his function to get preprocessed data of _all datasets_ (because i missed them before).
 * Blatantly copying and moving up the latter was boring for me, therefore I tried to use MICE to fill missing values and then use my models. Ran out of computer memory and scraped the project.
 
