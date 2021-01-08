Porto Seguro's Save Driver Prediction

In this project, I dealt with the heavily imbalanced Porto Seguro Dataset from Kaggle, which is available here:
https://www.kaggle.com/c/porto-seguro-safe-driver-prediction/data (too large for GitHub)

The focus lies on resampling and the combination of different kinds of preprocessing for different types of features.
The dataset contains binary, categorical, orindal and continuous data. 
The preprocessing is taking place within a scikit-learn Pipeline.

Also, the Kaggle competition uses the Normalized Gini Coefficient. It is defined as

2 * AUC - 1 (https://stats.stackexchange.com/questions/306287/why-use-normalized-gini-score-instead-of-auc-as-evaluation)

The scoring method is not an out of the box metrix, so another challenge was to embed this score in a cross validation.


