## Grid search metrics

This folder's content is made of results and codes for tests whose objective was to explore which performance metric would be more appropriate for grid search under K-folds cross-validation aplied for binary classification datasets. The two performance metrics tested were ROC-AUC and average precision score, this last taken here also as a representative of precision-recall AUC. Two different learning methods were used during tests: logistic regression and Gradient Boosting Model (GBM).
<br>
Results reveal that both ROC-AUC and average precision score lead to very similar hyper-parameter choices, for both learning methods used. However, one should pay attention whenever a small dataset is at hand, or, more generally, whenever a hard classification task is to be performed, since this may imply different results for ROC-AUC and average precision score. In these circumstances, it could be appropriate to run two sets of grid searches, one constructed upon ROC-AUC and the other using average precision score. Then, test set performance metrics should be compared to suggest a final hyper-parameters choice.
<br>
This folder has the following components: codes (Python scripts) of model estimations for implementing experiments, and a pair of Jupyter notebook and html files with results and discussion.
