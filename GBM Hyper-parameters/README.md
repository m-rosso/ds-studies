## GBM hyper-parameters

This folder brings results and codes from tests conducted with the goal of assessing the specification of Gradient Boosting Model (GBM) hyper-parameters. The study opposes theoretical considerations to empirical evidence constructed upon the classification of several distinct datasets with binary response variables. As a result, appropriate ranges of values for the most important GBM hyper-parameters could be defined, namely: subsample, max depth of trees, learning rate, and the number of estimators. For the datasets used during tests, a subsample < 1 (with best outcomes for subsample = 0.75), a moderate value of max depth (no larger than 5), a very small learning rate (like 0.01), and a moderate number of estimators (no more than 500) displayed the best performance metrics.
<br>
As discussed in the file "Tests over GBM Hyper-parameters", these tests do not aim to put a final word on how to define GBM hyper-parameters for any supervised learning task. Instead, they help understanding how main hyper-parameters may affect predictive accuracy, and consequently may contribute defining appropriate values for data science tests in which performance metrics are not expected to be the best possible.
<br>
In addition to that file (Jupyter notebook and html extensions) with results presentation and discussion, this folder contains all codes used for model estimation.
<br>
<br>
**Note:** this inquirement does not consider the application of AutoML techniques. Even so, it is not contradictory with automated methods for model selection and hyper-parameters definition, since it may guide the specification of range of values for hyper-parameters of GBM. A robust machine learning system, though, should define its hyper-parameters preferably through automated methods.
