<h1 align="center">Brain MRI Regression</h1>

This is the code for the Advanced Machine Learning (AML) project. The assignment consisted in developing an ML model, using brain MRI features, capable of predicting an individual's age.

To simplify the training process, 832 features had been extracted from MRI scans using [FreeSurfer](https://surfer.nmr.mgh.harvard.edu/). These features were made available in a tabular format.

Then, to complicate the task, the data was modified to introduce
- missing values        🡒 requiring imputation
- outliers              🡒 requiring outlier detection
- irrelevant features   🡒 requiring feature extraction

## 🛠️ Installation

## 🏗️ Training

## 🔮 Inference

## 📏 Evaluation Metric
The model's performance was measured using the **coefficient of determination ($R^2$ score)**.

Specifically, the $R^2$ score is the proportion of the variance in the dependent variable that is predictable from the independent variables. 

The $R^2$ score takes values in the range $(-\infty, 1]$, with 1 indicating a perfect fit.