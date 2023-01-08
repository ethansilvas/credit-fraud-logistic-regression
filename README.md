# Module 11 Challenge - Credit Fraud Logistic Regression

In this project I apply and compare logistic regression models to an imbalanced dataset of historical lending activity in order to predict healthy and high-risk loans. 

### Data Used
[lending_data.csv](/Resources/lending_data.csv) - labeled (0 - healthy, 1 - high-risk) historical lending activity from a peer-to-peer lending services company

---

## Overview of the Analysis

This analysis aims to compare two logistic regression models, one that trains with imbalanced data and one that uses random oversampling, to see the differences in their predictive performance. The dataset used is labeled loan data with features loan_size, interest_rate, borrower_income, debt_to_income, num_of_accounts, derogatory_marks, and total_debt as shown below: 

![DataFrame head showing features loan_size, interest_rate, borrower_income, debt_to_income, num_of_accounts, derogatory_marks, and total_debt](/Resources/Images/features.png)

The loan_status column is the label to distinguish between healthy loans (0) and high-risk loans (1), but the original data is heavily imbalanced with 75,036 healthy loans and 2500 high-risk loans. 

Both models are mostly the same as both are scikit-learn LogisticRegression models. However they differ because, after splitting the data into training and testing data, one is trained using the original (imbalanced) data and the other is trained using randomly oversampled data which end up in an even 56,271 values for both healthy and high-risk loans. After they are trained, they both predict on the same testing data and the results are analyzed using scikit-learn's balanced_accuracy_score, confusion_matrix, and classification_report_imbalanced methods. 

## Results

* LogisticRegression model trained on original, imbalanced, data:
    * Balanced accuracy score = 0.9520479254722232
    * Precision scores:
        * Healthy loans = 1.0 = Of the loans that the model predicted to be healthy, about 100% of them were actually healthy loans
        * High-risk loans = 0.85 = Of the loans that the model predicted to be high-risk, about 85% of them were actually high-risk loans
    * Recall scores: 
        * Healthy loans = 0.99 = Of all the actually healthy loans, the model correctly predicted them to be healthy about 99% of the time
        * High-risk loans = 0.91 = Of all the actually high-risk loans, the model correctly predicted them to be high-risk about 91% of the time

    Confusion matrix:<br>
    ![Confusion matrix showing 18663 to 102 for healthy loans and 56 to 563 for high-risk loans](/Resources/Images/original-confusion-matrix.png)


* LogisticRegression model trained on randomly oversampled data:
    * Balanced accuracy score: 0.9936781215845847
    * Precision scores:
        * Healthy loans = 1.0 = Of the loans that the model predicted to be healthy, about 100% of them were actually healthy loans
        * High-risk loans = 0.84 = Of the loans that the model predicted to be high-risk, about 84% of them were actually high-risk loans
    * Recall scores: 
        * Healthy loans = 0.99 = Of all the actually healthy loans, the model correctly predicted them to be healthy about 99% of the time
        * High-risk loans = 0.99 = Of all the actually high-risk loans, the model correctly predicted them to be high-risk about 99% of the time

    Confusion matrix:<br>
    ![Confusion matrix showing 18649 to 116 for healthy loans and 4 to 615 for high-risk loans](/Resources/Images/oversampled-confusion-matrix.png)


## Summary

Since this model focuses on predicting high-risk loans, I would recommend using the randomly oversampled model because it has a 0.99 recall score for high-risk loans compared to the original data model's recall of 0.91 for high-risk loans. This increase in recall score only comes at the cost of a 0.01 reduction in precision for high-risk loans, but this is negligible since the score is still pretty high at 0.84. 

Things to keep in mind with these recommendation/results is that there will likely need to be a check for overfitting to our data and it would be a good idea to run this analysis with a validation set as well. However, assuming that the models learned well and aren't highly overfit to the dataset, then it can be said that oversampling for the purpose of predicting high-risk loans is beneficial to performance.

---

## Technologies

This is a Python 3.8 project ran in Google Colab but can be used in JupyterLab using a Conda dev environment. 

The following dependencies are used: 
1. [Jupyter](https://jupyter.org/) - Running code 
2. [Conda](https://github.com/conda/conda) (4.13.0) - Dev environment
3. [Pandas](https://github.com/pandas-dev/pandas) (1.3.5) - Data analysis
4. [Numpy](https://numpy.org/) (1.21.5) - Data calculations + Pandas support
5. [Scikit-learn](https://scikit-learn.org/stable/index.html) (1.0.2) - Machine learning models and tools
6. [Imbalanced-learn](https://imbalanced-learn.org/stable/) (0.10.1) - Imbalanced classification dataset tools

---

## Installation Guide

If you would like to run the program in JupyterLab, install the [Anaconda](https://www.anaconda.com/products/distribution) distribution and run `jupyter lab` in a conda dev environment.

To ensure that your notebook runs properly you can use the [requirements.txt](/Resources/requirements.txt) file to create an exact copy of the conda dev environment used in development of this project. 

Create a copy of the conda dev environment with `conda create --name myenv --file requirements.txt`

Then install the requirements with `conda install --name myenv --file requirements.txt`

---

## Usage

The Jupyter notebook [credit_risk_resampling_ipynb](/credit_risk_resampling.ipynb) will provide all steps of the data collection, preparation, and analysis. Data visualizations are shown inline and accompanying analysis responses are provided.

---

## Contributors

[Ethan Silvas](https://github.com/ethansilvas)

---

## License

This project uses the [GNU General Public License](https://choosealicense.com/licenses/gpl-3.0/)