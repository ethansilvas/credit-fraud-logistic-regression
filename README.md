# Module 11 Challenge - Credit Fraud Logistic Regression

In this project I apply and compare logistic regression models to an imbalanced dataset of historical lending activity in order to predict healthy and high-risk loans. 

### Data Used
[lending_data.csv](/Resources/lending_data.csv) - labeled (0 - healthy, 1 - high-risk) historical lending activity from a peer-to-peer lending services company

---

## Overview of the Analysis

This analysis aims to compare two logistic regression models, one that trains with imbalanced data and one that uses random oversampling, to see the differences in their predictive performance. The dataset used is labeled loan data with features loan_size, interest_rate, borrower_income, debt_to_income, num_of_accounts, derogatory_marks, and total_debt as shown below: 

![DataFrame head showing features loan_size, interest_rate, borrower_income, debt_to_income, num_of_accounts, derogatory_marks, and total_debt](/Resources/features.png)

The loan_status column is the label to distinguish between healthy loans (0) and high-risk loans (1), but the original data is heavily imbalanced with 75,036 healthy loans and 2500 high-risk loans. 

Each model is mostly the same as both are scikit-learn LogisticRegression models. However they differ because, after splitting the data into training and testing data, one is trained using the original (imbalanced) data and the other is trained using randomly oversampled data which end up in an even 56,271 values for both healthy and high-risk loans. After they are trained, they both predict on the same testing data and the results are analyzed using scikit-learn's balanced_accuracy_score, confusion_matrix, and classification_report_imbalanced methods. 

## Results

Using bulleted lists, describe the balanced accuracy scores and the precision and recall scores of all machine learning models.

* Machine Learning Model 1:
  * Description of Model 1 Accuracy, Precision, and Recall scores.



* Machine Learning Model 2:
  * Description of Model 2 Accuracy, Precision, and Recall scores.

## Summary

Summarize the results of the machine learning models, and include a recommendation on the model to use, if any. For example:
* Which one seems to perform best? How do you know it performs best?
* Does performance depend on the problem we are trying to solve? (For example, is it more important to predict the `1`'s, or predict the `0`'s? )

If you do not recommend any of the models, please justify your reasoning.

---

## Technologies

This is a Python 3.8 project ran in Google Colab but can be used in JupyterLab using a Conda dev environment. 

The following dependencies are used: 
1. [Jupyter](https://jupyter.org/) - Running code 
2. [Conda](https://github.com/conda/conda) (4.13.0) - Dev environment
3. [Pandas](https://github.com/pandas-dev/pandas) (1.3.5) - Data analysis
4. [Numpy](https://numpy.org/) (1.21.5) - Data calculations + Pandas support


---

## Installation Guide

If you would like to run the program in JupyterLab, install the [Anaconda](https://www.anaconda.com/products/distribution) distribution and run `jupyter lab` in a conda dev environment.

To ensure that your notebook runs properly you can use the [requirements.txt](/Resources/requirements.txt) file to create an exact copy of the conda dev environment used in development of this project. 

Create a copy of the conda dev environment with `conda create --name myenv --file requirements.txt`

Then install the requirements with `conda install --name myenv --file requirements.txt`

---

## Usage

The Jupyter notebook []() will provide all steps of the data collection, preparation, and analysis. Data visualizations are shown inline and accompanying analysis responses are provided.

---

## Contributors

[Ethan Silvas](https://github.com/ethansilvas)

---

## License

This project uses the [GNU General Public License](https://choosealicense.com/licenses/gpl-3.0/)