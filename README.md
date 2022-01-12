# Predict Customer Churn

- Project **Predict Customer Churn** of ML DevOps Engineer Nanodegree Udacity

## Project Description

This machine learning project trains a logistic and a random forest classifier that can be used
to identify credit card customers that are most likely to churn. The models that 
are trained are found in the './models' folder. 

Some analysis on the data used to train the models can be found in the folder 
'images/eda'. Reports related to the performance of the two classifiers can be found in 
the folder 'images/results'.

The data used for training the models and performing the analysis must be saved in 
'data/bank_data.csv'. The data currently used come from [Kaggle](https://www.kaggle.com/sakshigoyal7/credit-card-customers?select=BankChurners.csv). 
Data are expected to have the same format as the one found in the Kaggle csv file.

## Requirements
packages requirements:

  - joblib~=1.0.0
  - pandas~=1.2.0
  - numpy~=1.21.4
  - matplotlib~=3.3.2
  - seaborn~=0.11.1
  - scikit-learn~=0.23.2
  - pytest~=6.2.5

The packages can be installed using pip or conda
```bash
pip install <package>
```

## Running Files
To train the models run the following command:
```bash
python churn_library.py
```
All the functions used for the project are found in the 
'./churn_library.py' file.

## Tests
Tests are using pytest. To install pytest run the following command in your command line
```bash
pip install -U pytest
```
To make tests faster, we use a fixture that makes sure tests are run using only 8% of the Kaggle data.
Running the following command will test the functions used and run the code.
```bash
pytest churn_script_logging_and_tests.py
```
By default, pytest captures all log records emitted by your program. If you want to see 
log errors in './logs/churn_library.log', you need to disable this feature. 
To turn off the logging plugin run the following command:
```bash
pytest -p no:logging churn_script_logging_and_tests.py

```




