"""
This a library of functions used to predict customers who are likely to churn

author: Geoffroy de Gournay
date: January 9, 2022
"""

import os
import joblib
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import plot_roc_curve, classification_report

os.environ['QT_QPA_PLATFORM'] = 'offscreen'
sns.set()

# Columns we are interested in, found in the Kaggle data
Keep_Cols = [
    'Customer_Age',
    'Dependent_count',
    'Months_on_book',
    'Total_Relationship_Count',
    'Months_Inactive_12_mon',
    'Contacts_Count_12_mon',
    'Credit_Limit',
    'Total_Revolving_Bal',
    'Avg_Open_To_Buy',
    'Total_Amt_Chng_Q4_Q1',
    'Total_Trans_Amt',
    'Total_Trans_Ct',
    'Total_Ct_Chng_Q4_Q1',
    'Avg_Utilization_Ratio',
    'Gender_Churn',
    'Education_Level_Churn',
    'Marital_Status_Churn',
    'Income_Category_Churn',
    'Card_Category_Churn']


def import_data(pth):
    """
    returns dataframe for the csv found at pth

    input:
            pth: a path to the csv
    output:
            df: pandas dataframe
    """
    df = pd.read_csv(pth, index_col=0)
    return df


def perform_eda(df):
    """
    perform eda on df and save figures to images folder
    input:
            df: pandas dataframe

    output:
            None
    """
    # Plot histogram showing customer churn and save the plot in images
    df['Churn'] = df['Attrition_Flag'].apply(
        lambda val: 0 if val == "Existing Customer" else 1)
    plt.figure(figsize=(20, 10))
    df['Churn'].hist()
    plt.xlabel('category')
    plt.ylabel('nber customers')
    plt.title('Churn distribution')
    plt.savefig('./images/eda/churn_distribution.png')
    plt.close()

    # plot age histogram
    plt.figure(figsize=(20, 10))
    df['Customer_Age'].hist()
    plt.xlabel('age')
    plt.ylabel('nber customers')
    plt.title('Age distribution')
    plt.savefig('./images/eda/customer_age_distribution.png')
    plt.close()

    # plot marital status
    plt.figure(figsize=(20, 10))
    df.Marital_Status.value_counts('normalize').plot(kind='bar')
    plt.ylabel('nber customers')
    plt.title('Marital status distribution')
    plt.savefig('./images/eda/marital_status_distribution.png')
    plt.close()

    # plot total trans ct:
    plt.figure(figsize=(20, 10))
    sns.distplot(df['Total_Trans_Ct'])
    plt.title('total transaction distribution')
    plt.savefig('./images/eda/total_transaction_distribution.png')
    plt.close()

    # plot correlation heatmap
    plt.figure(figsize=(20, 10))
    sns.heatmap(df.corr(), annot=False, cmap='Dark2_r', linewidths=2)
    plt.savefig('./images/eda/heatmap.png')
    plt.close()


def encoder_helper(df, category_lst, response=None):
    """
    helper function to turn each categorical column into a new column with
    proportion of churn for each category - associated with cell 15 from the notebook

    input:
            df: pandas dataframe
            category_lst: list of columns that contain categorical features
            response: string of response name [optional argument that could be used for
            naming variables or index y column]

    output:
            df: pandas dataframe with new columns for
    """
    if response is None:
        response = [category + '_Churn' for category in category_lst]

    for category, col_name in zip(category_lst, response):
        # add a column 'col_name', which rows are the average value for
        # the category the row belongs to
        df[col_name] = df.groupby(category)['Churn'].transform('mean')

    return df


def perform_feature_engineering(df, response=None):
    """
    input:
              df: pandas dataframe
              response: string of response name [optional argument that could be used
              for naming variables or index y column]

    output:
              X_train: X training data
              X_test: X testing data
              y_train: y training data
              y_test: y testing data
    """
    cat_columns = [
        'Gender',
        'Education_Level',
        'Marital_Status',
        'Income_Category',
        'Card_Category'
    ]

    if response is None:
        response = [category + '_Churn' for category in cat_columns]

    df = encoder_helper(df, cat_columns, response)
    y = df['Churn']
    X = pd.DataFrame()
    keep_cols = Keep_Cols

    X[keep_cols] = df[keep_cols]
    # train test split
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.3, random_state=42)
    return X_train, X_test, y_train, y_test


def classification_report_image(y_train,
                                y_test,
                                y_train_preds_lr,
                                y_train_preds_rf,
                                y_test_preds_lr,
                                y_test_preds_rf):
    """
    produces classification report for training and testing results and stores report as image
    in images folder
    input:
            y_train: training response values
            y_test:  test response values
            y_train_preds_lr: training predictions from logistic regression
            y_train_preds_rf: training predictions from random forest
            y_test_preds_lr: test predictions from logistic regression
            y_test_preds_rf: test predictions from random forest

    output:
             None
    """
    methods = ['rf', 'lr']
    predictions = {
        'lr': {
            'train': y_train_preds_lr,
            'test': y_test_preds_lr,
            'name': 'Logistic Regression',
            'file': 'logistic_results.png'},
        'rf': {
            'train': y_train_preds_rf,
            'test': y_test_preds_rf,
            'name': 'Random Forest',
            'file': 'rf_results.png'}}
    for method in methods:
        pred = predictions[method]['train']
        title = predictions[method]['name']
        plt.figure()
        plt.rc('figure', figsize=(8, 8))
        plt.text(0.01, 1.1, str(title + ' Train'),
                 {'fontsize': 10}, fontproperties='monospace')
        plt.text(0.01, 0.7, str(classification_report(y_train, pred)), {'fontsize': 10},
                 fontproperties='monospace')  # approach improved by OP -> monospace!
        plt.text(0.01, 0.6, str(title + ' Test'),
                 {'fontsize': 10}, fontproperties='monospace')
        plt.text(0.01, 0.2, str(classification_report(y_test, y_test_preds_lr)), {'fontsize': 10},
                 fontproperties='monospace')  # approach improved by OP -> monospace!
        report_name = predictions[method]['file']
        plt.axis('off')
        plt.title(report_name)
        plt.savefig('./images/results/' + report_name)
        plt.close()


def feature_importance_plot(model, X_data, output_pth):
    """
    creates and stores the feature importances in pth
    input:
            model: model object containing feature_importances_
            X_data: pandas dataframe of X values
            output_pth: path to store the figure

    output:
             None
    """
    # Calculate feature importances
    importances = model.feature_importances_

    # Sort feature importances in descending order
    indices = np.argsort(importances)[::-1]

    # Rearrange feature names so they match the sorted feature importances
    names = [X_data.columns[i] for i in indices]

    # Create plot
    plt.figure(figsize=(20, 16))

    # Create plot title
    plt.title("Feature Importance")
    plt.ylabel('Importance')

    # Add bars
    plt.bar(range(X_data.shape[1]), importances[indices])

    # Add feature names as x-axis labels
    plt.xticks(range(X_data.shape[1]), names, rotation=90)

    # save figure
    plt.savefig(output_pth)
    plt.close()


class Classifier():
    """
    Parent class for the classifiers
    """

    def __init__(self):
        self.model = None
        self.name = ""

    def train_model(self, X_train, y_train):
        """
            train model
            input:
                      X_train: X training data
                      X_test: X testing data
                      y_train: y training data
                      y_test: y testing data
            output:
                      None
        """
        self.model.fit(X_train, y_train)

    def save_model(self):
        """
        Save the model in the folder './model'
        """
        path = './models/{}_model.pkl'.format(self.name)
        joblib.dump(self.model, path)


class LRClassifier(Classifier):
    """
    Class handling operations using Logistic Regression.
    """

    def __init__(self):
        super().__init__()
        self.model = LogisticRegression()
        self.name = 'logistic'


class RFClassifier(Classifier):
    """
    Class handling operations using Random Forest.
    """
    def __init__(self):
        super().__init__()
        self.estimator = RandomForestClassifier(random_state=42)
        self.name = 'rfc'
        self.param_grid = {
            'n_estimators': [200, 500],
            'max_features': ['auto', 'sqrt'],
            'max_depth': [4, 5, 100],
            'criterion': ['gini', 'entropy']
        }

    def train_model(self, X_train, y_train):
        """
        Super class training method is overwritten for RFC
        """
        rfc = self.estimator
        cv_rfc = GridSearchCV(estimator=rfc, param_grid=self.param_grid, cv=5)
        cv_rfc.fit(X_train, y_train)
        rfc_best = cv_rfc.best_estimator_
        self.model = rfc_best


def train_models(X_train, X_test, y_train, y_test):
    """
    train, store model results: images + scores, and store models.
    If rfc and lrc are provided no training is done.
    input:
              X_train: X training data
              X_test: X testing data
              y_train: y training data
              y_test: y testing data
    output:
              None
    """
    rfc = RFClassifier()
    rfc.train_model(X_train, y_train)
    lrc = LRClassifier()
    lrc.train_model(X_train, y_train)

    # plots
    plt.figure(figsize=(15, 8))
    plot_roc_curve(lrc.model, X_test, y_test)
    ax = plt.gca()
    plot_roc_curve(rfc.model, X_test, y_test, ax=ax, alpha=0.8)

    # save figure
    plt.savefig('./images/results/roc_curve_result.png')
    plt.close()

    # save reports
    y_train_preds_rf = rfc.model.predict(X_train)
    y_test_preds_rf = rfc.model.predict(X_test)
    y_train_preds_lr = lrc.model.predict(X_train)
    y_test_preds_lr = lrc.model.predict(X_test)
    classification_report_image(
        y_train,
        y_test,
        y_train_preds_lr,
        y_train_preds_rf,
        y_test_preds_lr,
        y_test_preds_rf)

    # plot feature importance
    feature_importance_plot(
        rfc.model,
        X_train,
        './images/results/feature_importances.png')

    # save best models
    rfc.save_model()
    lrc.save_model()


def get_models():
    """
    Read the models that have already been trained
    input:
              None
    output:
              lr_model, rfc_model
            where lr_model is the logistic regression pretrained model and
            rfc_model is the random forest pretrained model.
    """
    rfc_model = joblib.load('./models/rfc_model.pkl')
    lr_model = joblib.load('./models/logistic_model.pkl')
    return lr_model, rfc_model


if __name__ == '__main__':
    df1 = import_data(r"./data/bank_data.csv")
    perform_eda(df1)
    X_train1, X_test1, y_train1, y_test1 = perform_feature_engineering(df1)
    train_models(X_train1, X_test1, y_train1, y_test1)
