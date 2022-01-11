"""
Tests for churn_library.py

author: Geoffroy de Gournay
date: January 9, 2022
"""
import os
import logging
import pytest
import churn_library as cls

logging.basicConfig(
    filename='./logs/churn_library.log',
    level=logging.INFO,
    filemode='w',
    format='%(name)s - %(levelname)s - %(message)s')


def test_import():
    """
    test data import - this example is completed for you to assist with the other test functions
    """
    try:
        data1 = cls.import_data("./data/bank_data.csv")
        logging.info("Testing import_data: SUCCESS")
    except FileNotFoundError as err:
        logging.error("Testing import_eda: The file wasn't found")
        raise err

    try:
        assert data1.shape[0] > 0
        assert data1.shape[1] > 0
    except AssertionError as err:
        logging.error(
            "Testing import_data: The file doesn't appear to have rows and columns")
        raise err
    return data1


@pytest.fixture()
def data():
    """
    Creates a dataframe with the proper columns that can be used to run tests without
    having to download data from Kaggle. This allows to run tests faster. The first time
    this fixture is called, it reads the Kaggle data and save the 200 first
    data './data/test_data.csv'
    """
    try:
        df = cls.import_data('./data/test_data.csv')
    except FileNotFoundError:
        df = cls.import_data("./data/bank_data.csv")
        df = df.sample(frac=0.08, replace=False)
        df.to_csv('./data/test_data.csv')
    # makes sure the column 'Churn' is available for all tests
    df['Churn'] = df['Attrition_Flag'].apply(
        lambda val: 0 if val == "Existing Customer" else 1)
    return df


def test_eda(data):
    """
    test perform eda function
    """
    perform_eda = cls.perform_eda
    perform_eda(data)
    try:
        assert len(os.listdir('./images/eda')) > 0
        logging.info('perform_eda: SUCCESS')
    except AssertionError:
        logging.error(
            "Testing perform_eda: The output pictures haven't been saved in the eda folder")


def test_encoder_helper(data):
    """
    test encoder helper
    """
    encoder_helper = cls.encoder_helper
    category_lst = [
        'Gender',
        'Education_Level',
        'Marital_Status',
        'Income_Category',
        'Card_Category']
    try:
        df_new = encoder_helper(data, category_lst, None)
        # names of the columns created in encoder_helper
        response = [category + '_Churn' for category in category_lst]
        assert set(response) <= set(df_new.columns)
        logging.info('encoder_helper: SUCCESS')
    except AssertionError:
        logging.error(
            "Testing encoder_helper: The columns that should have been created in "
            "the output dataframe are not there.")


@pytest.fixture()
def feature_engineering(data):
    """
    Fixture returning (X_train, X_test, y_train, y_test) as processed
    by perform_feature_engineering.
    """
    perform_feature_engineering = cls.perform_feature_engineering
    X_train, X_test, y_train, y_test = perform_feature_engineering(data)
    return X_train, X_test, y_train, y_test


def test_perform_feature_engineering(feature_engineering):
    """
    test perform_feature_engineering
    """
    keep_cols = cls.Keep_Cols
    X_train, X_test, y_train, y_test = feature_engineering
    try:
        assert set(keep_cols) == set(X_train.columns)
        logging.info('perform_feature_engineering: SUCCESS')
    except AssertionError:
        logging.error(
            "Testing perform_feature_engineering: The output dataframes don't have "
            "the expected columns")
    return X_train, X_test, y_train, y_test


def test_train_models(feature_engineering):
    """
    test train_models
    """
    X_train, X_test, y_train, y_test = feature_engineering
    train_models = cls.train_models
    train_models(X_train, X_test, y_train, y_test)
    try:
        assert len(os.listdir('./images/results')) > 0
        assert len(os.listdir('./models')) > 0
        logging.info('train_models: SUCCESS')
    except AssertionError:
        logging.error(
            "Testing train_models: some file haven't been saved in the results "
            "folder or in the models folder")
