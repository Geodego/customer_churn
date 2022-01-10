"""
Tests for churn_library.py

author: Geoffroy de Gournay
date: January 9, 2022
"""
import os
import logging
import churn_library as cls

logging.basicConfig(
    filename='./logs/churn_library.log',
    level=logging.INFO,
    filemode='w',
    format='%(name)s - %(levelname)s - %(message)s')


def test_import(import_data):
    """
    test data import - this example is completed for you to assist with the other test functions
    """
    try:
        df = import_data("./data/bank_data.csv")
        logging.info("Testing import_data: SUCCESS")
    except FileNotFoundError as err:
        logging.error("Testing import_eda: The file wasn't found")
        raise err

    try:
        assert df.shape[0] > 0
        assert df.shape[1] > 0
    except AssertionError as err:
        logging.error(
            "Testing import_data: The file doesn't appear to have rows and columns")
        raise err
    return df


def test_eda(perform_eda, df):
    """
    test perform eda function
    """
    df['Churn'] = df['Attrition_Flag'].apply(
        lambda val: 0 if val == "Existing Customer" else 1)
    perform_eda(df)
    try:
        assert len(os.listdir('./images/eda')) > 0
        logging.info('perform_eda: SUCCESS')
    except AssertionError:
        logging.error(
            "Testing perform_eda: The output pictures haven't been saved in the eda folder")


def test_encoder_helper(encoder_helper, df, category_lst):
    """
    test encoder helper
    """
    try:
        df_new = encoder_helper(df, category_lst, None)
        # names of the columns created in encoder_helper
        response = [category + '_Churn' for category in category_lst]
        assert set(response) <= set(df_new.columns)
        logging.info('encoder_helper: SUCCESS')
    except AssertionError:
        logging.error(
            "Testing encoder_helper: The columns that should have been created in "
            "the output dataframe are not there.")


def test_perform_feature_engineering(perform_feature_engineering, df):
    """
    test perform_feature_engineering
    """
    keep_cols = [
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
    X_train, X_test, y_train, y_test = perform_feature_engineering(df)
    try:
        assert set(keep_cols) == set(X_train.columns)
        logging.info('perform_feature_engineering: SUCCESS')
    except AssertionError:
        logging.error(
            "Testing perform_feature_engineering: The output dataframes don't have "
            "the expected columns")
    return X_train, X_test, y_train, y_test


def test_train_models(train_models, X_train, X_test, y_train, y_test):
    """
    test train_models
    """
    train_models(X_train, X_test, y_train, y_test)
    try:
        assert len(os.listdir('./images/results')) > 0
        assert len(os.listdir('./models')) > 0
        logging.info('train_models: SUCCESS')
    except AssertionError:
        logging.error(
            "Testing train_models: some file haven't been saved in the results "
            "folder or in the models folder")


if __name__ == "__main__":
    df1 = test_import(cls.import_data)
    test_eda(cls.perform_eda, df1)
    cat_columns = [
        'Gender',
        'Education_Level',
        'Marital_Status',
        'Income_Category',
        'Card_Category']
    test_encoder_helper(cls.encoder_helper, df1, cat_columns)
    X_train1, X_test1, y_train1, y_test1 = test_perform_feature_engineering(
        cls.perform_feature_engineering, df1)
    test_train_models(cls.train_models, X_train1, X_test1, y_train1, y_test1)







