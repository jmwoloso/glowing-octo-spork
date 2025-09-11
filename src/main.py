

# imports
import pandas as pd
import numpy as np
from sklearn.pipeline import Pipeline
from sklearn.feature_selection import chi2, SelectKBest
from sklearn.preprocessing import OneHotEncoder
from sklearn.linear_model import LogisticRegression
from sklearn.compose import ColumnTransformer
from sklearn.calibration import CalibratedClassifierCV, calibration_curve, CalibrationDisplay
from sklearn.metrics import roc_auc_score, brier_score_loss
import matplotlib.pyplot as plt


FILE_NAMES = ['2016Q1', '2016Q2', '2016Q3', '2016Q4', '2017Q1', '2017Q2', '2017Q3', '2017Q4']
TRAIN_FILE_NAMES = ['2016Q1', '2016Q2', '2016Q3', '2016Q4', '2017Q1', '2017Q2']
TEST_FILE_NAMES = ['2017Q3']
BACKTEST_FILE_NAMES = ['2017Q4']

# Reasoning in SUMMARY.md
COLS_TO_RMV = ['pub_rec_bankruptcies', 'acc_now_delinq', 'pub_rec',
                'sec_app_chargeoff_within_12_mths', 'chargeoff_within_12_mths', 'out_prncp',
                'sec_app_collections_12_mths_ex_med', 'tot_cur_bal', 'avg_cur_bal',
                'tot_coll_amt', 'mths_since_last_record', 'last_fico_range_low',
                'last_fico_range_high', 'last_credit_pull_d']

# Categorical features
CAT_FEATS = ['term', 'sub_grade', 'emp_title', 'emp_length', 'home_ownership',
                'verification_status', 'issue_d', 'purpose', 'title', 'zip_code',
                'addr_state', 'earliest_cr_line', 'initial_list_status',
                'application_type', 'verification_status_joint', 'sec_app_earliest_cr_line']

# Values of loan_status that result in default/non-default status
DEFAULT_STATUS = ['Charged Off', 'Late (31-120 days)']
NONDEFAULT_STATUS = ['Fully Paid', 'Current', 'In Grace Period', 'Late (16-30 days)']


def process_data():
    """
    Loads in the data from CSV files, cleans up data (dropping columns, converting types, and
    handling missing values), and splits data into training and test data.
    """
    # Handle types, missing values, and obvious outliers.
    # Document any dropped columns and why.
    # Produce a quick data summary (rows, columns, target prevalence).

    train_data = pd.DataFrame
    test_data = pd.DataFrame

    # Load in train data
    for file_name in TRAIN_FILE_NAMES:
        if train_data.empty:
            train_data = pd.read_csv('data/archive/' + file_name + '.csv', low_memory=False)
        else:
            temp = pd.read_csv('data/archive/' + file_name + '.csv', low_memory=False)
            train_data = pd.concat([train_data, temp])

    # Load in test data
    for file_name in TEST_FILE_NAMES:
        if test_data.empty:
            test_data = pd.read_csv('data/archive/' + file_name + '.csv', low_memory=False)
        else:
            temp = pd.read_csv('data/archive/' + file_name + '.csv', low_memory=False)
            test_data = pd.concat([test_data, temp])

    # Cleanup data
    train_data.drop(columns=COLS_TO_RMV, inplace=True)
    train_data['int_rate'] = train_data['int_rate'].str.rstrip('%').astype(float) / 100
    train_data['revol_util'] = train_data['revol_util'].str.rstrip('%').astype(float) / 100
    train_data.dropna(axis=1, how='all')

    test_data.drop(columns=COLS_TO_RMV, inplace=True)
    test_data['int_rate'] = test_data['int_rate'].str.rstrip('%').astype(float) / 100
    test_data['revol_util'] = test_data['revol_util'].str.rstrip('%').astype(float) / 100
    test_data.dropna(axis=1, how='all')

    # Change loan_status into default
    train_data['default_status'] = train_data['loan_status'].map(lambda x:
                        '1' if x in DEFAULT_STATUS
                        else '0')

    test_data['default_status'] = test_data['loan_status'].map(lambda x:
                        '1' if x in DEFAULT_STATUS
                        else '0')

    X_train = train_data.drop(['loan_status', 'default_status'], axis=1)
    y_train = train_data[['default_status']]
    X_test = test_data.drop('default_status', axis=1)
    y_test = test_data[['default_status']]

    return X_train, y_train, X_test, y_test


def pipeline(X_train, y_train, X_test, y_test, calibration=None):
    """
    Finish preprocessing the data, train the model, and calculate/print scores. Option of
    doing no calibration, isotonic calibration, and/or sigmoid calibration.
    """
    # Handle categorical data
    cat_tf = Pipeline(
        steps=[
            ('encoder', OneHotEncoder(handle_unknown='ignore')),
            # SelectKBest performed better than SelectPercentile
            ('selector', SelectKBest(chi2, k=20))
        ]
    )

    preprocessor = ColumnTransformer(
        transformers=[
            ('cat', cat_tf, CAT_FEATS),
        ]
    )

    clf = Pipeline(
        steps=[
            ('preprocessor', preprocessor),
            ('classifier', LogisticRegression(max_iter=10000, random_state=0)),
        ]
    )

    # Train Model and Calculate Scores/Probablities
    if calibration is None or calibration == 'all':
        clf.fit(X_train, y_train.values.ravel())
        prob_pos_clf = clf.predict_proba(X_test)[:, 1]
        clf_roc_score = roc_auc_score(y_test, prob_pos_clf)
        clf_brier_score = brier_score_loss(y_test.astype(int), prob_pos_clf)

        print('With No Calibration')
        print('  Accuracy Score: ', clf.score(X_test, y_test))
        print('  ROC-AUC Score: ', clf_roc_score)
        print('  Brier Score: ', clf_brier_score)

    if calibration == 'isotonic' or calibration == 'all':
        clf_isotonic = CalibratedClassifierCV(clf, cv=2, method='isotonic')
        clf_isotonic.fit(X_train, y_train.values.ravel())
        prob_pos_isotonic = clf_isotonic.predict_proba(X_test)[:, 1]
        isotonic_roc_score = roc_auc_score(y_test, prob_pos_isotonic)
        isotonic_brier_score = brier_score_loss(y_test.astype(int), prob_pos_isotonic)
        isotonic_prob_true, isotonic_prob_pred = calibration_curve(y_test.astype(int), prob_pos_isotonic, n_bins=10)

        print('With Isotonic Calibration')
        print('  Accuracy Score: ', clf_isotonic.score(X_test, y_test))
        print('  ROC-AUC Score: ', isotonic_roc_score)
        print('  Calibration (Reliability Curve):')
        print('    Fraction of Positivies: ', isotonic_prob_true)
        print('    Mean Predicted Probability: ', isotonic_prob_pred)
        print('  Brier Score: ', isotonic_brier_score)

        disp = CalibrationDisplay.from_predictions(y_test, prob_pos_isotonic, n_bins=10)
        plt.savefig('isotonic_calibration_curve.png')

    if calibration == 'sigmoid' or calibration == 'all':
        clf_sigmoid = CalibratedClassifierCV(clf, cv=2, method='sigmoid')
        clf_sigmoid.fit(X_train, y_train.values.ravel())
        prob_pos_sigmoid = clf_sigmoid.predict_proba(X_test)[:, 1]
        sigmoid_roc_score = roc_auc_score(y_test, prob_pos_sigmoid)
        sigmoid_brier_score = brier_score_loss(y_test.astype(int), prob_pos_sigmoid)
        sigmoid_prob_true, sigmoid_prob_pred = calibration_curve(y_test.astype(int), prob_pos_sigmoid, n_bins=10)

        print('With Sigmoid Calibration')
        print('  Accuracy Score: ', clf_sigmoid.score(X_test, y_test))
        print('  ROC-AUC Score: ', sigmoid_roc_score)
        print('  Calibration (Reliability Curve):')
        print('    Fraction of Positivies: ', sigmoid_prob_true)
        print('    Mean Predicted Probability: ', sigmoid_prob_pred)
        print('  Brier Score: ', sigmoid_brier_score)

        disp = CalibrationDisplay.from_predictions(y_test, prob_pos_sigmoid, n_bins=10)
        plt.savefig('sigmoid_calibration_curve.png')

def main():
    X_train, y_train, X_test, y_test = process_data()

    pipeline(X_train, y_train, X_test, y_test, 'all')


if __name__ == '__main__':
    main()