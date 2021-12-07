import  numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import precision_score, recall_score, roc_auc_score, confusion_matrix, accuracy_score, f1_score, make_scorer
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split, cross_val_score, cross_val_predict, GridSearchCV, StratifiedKFold
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.svm import SVC
from sklearn.neighbors import KNeighborsClassifier


def save_solution(prediction, prefix=''):
    sub = pd.read_csv('gender_submission.csv')
    new_sub = sub.copy(deep=True)
    new_sub['Survived'] = prediction
    new_sub.to_csv(prefix + '_submission.csv', index=False)


def model_differences(model1_pred, model2_pred):
    diff = [i != j for i, j in zip(model1_pred, model2_pred)]
    ind = [i for i, x in enumerate(diff) if x]
    return f'number of mismatched responses: {sum(diff)},\nindexes : {ind}'


def ensemble(*models):

    if len(models) == 0:
        log_reg_sub = pd.read_csv('log_reg_submission.csv')
        rf_sub = pd.read_csv('random_forest_submission.csv')
        gb_sub = pd.read_csv('gradient_boosting_submission.csv')
        svm_sub = pd.read_csv('support_vector_submission.csv')
        knn_sub = pd.read_csv('k_neighbors_submission.csv')
        df = pd.DataFrame()
        df['log_reg'] = log_reg_sub['Survived']
        df['rand_for'] = rf_sub['Survived']
        df['grad_boost'] = gb_sub['Survived']
        df['sup_vec'] = svm_sub['Survived']
        df['k_neigh'] = knn_sub['Survived']
        df['majority'] = df.mode(axis=1)[0] # majority vote
    else:
        df = pd.read_csv('gender_submission.csv')
        for i, m in enumerate(models, 1):
            df[f'model_{i}'] = m
        df.drop(['PassengerId', 'Survived'], axis=1, inplace=True)
        df['majority'] = df.mode(axis=1)[0] # majority vote

    return df


def hyperparameter_optimization(data, target, model, parameters, metric):

    my_score = make_scorer(metric)
    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
    clf = GridSearchCV(model, parameters, scoring=my_score, cv=cv)
    clf.fit(data, target)

    print(f'best estimator:\n{clf.best_estimator_}')
    print(f'best score: {clf.best_score_}')
    print(f'best params:\n{clf.best_params_}')
    print(f'cross-val results:\n{clf.cv_results_}')

    return clf


def log_reg_model(X_train, X_test, y_train, y_test=None):

    # log_reg = LogisticRegression(penalty='elasticnet', fit_intercept=False, solver='saga', l1_ratio=0.5, random_state=42)
    log_reg = LogisticRegression(penalty='l2', fit_intercept=False, random_state=42)
    log_reg.fit(X_train, y_train) # log_reg learning
    test_pred = log_reg.predict(X_test) # prediction

    if y_test is not None:
        train_pred = log_reg.predict(X_train)
        # HOLD OUT PRECISION SCORE
        train_precision_score = precision_score(train_pred, y_train)
        test_precision_score = precision_score(test_pred, y_test)
        # HOLD OUT RECALL SCORE
        train_recall_score = recall_score(train_pred, y_train)
        test_recall_score = recall_score(test_pred, y_test)
        # HOLD OUT ROC AUC SCORE
        train_roc_auc_score = roc_auc_score(train_pred, y_train)
        test_roc_auc_score = roc_auc_score(test_pred, y_test)
        # RESULTS
        print(f'train_precision_score: {train_precision_score}')
        print(f'test_precision_score: {test_precision_score}')
        print(f'train_recall_score: {train_recall_score}')
        print(f'test_recall_score: {test_recall_score}')
        print(f'train_roc_auc_score: {train_roc_auc_score}')
        print(f'test_roc_auc_score: {test_roc_auc_score}')

    return test_pred


def random_forest_model(X_train, X_test, y_train, y_test=None):

    rfc = RandomForestClassifier(criterion='gini', n_estimators=3650, # criterion='entropy'
                                max_depth=15, min_samples_split=10,
                                min_samples_leaf=1, max_features=3,
                                bootstrap = True, oob_score=True,
                                random_state=42, n_jobs=-1)
    rfc.fit(X_train, y_train) # random forest learning
    test_pred = rfc.predict(X_test) # prediction

    if y_test is not None:
        train_pred = rfc.predict(X_train)
        # ACCURACY
        print('RF train Accuracy: ' + repr(round(rfc.score(X_train, y_train) * 100, 2)) + '%')
        print('RF test Accuracy: ' + repr(round(rfc.score(X_test, y_test) * 100, 2)) + '%')
        # HOLD OUT ROC AUC SCORE
        train_roc_auc_score = roc_auc_score(train_pred, y_train)
        test_roc_auc_score = roc_auc_score(test_pred, y_test)
        print(f'train_roc_auc_score: {train_roc_auc_score}')
        print(f'test_roc_auc_score: {test_roc_auc_score}')
        # CROSS VALIDATION SCORE
        result_cv = cross_val_score(rfc, X_train, y_train, cv=5, scoring='accuracy', n_jobs=-1)
        print('The CV score for RF is:', round(result_cv.mean() * 100, 2))
        # CONFUSION MATRIX
        y_pred = cross_val_predict(rfc, X_train, y_train, cv=5, n_jobs=-1)
        rfcm = confusion_matrix(y_train, y_pred)
        print(f'confusion matrix for CVRF: {np.ndarray.flatten(rfcm)}')

    return test_pred


def gradient_boosting_model(X_train, X_test, y_train, y_test=None):

    gbc = GradientBoostingClassifier(loss='deviance', learning_rate=0.05,
                                    n_estimators = 100, subsample=0.5,
                                    criterion='friedman_mse', max_depth=4,
                                    random_state=42, max_features=4,
                                    n_iter_no_change=10, tol=0.0011)
    gbc.fit(X_train, y_train) # gradient boosting learning
    test_pred = gbc.predict(X_test) # prediction

    if y_test is not None:
        train_pred = gbc.predict(X_train)
        # HOLD OUT ROC AUC SCORE
        train_roc_auc_score = roc_auc_score(train_pred, y_train)
        test_roc_auc_score = roc_auc_score(test_pred, y_test)
        print(f'train_roc_auc_score: {train_roc_auc_score}')
        print(f'test_roc_auc_score: {test_roc_auc_score}')
        # CONFUSION MATRIX
        y_pred = cross_val_predict(gbc, X_train, y_train, cv=5, n_jobs=-1)
        gbcm = confusion_matrix(y_train, y_pred)
        print(f'confusion matrix for CVGB: {np.ndarray.flatten(gbcm)}')

    return test_pred


def support_vector_model(X_train, X_test, y_train, y_test=None):

    svc = SVC(C=2, kernel='poly', degree=2,
              gamma='auto', max_iter=-1,
              random_state=42)
    svc.fit(X_train, y_train) # support vector machine learning
    test_pred = svc.predict(X_test) # prediction

    if y_test is not None:
        train_pred = svc.predict(X_train)
        # HOLD OUT F1 SCORE
        train_f1_score = f1_score(train_pred, y_train)
        test_f1_score = f1_score(test_pred, y_test)
        # HOLD OUT ROC AUC SCORE
        train_roc_auc_score = roc_auc_score(train_pred, y_train)
        test_roc_auc_score = roc_auc_score(test_pred, y_test)
        # RESULTS
        print(f'train_f1_score: {train_f1_score}')
        print(f'test_f1_score: {test_f1_score}')
        print(f'train_roc_auc_score: {train_roc_auc_score}')
        print(f'test_roc_auc_score: {test_roc_auc_score}')

    return test_pred


def knn_model(X_train, X_test, y_train, y_test=None):

    knn = KNeighborsClassifier(n_neighbors=5, algorithm='brute',
                                metric='seuclidean', n_jobs=-1)
    knn.fit(X_train, y_train) # k neighbors learning
    test_pred = knn.predict(X_test) # prediction

    if y_test is not None:
        train_pred = knn.predict(X_train)
        # HOLD OUT PRECISION SCORE
        train_precision_score = precision_score(train_pred, y_train)
        test_precision_score = precision_score(test_pred, y_test)
        # HOLD OUT RECALL SCORE
        train_recall_score = recall_score(train_pred, y_train)
        test_recall_score = recall_score(test_pred, y_test)
        # HOLD OUT ROC AUC SCORE
        train_roc_auc_score = roc_auc_score(train_pred, y_train)
        test_roc_auc_score = roc_auc_score(test_pred, y_test)
        # RESULTS
        print(f'train_precision_score: {train_precision_score}')
        print(f'test_precision_score: {test_precision_score}')
        print(f'train_recall_score: {train_recall_score}')
        print(f'test_recall_score: {test_recall_score}')
        print(f'train_roc_auc_score: {train_roc_auc_score}')
        print(f'test_roc_auc_score: {test_roc_auc_score}')

    return test_pred


def get_features(train_file_name, test_file_name, scale=False):

    train = pd.read_csv(train_file_name)
    test = pd.read_csv(test_file_name)
    target = train['Survived']
    features = train.drop('Survived', axis=1)

    if scale:
        std = StandardScaler()
        features = std.fit_transform(features)
        test = std.transform(test)

    return features, test, target


features, test, target = get_features('new_train.csv', 'new_test.csv')

X_train, X_test, y_train, y_test = train_test_split(features, target, test_size=0.2, 
                                                    stratify=target, random_state=42, shuffle=True)

rf_parameters={'n_estimators': (620, 630, 640),
               'criterion': ('gini', 'entropy'),
               'max_depth': [14, 15, 16],
               'max_features': (2, 3, 4, 5),
               'min_samples_split': (8, 10, 12)}

gb_parameters={'learning_rate': (0.1, 0.05, 0.01),
                'subsample': (1, 0.5),
                'max_depth': (3, 4, 5),
                'max_features': (3, 4, 5),
                'n_estimators': (90, 100, 110),
                'n_iter_no_change': (3, 5, 10)}

svm_parameters={'C': (0.25, 0.5, 1, 2, 4)}
                # 'degree': (2, 3)}
                # 'gamma': ('scale', 'auto')}

knn_parameters={'n_neighbors': (2, 3, 4, 5, 6, 7, 8, 9, 10, 15, 21, 25, 31, 35, 41, 45),
                'metric': ('euclidean', 'seuclidean')}    

classifier = KNeighborsClassifier(algorithm='brute', n_jobs=-1)

knn_opt = hyperparameter_optimization(features, target, classifier,
                                  knn_parameters, metric=accuracy_score)

# log_reg_pred = log_reg_model(X_train, X_test, y_train, y_test)
log_reg_pred = log_reg_model(features, test, target)
save_solution(log_reg_pred, 'new')

# gb_pred = gradient_boosting_model(X_train, X_test, y_train, y_test)
gb_pred = gradient_boosting_model(features, test, target)
save_solution(gb_pred, 'gradient_boosting')

# svm_pred = support_vector_model(X_train, X_test, y_train, y_test)
svm_pred = support_vector_model(features, test, target)
save_solution(svm_pred, 'support_vector')

# knn_pred = knn_model(X_train, X_test, y_train, y_test)
knn_pred = knn_model(features, test, target)
save_solution(knn_pred, 'k_neighbors')

features, test, target = get_features('rf_train.csv', 'rf_test.csv')
# rf_pred = random_forest_model(X_train, X_test, y_train, y_test)
rf_pred = random_forest_model(features, test, target)
save_solution(rf_pred, 'random_forest_2')

print(model_differences(log_reg_pred, rf_pred)) # number of mismatched responses

ens = ensemble()
# ens = ensemble(log_reg_pred, gb_pred, rf_pred)
print(ens.head())
save_solution(ens['majority'], 'ensemble_1')
