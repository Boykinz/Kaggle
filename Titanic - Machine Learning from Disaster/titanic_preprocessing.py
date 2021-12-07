import pandas as pd
import math


def rename(name):
    # new name categories
    for item in ['Mr.', 'Miss.', 'Mrs.']:
        if item in name:
            return item
    return 'Unknown'


def pclass_encoder(x):

    if x == 3:
        return 0.2424 # count(survived=1)/count(all_с3) where pclass = 3
    elif x == 2:
        return 0.4728 # count(survived=1)/count(all_с2) where pclass = 2
    else:
        return 0.6296 # count(survived=1)/count(all_с1) where pclass = 1


def name_encoder(x):

    if x == 'Mr.':
        return 0.1567 # count(mr_survived=1)/count(all_mr)
    elif x == 'Miss.':
        return 0.6978 # count(miss_survived=1)/count(all_miss)
    elif x == 'Mrs.':
        return 0.7920 # count(mrs_survived=1)/count(all_mrs)
    else:
        return 0.5224 # count(unknown_survived=1)/count(all_unknown)


def titanic_preprocessing(data):

    # used for log reg, gradient boosting, KNN and SVM

    data['Name'] = data['Name'].apply(rename)
    data['Sex'] = data['Sex'].apply(lambda x: 1 if x=='female' else -1)
    data['Fare'] = data.Fare.apply(lambda x: math.log2(x+1))
    data['Cabin'] = data['Cabin'].isna().apply(int)
    data['Embarked'] = data.Embarked.fillna('S')
    data['Fare'] = data['Fare'].fillna(data.Fare.median())
    data['Is_Alone'] = (data.Parch + data.SibSp == 0).apply(int) # loneliness feature
    data['Family'] = data.Parch + data.SibSp # Family feature
    data_name = pd.get_dummies(data['Name']) # Name categories one hot encoding
    data = pd.concat([data, data_name], axis=1)
    data_emb = pd.get_dummies(data['Embarked'], prefix='Emb') # Embarked categories one hot encoding
    data = pd.concat([data, data_emb], axis=1)
    data_pcl = pd.get_dummies(data['Pclass'], prefix='Pclass') # Pclass categories one hot encoding
    data = pd.concat([data, data_pcl], axis=1)

    # filling the age with the median in this class
    mask1 = (data['Pclass'] == 1) & data['Age'].isnull()
    mask2 = (data['Pclass'] == 2) & data['Age'].isnull()
    mask3 = (data['Pclass'] == 3) & data['Age'].isnull()
    data.loc[mask1, 'Age'] = data[data['Pclass'] == 1]['Age'].median()
    data.loc[mask2, 'Age'] = data[data['Pclass'] == 2]['Age'].median()
    data.loc[mask3, 'Age'] = data[data['Pclass'] == 3]['Age'].median()

    data.drop(columns=['PassengerId', 'Ticket', 'Parch',
                        'SibSp', 'Embarked', 'Pclass', 'Name'], inplace=True)
    print(data.head())

    return data


def titanic_preprocessing_2(data):

    # used for random forest

    data['Name'] = data['Name'].apply(rename)
    data['Name'] = data['Name'].apply(name_encoder)
    data['Sex'] = data['Sex'].apply(lambda x: 1 if x=='female' else -1)
    data['Fare'] = data.Fare.apply(lambda x: math.log2(x+1))
    data['Cabin'] = data['Cabin'].isna().apply(int)
    data['Embarked'] = data.Embarked.fillna('S')
    data['Fare'] = data['Fare'].fillna(data.Fare.median())
    data['Is_Alone'] = (data.Parch + data.SibSp == 0).apply(int) # loneliness feature
    data['Family'] = data.Parch + data.SibSp # Family feature
    # data['Ticket_Frequency'] = data.groupby('Ticket')['Ticket'].transform('count')
    data_emb = pd.get_dummies(data['Embarked'], prefix='Emb') # Embarked categories one hot encoding
    data = pd.concat([data, data_emb], axis=1)

    # filling the age with the median in this class
    mask1 = (data['Pclass'] == 1) & data['Age'].isnull()
    mask2 = (data['Pclass'] == 2) & data['Age'].isnull()
    mask3 = (data['Pclass'] == 3) & data['Age'].isnull()
    data.loc[mask1, 'Age'] = data[data['Pclass'] == 1]['Age'].median()
    data.loc[mask2, 'Age'] = data[data['Pclass'] == 2]['Age'].median()
    data.loc[mask3, 'Age'] = data[data['Pclass'] == 3]['Age'].median()
    
    data['Age'] = data['Age'].apply(lambda x: 10/x)
    data['Pclass'] = data['Pclass'].apply(pclass_encoder)

    data.drop(columns=['PassengerId', 'Ticket', 'Parch',
                        'SibSp', 'Embarked', 'Emb_S'], inplace=True)
    print(data.head())
    
    return data


def save_new_data(train, test, preprocessing_func, prefix=''):
    new_train = preprocessing_func(train)
    new_test = preprocessing_func(test)
    new_train.to_csv(prefix + '_train.csv', index=False)
    new_test.to_csv(prefix + '_test.csv', index=False)
    pass


train = pd.read_csv('train.csv')
test = pd.read_csv('test.csv')

save_new_data(train, test, titanic_preprocessing, prefix='new')
save_new_data(train, test, titanic_preprocessing_2, prefix='rf')

# x1 = train[train['Embarked'] == 'S']['Embarked'].count()
# x2 = train[train['Embarked'] == 'C']['Embarked'].count()
# x3 = train[train['Embarked'] == 'Q']['Embarked'].count()
# print(x1, x2, x3)

# t11 = train[(train['Pclass'] == 1) & (train['Survived'] == 1)]['Survived'].count()
# t10 = train[(train['Pclass'] == 1) & (train['Survived'] == 0)]['Survived'].count()
# p1 = t11/(t11+t10)

# t21 = train[(train['Pclass'] == 2) & (train['Survived'] == 1)]['Survived'].count()
# t20 = train[(train['Pclass'] == 2) & (train['Survived'] == 0)]['Survived'].count()
# p2 = t21/(t21+t20)

# t31 = train[(train['Pclass'] == 3) & (train['Survived'] == 1)]['Survived'].count()
# t30 = train[(train['Pclass'] == 3) & (train['Survived'] == 0)]['Survived'].count()
# p3 = t31/(t31+t30)

# print(p1, p2, p3)
