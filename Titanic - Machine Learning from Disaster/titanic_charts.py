import pandas as pd
import math
import matplotlib.pyplot as plt
import seaborn as sns
from matplotlib.colors import ListedColormap


def rename(name):
    # new name categories
    for item in ['Mr.', 'Miss.', 'Mrs.']:
        if item in name:
            return item
    return 'Unknown'


train = pd.read_csv('train.csv')
train['Name'] = train['Name'].apply(rename)
name_groups = train.groupby('Name').groups
print(train.head())


def hist_plot(axes, names, colors):

    for ax, name, color in zip(axes.flatten(), names, colors):
        ax.hist(train['Survived'][name_groups[name]], color=color)
        ax.set_title(name + ' survived')
        ax.set_xlabel('is survived')
        ax.set_ylabel('number')
        ax.grid()


fig1, ax1 = plt.subplots(nrows=2, ncols=2, figsize=(8, 6))
hist_plot(ax1, ['Mr.', 'Miss.', 'Mrs.', 'Unknown'],
              ['red', 'green', 'blue', 'black'])
fig1.tight_layout()


fig2, ((distribution1, distribution2), (ax2, ax3)) = plt.subplots(nrows=2, ncols=2, figsize=(8, 6))
train.groupby('Survived').Fare.plot.kde(ax=distribution1, color={'0':'orange', '1':'black'})
distribution1.set_title('Fare distributions')
distribution1.set_xlabel('Fare')
distribution1.legend()
distribution1.grid()

train.groupby('Survived').Age.plot.kde(ax=distribution2, color={'0':'orange', '1':'black'})
distribution2.set_title('Age distributions')
distribution2.set_xlabel('Age')
distribution2.legend()
distribution2.grid()

sex = train.groupby('Sex').groups
ax2.hist(train['Survived'][sex['male']], color='green')
ax2.set_title('Male')
ax2.set_xlabel('is survived')
ax2.set_ylabel('number')
ax2.grid()

ax3.hist(train['Survived'][sex['female']], color='purple')
ax3.set_title('Female')
ax3.set_xlabel('is survived')
ax3.set_ylabel('number')
ax3.grid()
fig2.tight_layout()


fig3, ax3 = plt.subplots(figsize=(8, 6))
colors = ['grey','orange']
ax3.scatter(train['Fare'], train['Age'],  c=train['Survived'],
            cmap=ListedColormap(colors), marker=(5,1))
ax3.set_title('Age-Fare scatter plot')
ax3.set_xlabel('Fare')
ax3.set_ylabel('Age')
ax3.grid()
fig3.tight_layout()


fig4, ax4 = plt.subplots(nrows=1, ncols=2, figsize=(8, 4))
ax4[0].hist(train.Fare.apply(lambda x: math.log10(x+1)), bins=20, label='Fare', color='red')
ax4[0].legend()
ax4[0].set_title('Fare distribution')
ax4[0].grid()
ax4[1].hist(train.Age-train.Age.mean(), bins=20, label='Age', color='green')
ax4[1].legend()
ax4[1].set_title('Age distribution')
ax4[1].grid()
fig4.tight_layout()


fig5, ax5 = plt.subplots(nrows=2, ncols=2, figsize=(8, 6))
sns.countplot('Survived', hue='Embarked', data=train, ax=ax5[0, 0])
ax5[0, 0].set_title('Embarked Countplot')
sns.countplot('Survived', hue='Pclass', data=train, ax=ax5[0, 1])
ax5[0, 1].set_title('Pclass Countplot')
corr_train = train.drop('PassengerId', axis=1).corr()
sns.heatmap(corr_train, ax=ax5[1, 0])
ax5[1, 0].set_title('Features correlation')
sns.countplot('Survived', data=train, ax=ax5[1, 1])
ax5[1, 1].set_title('Class Balance')
fig5.tight_layout()


plt.figure(figsize=(6,6))
sns.heatmap(train.isnull(), cbar=False).set_title('Diagram of missing values')
plt.tight_layout()


train['Sex'] = train['Sex'].apply(lambda x: 1 if x=='female' else -1)
sns.pairplot(train[['Age', 'Fare', 'Pclass', 'Sex', 'Survived']], hue='Survived',
             diag_kind='hist', height=1.5)
plt.tight_layout()
plt.show()
