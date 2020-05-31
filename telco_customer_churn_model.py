#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Apr 19 13:08:05 2020

@author: akashjalil
"""

import pandas as pd
import os
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns

# Retrieve the data
data = pd.read_csv('Telco_customer_churn.csv')

"""
We will first start with data exploration using the following key steps:
* Variable indentification
* Null detection
* Univariate analysis
* Multivariate analysis
* Outlier detection
* Feature engineering
"""

# Discover what each column is and its type
data.dtypes
describe = data.describe(include = 'all').T
data.info()

# Note that in the data there are empty strings ' ' for empty cells. We can convert these to nan
data = data.replace(' ',np.nan)

# Remove customerID
data = data.drop('customerID',axis = 1)

# How many nulls do we have in the dataset?
data.isnull().sum().sort_values(ascending = False)
data.isnull().sum().sort_values(ascending = False)*100/len(data)

"""
The only column will null values is TotalCharges
TotalCharges 11 0.16%

There are only 11 nulls in this column which is a tiny fraction of all the data.
We can either remove these rows or we can impute these values by using either mean, median, or mode

We shall impute this value but first it's best to understand the total charges column by it's distribution
before we replace the missing values
"""

# Distribution for TotalCharges
non_null_data = data.dropna()
non_null_data = non_null_data.apply(pd.to_numeric,errors = 'ignore')

sns.distplot(non_null_data['TotalCharges'])

"""
The distribution of TotalCharges is positively (right) skewed which indicates 
the mean is greater than the median. Taking the median would better serve as imputing the nulls.
count    7032.000000
mean     2283.300441
std      2266.771362
min        18.800000
25%       401.450000
50%      1397.475000
75%      3794.737500
max      8684.800000

How does the distribution fair with SeniorCitizen and gender?
"""

sns.catplot(x = 'gender', y = 'TotalCharges', hue = 'SeniorCitizen', kind = 'box', data = non_null_data)

"""
Across the two genders the value of Totalcharges is roughly the same.
But senior citizens tend to have higher totalcharges.
Note the 11 missing TotalCharges are from those who are not senior citizens, have depedents, and have a contract of only two years
This can be checked using: """

null_data = data[data.isnull().any(axis=1)]

"""
Our impute approach will be to take the median of TotalCharges for individuals 
similar to the missing TotalCharges.
Taking the median gives us 2659.2, this is what we will impute TotalCharges with
"""

median_fill_value = np.median(non_null_data[(non_null_data['SeniorCitizen']==0) & (non_null_data['Dependents']=='Yes') & (non_null_data['Contract']=='Two year')]['TotalCharges'])

data['TotalCharges'] = data['TotalCharges'].fillna(median_fill_value)
data = data.apply(pd.to_numeric,errors = 'ignore')


"""
Now that we have a complete dataset, let's continue with our exploring our data.
We'll try to understand what our dataset shows in relation to customer churning
"""

# Proportion of customers churning in the dataset
data['Churn'].value_counts(normalize = True)

"""
Our dataset is unbalanced with over 70% of customers not churning. This will be a factor when 
we start to build our model as an imbalanced dataset can be devious to some models
"""

"""
Let's start by exploring the numerical variables first as there are only three of these:
* tenure
* MonthlyCharges
* TotalCharges
"""
churn_customers, non_churn_customers = data[data['Churn']=='Yes'],data[data['Churn']!='Yes']

def numerical_plots(variable):
    plt.title(f'Distribution for {variable} by churn type')
    sns.distplot(churn_customers[variable], color = 'orange',label = 'Churned')
    sns.distplot(non_churn_customers[variable], color = 'blue',label = 'Not Churned')
    plt.legend()

# numerical_plots('tenure')
# numerical_plots('MonthlyCharges')
# numerical_plots('TotalCharges')

"""
The distribution of tenure is quite informative as it shows customers are most likely to churn in the first couple of months. 
For customers who have not churned the distribution is quite flat.

In the monthly charges distribution we see most of the disparity for churned customers around the 70-100 monthly charge and for
non churned customers the peak is in the left near the lowest value of 20. 

Finally, TotalCharges shows a similar distribution shape for churned and non churned customers. We saw that for churned customers the
peak of their tenure was in the early months which can explain why the peak for TotalCharges is at the lowest level. For non churned
the peak is not as towering however.

These three metrics, or more likely tenure and MonthlyCharges, may be quite important factors for predicting churn. 
"""

"""
How do these metrics compare with each other (multicollinearity) and how do they compare against churn output (point biserial correlation)
"""

sns.pairplot(data[['tenure','MonthlyCharges','TotalCharges','Churn']],hue = 'Churn')
corr = data[['tenure','MonthlyCharges','TotalCharges']].corr()
sns.heatmap(corr,annot = True)

"""
Theres a high correlation between TotalCharges and MonthlyCharges and tenure. Could it be that
TotalCharges is a calculated field using tenure and MonthlyCharges? We can check this quickly
comparing the two columns
"""

total_charges = data["TotalCharges"].values
calc_total_charges = np.array(data["MonthlyCharges"]*data["tenure"])
difference = total_charges - calc_total_charges

print("Average difference: {0}\nMedian difference: {1}\nCorrelation coef: {2}".format(np.mean(difference),np.median(difference),np.corrcoef(total_charges,calc_total_charges)[0][1]))

"""
Our difference between TotalCharges in the dataset and the product of tenure and Monthlycharges has a low mean and median.
As well as a high correlation. We will need to reduce this multicollinearity by reducing the number of features.

As totalcharges is highly correlated with tenue and monthlycharges. We will remove this.

Just before we do that, let's try to identify any erroneous data within these columns
"""

"""
Outlier detection
We will first see if there are any outliers using boxplots for the three numerical variables
"""

def boxplot(variable):
    plt.title(f"Boxplot for {variable}")
    sns.boxplot(x = "Churn", y = variable, data = data)
    plt.ylabel(f"Total {variable}")
    plt.xlabel("Has Churned?")

# boxplot('tenure')
# boxplot('MonthlyCharges')
# boxplot('TotalCharges')

"""
There may be some outliers in tenure and TotalCharges. Let's view a scatter plot of these
"""

def scatterplot(x_val,y_val):
    sns.scatterplot(x = x_val, y = y_val, hue = "Churn",data = data)
    plt.ylabel(y_val)
    plt.xlabel(x_val)

scatterplot("TotalCharges", "tenure")
scatterplot("TotalCharges", "MonthlyCharges")

"""
There are at least two outliers we can pin point in the TotalCharges vs MonthlyCharges scatterplot.
Upon inspecting the dataset, we can actually see there are a number of customers with 0 tenure
but with 2659.2 in total charges. This could be an data entry issue or data collection issue from the company.
"""

data[["tenure", "MonthlyCharges","Churn"]].sort_values(by = "tenure").iloc[:11]

removal_index = data[["tenure", "MonthlyCharges","Churn"]].sort_values(by = "tenure").iloc[:11].index

data = data[~data.index.isin(removal_index)].reset_index(drop = True)

data = data.drop("TotalCharges",axis = 1)



"""
The remaining variables in our data are all categorical. Churn is our dependent variable so we
shall quickly transform this into a binary variable
"""

data['Churn'] = data['Churn'].map({'Yes':1,'No':0})
data['SeniorCitizen'] = data['SeniorCitizen'].map({1:'Yes',0:'No'})

categorical_cols = data.select_dtypes(include = 'object').columns.tolist()

def categorical_count_plots(variable,total = len(data)):
    plt.title(f'Count of {variable} by churn type')
    ax = sns.countplot(x = variable,hue = 'Churn',data = data)
    for p in ax.patches:
        height = p.get_height()
        ax.text(p.get_x()+p.get_width()/2.,height + 3,'{:1.0f}%'.format(height*100/total),ha="center") 
    plt.ylabel('Number of customers')
    plt.legend()


# categorical_count_plots('SeniorCitizen')
# categorical_count_plots('gender')
# categorical_count_plots('Partner')
# categorical_count_plots('Dependents')
# categorical_count_plots('PhoneService')
# categorical_count_plots('SeniorCitizen')
# categorical_count_plots('MultipleLines')
# categorical_count_plots('InternetService')
# categorical_count_plots('OnlineSecurity')
# categorical_count_plots('OnlineBackup')
# categorical_count_plots('DeviceProtection')
# categorical_count_plots('TechSupport')
# categorical_count_plots('StreamingTV')
# categorical_count_plots('StreamingMovies')
# categorical_count_plots('Contract')
# categorical_count_plots('PaperlessBilling')
# categorical_count_plots('PaymentMethod')


"""
We have chosen 5 plots to analyse and the rest can be found in the appendix.
Firstly, we note
"""

"""
We will replace No internet service with No as it will reduce the number of factors in the following columns:
'OnlineSecurity', 
'OnlineBackup', 
'DeviceProtection',
'TechSupport',
'StreamingTV', '
StreamingMovies'
"""

data = data.replace({'No internet service': 'No'})


"""
Let's one-hot encode our categorical variables. The reason for one-hot encoding these variables
is because they are nominal variables and do not have a logical ordering hence why we are unable to
label them numerically
"""

encoded_data = pd.get_dummies(data,drop_first = True)
encoded_data = encoded_data.apply(pd.to_numeric)

"""
Now our data is almost ready to be used from trained but first let's explore correlations in our numeric data
"""

corr = encoded_data.corr()
correlation_threshold = 0.65

def cols_with_high_correlations(correlation_threshold):
    upper_triangle = corr.where(np.triu(np.ones(corr.shape), k=1).astype(np.bool))
    cols_to_drop = [col for col in upper_triangle.columns if any(upper_triangle[col].abs() > correlation_threshold)]

    return cols_to_drop

high_correlations = cols_with_high_correlations(correlation_threshold)

"""
'MultipleLines_No phone service' is perfectly negatively correlated with 'PhoneService_Yes' so we can drop column
"""

encoded_data = encoded_data.drop(high_correlations,axis = 1)


"""
We're almost ready to train our first model but first we need to have an even dataset for customers
who have churned and those who have not. 73% of customers have no churned so a model could predict 0
for all customers and gain a accuracy of 73%. To eliminate this we will balance the dataset using oversampling.
"""
from imblearn.over_sampling import RandomOverSampler

def oversample_data():

    X,y = encoded_data.drop('Churn',axis = 1), encoded_data[['Churn']]
    
    oversampler = RandomOverSampler(random_state = 1)
    oversampler.fit(X,y)
    
    X_oversampled, y_oversampled = oversampler.fit_sample(X,y)
    
    return X_oversampled, y_oversampled

X_oversampled, y_oversampled = oversample_data()

y_oversampled['Churn'].value_counts()


"""
Now that our dataset is balanced we can go ahead with training our first model which will be logistic regression
"""
from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix, classification_report
import statsmodels.api as sm

X_train, X_test, y_train, y_test = train_test_split(X_oversampled, y_oversampled, test_size=0.25, random_state=1)


def logistic_model(X_train, X_test, y_train, y_test):
    
    model = sm.Logit(y_train,X_train)
    results = model.fit()
    
    prediction = (results.predict(X_test) > 0.5).astype(int)
    confusion = confusion_matrix(y_test,prediction)
    report = classification_report(y_test,prediction,output_dict=True)
    return results, confusion, prediction, report

full_logit, full_logit_confusion,_,_, = logistic_model(X_train, X_test, y_train, y_test)

"""
To interpret the results of this logistic regression we'll take each column in turn
* Coef - This shows the estimated coefficient for each independent variable. This tells us the amount of increase
         in the predicted log odds for Churn = 1 if all other independent variables are fixed.
         Taking tenure for example for every 1 unit increase in tenure we expect a 0.05 decrease in the log odds
         of the dependent variable Churn

* Std err - This is the standard error for each independent variable which are used to test the hypothesis that the coef
            for a variable is 0.
            
* z-value - The z value is simply coef/std error. A simple interpretation is that if abs(z-value) > 2 then the variable can be
            deemed significant

* p>|z| - This is the 2-tailed p value used in testing the hypothesis that the coef for a given varible is 0. If we used a = 0.05 then
          any variable with p value < a will be statistically significant.
          
          
Thus we can say that 

tenure,                                  
MonthlyCharges,                                                     
SeniorCitizen_Yes,                         
PhoneService_Yes,                                     
OnlineSecurity_Yes,               
OnlineBackup_Yes,               
DeviceProtection_Yes,              
TechSupport_Yes,              
StreamingTV_Yes,              
Contract_One year,             
Contract_Two year
PaperlessBilling_Yes,                      
PaymentMethod_Electronic check      

are all significant in predicting churn.      
"""

significant_columns = ['tenure','MonthlyCharges','SeniorCitizen_Yes','PhoneService_Yes',                        
                       'OnlineSecurity_Yes',               
                       'OnlineBackup_Yes','DeviceProtection_Yes','TechSupport_Yes',              
                       'Contract_One year','Contract_Two year','PaperlessBilling_Yes','PaymentMethod_Electronic check']

new_logit, new_confusion,_,_ = logistic_model(X_train[significant_columns],X_test[significant_columns],y_train, y_test)


"""
Note that now we have InternetService_Fiber optic and InternetService_No as being not significant. Let's remove these
and rerun our model
"""

significant_columns = [x for x in significant_columns if x not in ['StreamingTV_Yes']]

new_logit, new_confusion, preds, report = logistic_model(X_train[significant_columns],X_test[significant_columns],y_train, y_test)
tn, fp, fn, tp = new_confusion.ravel()

"""
Confusion has predicted classes as going across
and true classes going down on the left side
"""

"""
We now have a logistic regression model that has all significant variables and our metrics are:
Accuracy = 76%
Precision = 75%
Recall = 79%
F1 Score = 79%

False negative = 21%
False positive = 26%
"""

"""
Let's move on to a different type of model... Random Forests.

This is an ensemble mode, ie. it is made up of smaller models (Decisions Trees) in which the 
final classification result is the mode (in classification problems) or mean (in regression problems)
of the smaller models. Random Forests reduce the overfitting nature of Decision Trees 
"""

from sklearn.ensemble import RandomForestClassifier

def random_forest(X_train,X_test,y_train, y_test):
    
    model = RandomForestClassifier()
    results = model.fit(X_train,y_train.values.ravel())
    
    prediction = (results.predict(X_test) > 0.5).astype(int)
    confusion = confusion_matrix(y_test,prediction)
    report = classification_report(y_test,prediction,output_dict=True)
    
    tn, fp, fn, tp = confusion.ravel()
    false_negative = fn/(fn+tp)
    false_positive = fp/(fp+tn)
    
    return results, confusion, prediction, report, false_negative, false_positive

rf1, rf1_confusion, rf1_preds, rf1_report,rf1_fn,rf1_fp = random_forest(X_train,X_test,y_train, y_test)


"""
Our random forest model trained on all features as the following:
Accuracy = 89%
Precision = 85%
Recall = 95%
F1 Score = 90%

False negative = 5%
False positive = 16%
"""

"""
To extract the important features we'll use permutation importance.
This method takes each feature, shuffles the data, creates a prediction using the shuffled data,
and calculates the score (accuracy in this case).

So, if there is a decrease in the score then the feature is important as it is not random.
"""
from sklearn.inspection import permutation_importance

def feature_importance_func(model, X_test, y_test):
    perm = permutation_importance(rf1, X_test, y_test,n_repeats=30,random_state=1)
    
    perm_sorted = perm.importances_mean.argsort()[::-1]
    
    important_dict = {}
    for i in perm_sorted:
        feature, mean, std = X_train.columns[i], perm.importances_mean[i], perm.importances_std[i]
        if mean - 2*std > 0:
            important_dict[feature] = [mean,std]
        
    importance_df = pd.DataFrame.from_dict(important_dict, orient='index',columns = ['Mean','STD'])
    
    return importance_df

importance_df = feature_importance_func(rf1, X_test, y_test)

top_10_features = importance_df.index[0:10]

rf2, rf2_confusion, rf2_preds, rf2_report,rf2_fn,rf2_fp = random_forest(X_train[top_10_features],X_test[top_10_features],y_train, y_test)

"""
We calculate the importance for each feature and note that tenure and monthly charges are the most
important features of our dataset.

To reduce the number of features in our dataset, we'll pick the top 10 as to reduce our features by 50%
Our accuracy metrics seem unaffected by reducing our features, the features we removed thus were not
contributing to anything for our model.
"""

"""
From here on out we will use the top 10 features only in our dataset and run cross validations
for the remaining models we train. At the end we should have a dataframe filled with accuracy metrics
for each model.
"""

X_train_imp, X_test_imp = X_train[top_10_features], X_test[top_10_features]

from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.neighbors import KNeighborsClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.model_selection import cross_validate

models = [LogisticRegression(random_state=1), RandomForestClassifier(random_state=1),SVC(probability = True,random_state = 1),KNeighborsClassifier(random_state=1), GaussianNB(random_state=1)]

clf = SVC(probability = True, random_state = 1)
scoring = ['accuracy','precision','recall','f1','roc_auc']
scores = cross_validate(clf,X_oversampled[top_10_features],y_oversampled.values.ravel(),cv = 5, scoring = scoring)
m = {metric: round(np.mean(score),5) for metric,score in scores.items()}


def cross_validation_models():
    scoring = ['accuracy','precision','recall','f1','roc_auc']
    test_accuracy,test_precision, test_recall, test_f1,test_auc = [],[],[],[],[]
    for model in models:
        scores = cross_validate(model,X_oversampled[top_10_features],y_oversampled.values.ravel(),cv = 5, scoring = scoring)
        m = {metric: round(np.mean(score),5) for metric,score in scores.items()}
        test_accuracy.append(m['test_accuracy'])
        test_precision.append(m['test_precision'])
        test_recall.append(m['test_recall'])
        test_f1.append(m['test_f1'])
        test_auc.append(m['test_roc_auc'])
        
    model_names = ['LR','RF','SVM','KNN','GNB']
    
    return pd.DataFrame({'model_name': model_names,
                         'accuracy': test_accuracy,
                         'precision': test_precision,
                         'recall': test_recall,
                         'f1': test_f1,
                         'auc': test_auc})

cv_df = cross_validation_models()

"""
We see that our random forest model greatly exceeds the accuracy, precision, recall, and f1 score
of all the other models.

The random forest model will be used as the model of choice for predicting churn. However, we'll see
if we can fine tune the hyper parameters to get an even better random forest model.
"""

"""
A RF model contains the following hyperparameters:
* max_depth - Length of longest path between root and leaf nodes
* min_samples_split - Minimum number of observations in any node required to then split
* max_leaf_nodes - Maximum number of terminal nodes allowed
* min_samples_leaf - Minimum number of samples required in the leaf node. If this is not the case, the root node becomes the leaf node for the given terminal node
* n_estimators - Number of decision trees to use in the random forest. This will affect computation time
* max_features - Number of maximum features provided to each tree for splitting
"""
current_best_rf = models[1]

from sklearn.model_selection import GridSearchCV


def Evaluate_each_parameter(parameter, range_of_values):
    gs = GridSearchCV(current_best_rf,param_grid = {parameter: range_of_values},cv= 5,return_train_score = True,scoring = 'recall')
    grid_search = gs.fit(X_train[top_10_features],y_train.values.ravel())
    
    x = f'param_{parameter}'
        
    grid_search_df = pd.DataFrame(grid_search.cv_results_)[[x,'mean_train_score','mean_test_score']].apply(pd.to_numeric)
    grid_search_df = grid_search_df.set_index(x)
    
    return grid_search_df

parameter_dict = {'max_depth': np.arange(1,30,1),
                  'min_samples_split': np.arange(2,150,1),
                  'min_samples_leaf': np.arange(1,50,1),
                  'n_estimators': np.arange(1,100,1),
                  'max_features': np.arange(1,10,1)}

evaluated_parameter_dfs = [Evaluate_each_parameter(parameter, param_range) for parameter, param_range in dict.items(parameter_dict)]

for i, df in enumerate(evaluated_parameter_dfs):
    ax = df.plot(legend = True)
    ax.set_title(df.index.name)
    ax.set_ylabel('')
    ax.set_xlabel('')
    
"""
max_depth - The graph shows after a depth of 15, the model tends to over fit so we know the optimal
            depth of each decision should be between 10 and 15
min_samples_split - The graph decreases when the value for min_samples_split increases and tails off at around 40 or more.
                    The default value used when RandomForestClassifier is called seems the best with highest training and test accuracy
min_samples_leaf - Similar to min_samples_split the default value of min_samples_leaf of 1 seems best with it having the highest training and test score
n_estimators - As the number of trees increases, the model overfits and the score tails off. We can see that this happens after around 20 trees are used.
max_features - The number of features used to split a node does not seem to affect the training accuracy but splitting with 3 nodes tends to give a higher training accuracy
               As we only have 10 features, this hyperparameter seems to not affect the model so we'll keep it at default
"""

"""
Lets now utilise a few other features and look at what the best combination of hyperparameters we can come up with that gives the highest accuracy
These will be:
bootstrap - [True, False] to indicate whether to bootstrap the data or use all of it
criterion - ['gini','entropy'] this will indicate whether to use gini or entropy to measure the quality of the split at each node
"""

"""
Notice that we've changed some of the hyperparameters based on what we saw from the graphs above to give us the best value of the parameter relative to the accuracy of the model
"""
parameter_dict_grid = {'max_depth': np.arange(10,15,1),
                  'n_estimators': np.arange(10,20,1),
                  'bootstrap': [True, False],
                  'criterion': ['gini','entropy']
                  }

grid_search_forest = GridSearchCV(current_best_rf, parameter_dict_grid, cv=10, return_train_score = True, scoring='pre')
grid_search_forest.fit(X_train[top_10_features],y_train.values.ravel())

best_model = grid_search_forest.best_estimator_
best_params = grid_search_forest.best_params_

def best_model_review(model):
    prediction = (model.predict(X_test[top_10_features]) > 0.5).astype(int)
    confusion = confusion_matrix(y_test,prediction)
    report = classification_report(y_test,prediction)
    
    tn, fp, fn, tp = confusion.ravel()
    false_negative = fn/(fn+tp)
    false_positive = fp/(fp+tn)
    
    return confusion, prediction, report, false_negative, false_positive

best_model_confusion, best_model_predictions, best_model_report, best_model_fn, best_model_tn = best_model_review(best_model)

"""
The best model is a random forest model which has the following parameters adjusted:
bootstrap: False
criterion: 'gini', 
max_depth: 14
n_estimators: 17
"""

"""
The best features of our model remain the same with the top 5 being
"""

importance_plot = sns.barplot(x = 'index',y = 'Mean',data = importance_df.iloc[0:5,].reset_index())
importance_plot.set_xticklabels(importance_plot.get_xticklabels(), rotation=45, horizontalalignment='right')
importance_plot.set_ylabel('Importance')
importance_plot.set_title('Top 5 features of model')














