#!/usr/bin/env python
# coding: utf-8

# # Regression Project

# In this project, your goal is to build regression models of housing prices. The models should learn from data and be able to predict the median house price in a district (which is a population of 600 to 3000 people), given some predictor variables. 

# # Mit welchem RMSE wäre ich zufrieden?

# # Setup

# ### Import von benötigten Libraries

# In[1]:


get_ipython().run_line_magic('matplotlib', 'inline')

# Allgemein
import pandas as pd
import numpy as np
import seaborn as sns 
import matplotlib.pyplot as plt

from statsmodels.compat import lzip
import statsmodels.api as sm
from statsmodels.formula.api import ols
from statsmodels.stats.outliers_influence import variance_inflation_factor

import statsmodels.formula.api as smf
from statsmodels.tools.eval_measures import mse, rmse

# Einstellung für Visualisierungen
sns.set_theme(style="ticks", color_codes=True)


# Für sklearn pipeline
from sklearn.compose import ColumnTransformer
from sklearn.compose import make_column_selector as selector
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn import set_config
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.linear_model import Lasso
from sklearn.linear_model import LassoCV

from sklearn.model_selection import train_test_split

from sklearn.linear_model import LinearRegression
from sklearn.metrics import r2_score
from sklearn.metrics import mean_squared_error


import plotly.express as px

from patsy import dmatrix

from sklearn.preprocessing import SplineTransformer

import statsmodels.api as sm


# ### Erstellen der Pipeline für scikit learn Modelle

# In[2]:


# for numeric features
#Normalisierung (X=(X-Mittelwert) / Standardabweichung)
numeric_transformer = Pipeline(steps=[
    ('imputer', SimpleImputer(strategy='median')),
    ('scaler', StandardScaler())
    ])


# In[3]:


# for categorical features  
categorical_transformer = Pipeline(steps=[
    ('imputer', SimpleImputer(strategy='constant', fill_value='missing')), #Konstante bei fehlenden Werten reinschreiben
    ('onehot', OneHotEncoder(handle_unknown='ignore'))#pro Featureausprägung eine Spalte, zutreffendes Feature hat dann Wert 1 und die anderen Wert 0
    ])


# In[4]:


# Pipeline
preprocessor = ColumnTransformer(transformers=[
    ('num', numeric_transformer, selector(dtype_exclude="category")),
    ('cat', categorical_transformer, selector(dtype_include="category"))
        ])


# # Import Data

# Als nächstes wird der Datensatz aus dem GutHub Repository importiert

# In[5]:


ROOT = "https://raw.githubusercontent.com/jan-kirenz/project-OliScha/main/"
DATA = "project_data.csv?token=GHSAT0AAAAAABPCEITIYHBIEPRTFMZJXUGKYPKREJQ"

df = pd.read_csv(ROOT + DATA)


# In[6]:


# Prüfen ob Import funktioniert hat und ersten Blick auf Daten werfen
df


# # Data Split

# The error rate on new cases is called the generalization error (or out-of-sample error), and by evaluating our model on the test set, we get an estimate of this error. This value tells you how well your model will perform on instances it has never seen before

# *Im nächsten Schritt wird das df in Trainingsdaten und Testdaten aufgeteilt. Durch random state = 0 sind die Datensätze jederzeit replizierbar.*

# In[7]:


# Data Split
train_dataset = df.sample(frac=0.8, random_state=0)
test_dataset = df.drop(train_dataset.index)

train_dataset


# # Data Inspection 

# In[8]:


# Datentypen und fehlende Werte prüfen
train_dataset.info()


# In[9]:


# Identifizieren der NULL Werte via Heatmap
sns.heatmap(train_dataset.isnull(), 
            yticklabels=False,
            cbar=False, 
            cmap='viridis');


# In[10]:


# Identifizieren der NULL Werte via Liste
print(train_dataset.isnull().sum())


# In den vorherigen Schritten kann erkannt werden, das für das Attribut total_bedrooms 207 Null Values vorliegen. alle anderen Attribute sind vollständig

# *Anhand der hier abgebildeten Untersuchung kann erkannt werden, dass die Variable "total_bedrooms" in 207 Observations leer ist. Die hier entdeckten Anomalien werden später unter Feature Engineering entfernt.*

# # Deskriptive Statistik

# In[11]:


# summary statistics for all numerical columns
round(df.describe(),1).transpose()


# Die Zeilen mit 25 %, 50 % und 75 % zeigen die entsprechenden Perzentile: EinPerzentil besagt, dass ein bestimmter prozentualer Anteil der Beobachtungenunterhalb eines Werts liegt. Beispielsweise haben 25 % der Bezirke ein housing_median_age unter 18, 50 % liegen unter 29, und 75 % liegen unter 37. Diese nenntman oft das 25. Perzentil (oder 1. Quartil), den Median und das 75. Perzentil(oder 3. Quartil)

# In[12]:


sns.displot(data=df, x="median_house_value", kind="kde" )


# In[112]:



df.hist(bins=70, figsize=(20,15))
plt.show()


# # Exploratory Data Analysis

# In[134]:


#nimmt nur numerische variablen
sns.pairplot(data=train_dataset, hue="ocean_proximity");


# ***Erkenntnis:** Ein Zusammenhang zwischen Income und house value ist zu erkennen, ansonstn keine offensichtlichen Zusammenhänge.  
# Lineare Zusammenhänge zwischen rooms, bedrooms, households und population sind logisch und müssen bei der Modellierung später beachtet werden (Collinearity).*

# In[68]:


corr = train_dataset.corr()
corr['median_house_value'].sort_values(ascending=False)


# ---

# #### **EDA - Analyse kategorialer Varaiblen**

# *Als nächstes werden die kategorialen Variablen näher untersucht. Im Datensatz gibt es lediglich zwei kategoriale Variablen, "ocean_proximity" und "price_category". Da sich die Variable "price_category" direkt von der vorherzusagenden Variablen "median_house_value" ableitet, ist diese nicht zur Vorhersage geeignet und wird daher nicht weiter beachtet.*

# In[69]:


train_dataset['ocean_proximity'].value_counts()


# In[115]:


# Verteilung von ocean_proximity auf Geokoordinaten visualisieren
sns.jointplot(data=train_dataset, x='longitude', y='latitude', hue="ocean_proximity",height=10);


# In[121]:


# Visualisierung Dichte
sns.jointplot(data=train_dataset, x='longitude', y='latitude', hue="ocean_proximity",height=10, alpha=0.2 );


# In[133]:


train_dataset.plot(kind="scatter", x="longitude", y="latitude", alpha=0.4, figsize=(10,7),c="median_house_value", cmap=plt.get_cmap("jet"), colorbar=True,)
plt.legend()


# *Auf der Karte (Kalifornien) lässt sich eine klare Verteilung der Variablen "ocean_proximity" erkennen. Die Ausprägungen "Island" und "Near Bay" liegen nur an jeweils einer bestimmten Stelle auf der Karte vor.*

# In[71]:


# Untersuchung der kategroialen Variable "ocean_proximity" mit einem erweiterten Boxplot
sns.boxenplot(data=train_dataset, x="ocean_proximity", y="median_house_value");


# *Im Boxplot Diagramm lassen sich bereits einige Outlier erkennen, welche sich am unteren Wertebereich von "median_house_value" befinden.*

# In[72]:


# Ergänzung zum Boxplot um Menge und Dichte der Observations zu visualisieren
sns.stripplot(data=train_dataset, x="ocean_proximity", y="median_house_value" , size=1 );


# In[73]:


# Analyse von "ocean_proximity" mit displot
sns.displot(data=train_dataset, x="median_house_value", hue = "ocean_proximity", kind="kde" )


# >***Fazit:** "ocean_proximity" hat definitiv Einfluss auf unsere vorherzusagende Variable. Inland ist bspw. tenedneziell günstiger als die anderen Ausprägungen.   
# Auffällig ist, das es für die Ausprägung Island nur sehr wenige Observations gibt und Island und Near Bay stark auf lokale Gebiete beschränkt sind (siehe Karte).*

# ---

# #### **EDA - Analyse numerischer Varaiblen**

# *Vor allem die numerische Variable "median_income" sieht auf Basis des zu Beginn erstellen pairplots vielversprechend aus und wird daher nun näher untersucht. Aber auch andere numerische Varaiblen sollen noch genauer betrachtet werden.*

# In[74]:


# Grouped summary statistics for all numerical columns (in transposed view)
train_dataset.groupby(["ocean_proximity"]).describe().T


# In[75]:


round(train_dataset.describe(),2).T


# In[76]:


# Analyse der Variablen "median_income"
sns.jointplot(data=train_dataset, x='median_income', y='median_house_value', hue="ocean_proximity", );


# In[77]:


sns.lmplot(data=train_dataset, x='median_income', y='median_house_value');


# **Analyse "household_per_person"**

# In[78]:


sns.scatterplot(data=train_dataset, x='household_per_person', y='median_house_value')


# In[79]:


sns.lmplot(data=train_dataset, x='household_per_person', y='median_house_value');


# **Analyse "rooms_per_household"**

# In[80]:


sns.scatterplot(data=train_dataset, x='rooms_per_household', y='median_house_value')


# In[81]:


sns.lmplot(data=train_dataset, x='rooms_per_household', y='median_house_value');


# **Analyse "housing_median_age"**

# In[82]:


sns.scatterplot(data=train_dataset, x='housing_median_age', y='median_house_value')


# In[83]:


sns.lmplot(data=train_dataset, x='housing_median_age', y='median_house_value');


# #### **Correlation**

# In[84]:


# Create correlation matrix for numerical variables
corr_matrix = train_dataset.corr()
corr_matrix


# In[85]:


# Erstellen einer Heatmap um Abhängigkeiten zwischen den verschiedenen Variablen zu visualisieren

# Use a mask to plot only part of a matrix
mask = np.zeros_like(corr_matrix)
mask[np.triu_indices_from(mask)]= True

# Erstellen der Heatmap mit zusätzlichen Parametern
plt.subplots(figsize=(11, 15))
heatmap = sns.heatmap(corr_matrix, 
                      mask = mask, 
                      square = True, 
                      linewidths = .5,
                      cmap = 'coolwarm',
                      cbar_kws = {'shrink': .6,
                                'ticks' : [-1, -.5, 0, 0.5, 1]},
                      vmin = -1,
                      vmax = 1,
                      annot = True,
                      annot_kws = {"size": 10})


# # Initial Feature Engineering

# In[149]:


# create new, more relevant variables
train_dataset=train_dataset.assign(people_per_household=lambda train_dataset: train_dataset.population/train_dataset.households)
train_dataset=train_dataset.assign(household_per_person=lambda train_dataset: train_dataset.households/train_dataset.population)
train_dataset=train_dataset.assign(bedrooms_per_household=lambda train_dataset: train_dataset.total_bedrooms/train_dataset.households)
train_dataset=train_dataset.assign(rooms_per_household=lambda train_dataset: train_dataset.total_rooms/train_dataset.households)
train_dataset=train_dataset.assign(bedrooms_per_room=lambda train_dataset: train_dataset.total_bedrooms/train_dataset.total_rooms)


# In[150]:


# create new, more relevant variables
test_dataset=test_dataset.assign(people_per_household=lambda test_dataset: test_dataset.population/test_dataset.households)
test_dataset=test_dataset.assign(household_per_person=lambda test_dataset: test_dataset.households/test_dataset.population)
test_dataset=test_dataset.assign(bedrooms_per_household=lambda test_dataset: test_dataset.total_bedrooms/test_dataset.households)
test_dataset=test_dataset.assign(rooms_per_household=lambda test_dataset: test_dataset.total_rooms/test_dataset.households)
test_dataset=test_dataset.assign(bedrooms_per_room=lambda test_dataset: test_dataset.total_bedrooms/test_dataset.total_rooms)


# In[151]:


# drop remaining row with one missing value
train_dataset = train_dataset.dropna()


# In[152]:


#drop population outlier
train_dataset=train_dataset.drop([15360, 9880])
#drop people_per_household outlier
train_dataset=train_dataset.drop([19006, 16669, 13034, 3364, 9172, 12104, 16420])


# In[153]:


# change datatype zu string um str.replace transformation durchzuführen
train_dataset['median_house_value'] = train_dataset['median_house_value'].astype("string")
train_dataset['housing_median_age'] = train_dataset['housing_median_age'].astype("string")


# In[154]:


# Bereinigung der "fehlerhaften" Werte
train_dataset.median_house_value = train_dataset.median_house_value.str.replace("$", "", regex =True)
train_dataset.housing_median_age = train_dataset.housing_median_age.str.replace("years", "", regex =True)


# In[155]:


# data type anpassen
# value oder age könnte auch float sein, hier aber für int entschieden da keine Nachkommastellen vorhanden 
# ocean proximity und price category bleiben object, da categroical values ohne order, mit meaning und nur string
train_dataset['median_house_value'] = train_dataset['median_house_value'].astype("float64")
train_dataset['housing_median_age'] = train_dataset['housing_median_age'].astype("float64")
train_dataset['total_bedrooms'] = train_dataset['total_bedrooms'].astype("int64")
train_dataset['ocean_proximity'] = train_dataset['ocean_proximity'].astype("category")
train_dataset['price_category'] = train_dataset['price_category'].astype("category")


# In[156]:


# summary statistics for all categorical/object columns
train_dataset.describe(include=['category']).transpose()


# In[157]:


corr = train_dataset.corr()
corr['median_house_value'].sort_values(ascending=False)


# # Modelling

# ## 1. Linear OLS Regression

# In[158]:


# Select features for simple regression
features = ['median_income', 'household_per_person', 'ocean_proximity', 'rooms_per_household', 'housing_median_age']
X1 = train_dataset[features]

# Create response
y1 = train_dataset["median_house_value"]


# ### Model 1.1 - Linear OLS Regression with sklearn

# In[159]:



# Data Split für Modell Scikitlearn
X_train1, X_test1, y_train1, y_test1 = train_test_split(X1, y1, test_size=0.2, random_state=42)


# In[160]:


# Create pipeline with model
lin_pipe = Pipeline(steps=[
    ('preprocessor', preprocessor),
    ('lin', LinearRegression())
                        ])


# In[161]:


# show pipeline
set_config(display="diagram")
# Fit model
lin_pipe.fit(X_train1, y_train1)


# In[162]:


# Obtain model coefficients
lin_pipe.named_steps['lin'].coef_


# In[163]:


# get absolute values of coefficients
importance = np.abs(lin_pipe.named_steps['lin'].coef_)

sns.barplot(x=importance, 
            y=list_numerical);


# #### Evaluation with Test Data

# In[ ]:


y_pred1 = lin_pipe.predict(X_test1)


# In[ ]:


r2_score(y_test1, y_pred1)


# #### Evaluate with Training data

# In[ ]:


y_pred1train = lin_pipe.predict(X_train1)


# In[ ]:


y_pred1train


# In[ ]:


r2_score(y_train1, y_pred1train)  


# Ergebnis mit Top 4 features ['median_income', 'household_per_person', 'ocean_proximity', 'room_per_person'] : r2 = 0.6514777282231453
# + housing_median_age = 0.660916387867293

# In[ ]:


mean_squared_error(y_train1, y_pred1train)


# In[ ]:


# RMSE
mean_squared_error(y_train1, y_pred1train, squared=False)


# In[ ]:


sns.residplot(x=y_pred1train, y=y_train1, scatter_kws={"s": 80});


# #### Evaluation with Test Data

# In[ ]:


y_pred1test = lin_pipe.predict(X_test1)


# In[ ]:


print('MSE:', mean_squared_error(y_test1, y_pred1test))

print('RMSE:', mean_squared_error(y_test1, y_pred1test, squared=False))


# In[ ]:



# Plot with Plotly Express
px.scatter(x=X_test1['median_income'], y=y_test1, opacity=0.65, 
                trendline='ols', trendline_color_override='darkred')


# In[ ]:


import plotly.graph_objects as go

x_range = pd.DataFrame({ 'median_income': np.linspace(X_train['median_income'].min(), X_train['median_income'].max(), 100)})
y_range =  lin_pipe.predict(x_range)

go.Figure([
    go.Scatter(x=X_train.squeeze(), y=y_train, name='train', mode='markers'),
    go.Scatter(x=X_test.squeeze(), y=y_test, name='test', mode='markers'),
    go.Scatter(x=x_range.median_income, y=y_range, name='prediction')
])


# ### Model 1.2 - Linear OLS Regression with Statsmodels

# Im Folgenden wird die Lineare Regression mit Statsmodels einmal mit den klassischen Varaiblen und einmal mit selbst erstellten Varaiblen druchgeführt. Bei beiden Varianten werden mit Hilfe von Cook's Distance Outlier identifiziert und aus dem Modell entfernt.  
# Dann soll vergleichen werden, welche Vorgehensweise zu einem besseren Ergebnis führt.

# Backward / Forward Selection einbauen

# #### Model mit Original Variablen

# In[ ]:


# Fit Model with original features
lm1 = smf.ols(formula='median_house_value ~ median_income + housing_median_age + ocean_proximity + population + total_bedrooms', data=train_dataset2).fit()


# In[ ]:


# Short summary
lm1.summary().tables[1]


# In[ ]:


# Full summary
lm1.summary()


# Interpretation  
# 
# If Durbin–Watson is less than 1.0, there may be cause for concern.
# 
# Small values of d indicate successive error terms are positively correlated.
# 
# If d > 2, successive error terms are negatively correlated.

# Omnisbus und JB kann bei großen Datensätzen vernachlässigt werden (SKript? Regression Diagnostics)

# #### **Regression Diagnostics**

# In[ ]:


# influence plot 
fig = sm.graphics.influence_plot(lm1, criterion="cooks")
fig.tight_layout(pad=1.0)


# In[ ]:


# obtain Cook's distance 
lm1_cooksd = lm1.get_influence().cooks_distance[0]

# get length of df to obtain n
n = len(train_dataset2["median_income"])

# calculate critical d
critical_d = 4/n
print('Critical Cooks distance:', critical_d)

# identification of potential outliers with leverage
out_d = lm1_cooksd > critical_d

# output potential outliers with leverage
print(train_dataset2.index[out_d], "\n", 
    lm1_cooksd[out_d])


# In[ ]:


train_dataset2a=train_dataset2.drop(train_dataset.index[out_d])


# In[ ]:


# Fit Model with original features
lm2 = smf.ols(formula='median_house_value ~ median_income + housing_median_age + ocean_proximity + population + total_bedrooms', data=train_dataset2a).fit()


# In[ ]:


# Full summary
lm2.summary()


# #### Model mit selbst erstellten Variablen

# In[ ]:



lm3 = smf.ols(formula='median_house_value ~ median_income + ocean_proximity + household_per_person + rooms_per_household + housing_median_age', data=train_dataset2).fit()


# In[ ]:


# Short summary
lm3.summary().tables[1]


# P < 0,05, dh wir können Null Hyptohese verwerfen, dh Zusammenhang ist da

# In[ ]:


# Full summary
lm3.summary()


# In[ ]:


lm4 = smf.ols(formula='median_house_value ~ median_income + ocean_proximity + household_per_person + housing_median_age', data=train_dataset2).fit()


# In[ ]:


# obtain Cook's distance 
lm4_cooksd = lm4.get_influence().cooks_distance[0]

# get length of df to obtain n
n = len(train_dataset2["median_income"])

# calculate critical d
critical_d = 4/n
print('Critical Cooks distance:', critical_d)

# identification of potential outliers with leverage
out_d = lm4_cooksd > critical_d

# output potential outliers with leverage
print(train_dataset2.index[out_d], "\n", 
    lm4_cooksd[out_d])


# In[ ]:


train_dataset2b=train_dataset2.drop(train_dataset2.index[out_d])


# In[ ]:


lm4 = smf.ols(formula='median_house_value ~ median_income + ocean_proximity + household_per_person + housing_median_age', data=train_dataset2b).fit()


# In[ ]:


lm4.summary().tables[1]


# In[ ]:


lm4.summary()


# In[ ]:


print("SSR:", lm4.ssr)
print("MSE:", lm4.mse_resid)
print("RMSE", np.sqrt(lm4.mse_resid))


# In[ ]:


# Add the regression predictions (as "pred") to our DataFrame
train_dataset2b['y_pred2'] = lm4.predict()


# In[ ]:


# MSE
mse(train_dataset2b['median_house_value'], train_dataset2b['y_pred2'])


# In[ ]:


y_pred2 = lm4.predict(X_test1)


# In[ ]:


r2_score(y_test1, y_pred2)


# #### Regression Diagnostics

# In[ ]:


# Plot regression line 
plt.scatter(train_dataset2b['median_income'], train_dataset2b['median_house_value'],  color='black')
plt.plot(train_dataset2b['median_income'], train_dataset2b['y_pred2'], color='darkred', linewidth=3);


# In[ ]:


sns.lmplot(x='median_income', y='median_house_value', data=train_dataset2b, line_kws={'color': 'darkred'}, ci=False);


# In[ ]:


sns.lmplot(x='housing_median_age', y='median_house_value', data=train_dataset2b, line_kws={'color': 'darkred'}, ci=False);


# In[ ]:


sns.residplot(x="y_pred2", y="median_house_value", data=train_dataset2b, scatter_kws={"s": 80});


# In[ ]:


# Regression diagnostics für Variable "median_income"
fig = sm.graphics.plot_regress_exog(lm4, "median_income")
fig.tight_layout(pad=1.0)


# In[ ]:


# Regression diagnostics für Variable "housing_median_age"
fig = sm.graphics.plot_regress_exog(lm4, "housing_median_age")
fig.tight_layout(pad=1.0)


# In[ ]:


# Regression diagnostics für alle Variablen
fig = sm.graphics.plot_partregress_grid(lm4)
fig.tight_layout(pad=0.1)


# In[ ]:


# Inspect correlation
# Calculate correlation using the default method ( "pearson")
corr = train_dataset2b.corr()
# optimize aesthetics: generate mask for removing duplicate / unnecessary info
mask = np.zeros_like(corr, dtype=bool)
mask[np.triu_indices_from(mask)] = True
# Generate a custom diverging colormap as indicator for correlations:
cmap = sns.diverging_palette(220, 10, as_cmap=True)
# Plot
sns.heatmap(corr, mask=mask, cmap=cmap, annot=True,  square=True, annot_kws={"size": 12});


# In[ ]:


# Validation with Test Data?


# ## 2. Lasso Regression

# ### 2.1 Lasso Regression mit scikit-learn

# In[ ]:


# Erstellen der X und Y Variablen
y2 = train_dataset['median_house_value']
features = ['median_income', 'household_per_person', 'ocean_proximity', 'housing_median_age']
X2 = train_dataset[features]


# #### Split Data

# In[ ]:


# Data split
X_train2, X_test2, y_train2, y_test2 = train_test_split(X2, y2, test_size=0.3, random_state=10)


# In[ ]:


# make list of numerical features (League_N, Division_W and NewLeague_N are categorcial) 
list_numerical = X2.drop(['ocean_proximity'], axis=1).columns


# In[ ]:


list_numerical


# #### Standardization

# Lasso performs best when all numerical features are centered around 0 and have variance in the same order. If a feature has a variance that is orders of magnitude larger than others, it might dominate the objective function and make the estimator unable to learn from other features correctly as expected.
# 
# This means it is important to standardize our features. We do this by subtracting the mean from our observations and then dividing the difference by the standard deviation. This so called standard score  for an observation  is calculated as:
# 
#  
# where:
# 
# x is an observation in a feature
# 
#  is the mean of that feature
# 
# s is the standard deviation of that feature.
# 
# To avoid data leakage, the standardization of numerical features should always be performed after data splitting and only from training data. Furthermore, we obtain all necessary statistics for our features (mean and standard deviation) from training data and also use them on test data. Note that we don’t standardize our dummy variables (which only have values of 0 or 1).

# #### Lasso - Model

# In[ ]:



# Erstellen der Pipeline mit Lasso Modell
lasso_pipe = Pipeline(steps=[
    ('preprocessor', preprocessor),
    ('lasso', Lasso(alpha=1))
                        ])


# In[ ]:


# Fitten von Pipeline/Modell
lasso_pipe.fit(X_train2, y_train2)


# In[ ]:


#categorical_features = ['ocean_proximity']


# In[ ]:


# Erstellen einer Liste aller Feature Namen
feature_names = np.concatenate((list_numerical.to_numpy(), lasso_pipe.named_steps['preprocessor'].transformers_[1][1]['onehot'].get_feature_names_out()))
feature_names


# #### Lasso - Model Evaluation

# In[ ]:


print('R squared training set', round(lasso_pipe.score(X_train2, y_train2)*100, 2))
print('R squared test set', round(lasso_pipe.score(X_test2, y_test2)*100, 2))


# In[ ]:


# Training data
pred_train = lasso_pipe.predict(X_train2)
mse_train = mean_squared_error(y_train2, pred_train)
print('MSE training set', round(mse_train, 2))

# Test data
pred_test = lasso_pipe.predict(X_test2)
mse_test =mean_squared_error(y_test2, pred_test)
print('MSE test set', round(mse_test, 2))


# #### Lasso - k-fold cross validation

# find best value for alpha

# In[ ]:



# Create pipeline with model
lassoCV_pipe = Pipeline(steps=[
    ('preprocessor', preprocessor),
    ('lassoCV', LassoCV(cv=5, random_state=0, max_iter=10000))
                        ])


# In[ ]:


# Fit model
lassoCV_pipe.fit(X_train2, y_train2)


# In[ ]:


lassoCV_pipe.named_steps['lassoCV'].alpha_


# In[ ]:


# get absolute values of coefficients
importance = np.abs(lassoCV_pipe.named_steps['lassoCV'].coef_)

sns.barplot(x=importance, 
            y=feature_names);


# In[ ]:


from sklearn.feature_selection import SequentialFeatureSelector
from time import time

tic_fwd = time()

sfs_forward = SequentialFeatureSelector(
    lassoCV_pipe, n_features_to_select=2, 
    direction="forward").fit(X_train2, y_train2)

toc_fwd = time()
print(
    "Features selected by forward sequential selection: "
    f"{feature_names[sfs_forward.get_support()]}"
)
print(f"Done in {toc_fwd - tic_fwd:.3f}s")


# #### Lasso - Lasso Best

# In[ ]:


# Create pipeline with model
lassobest_pipe = Pipeline(steps=[
    ('preprocessor', preprocessor),
    ('lassobest', Lasso(alpha=lassoCV_pipe.named_steps['lassoCV'].alpha_))
                            ])


# In[ ]:


# Set best alpha

lassobest_pipe.fit(X_train2, y_train2)


# In[ ]:


print('R squared training set lasso best', round(lassobest_pipe.score(X_train2, y_train2)*100, 2))
print('R squared test set lasso best', round(lassobest_pipe.score(X_test2, y_test2)*100, 2))


# In[ ]:


print('R squared training set lasso not best', round(lasso_pipe.score(X_train2, y_train2)*100, 2))
print('R squared test set lasso not best', round(lasso_pipe.score(X_test2, y_test2)*100, 2))


# ## 3. Splines

# In[ ]:


y3 = train_dataset[['median_house_value']]
X3 = train_dataset[['median_income']]


# In[ ]:


# data split
X_train3, X_test3, y_train3, y_test3 = train_test_split(X3, y3, test_size=0.3, random_state=10)

X_train3


# ### 3.1 Splines with sclearn

# In[ ]:


from sklearn.pipeline import make_pipeline
from sklearn.linear_model import Ridge


# Create pipeline with model
splines_pipe = Pipeline(steps=[
    ('preprocessor', preprocessor),
    ('splines', make_pipeline(SplineTransformer(n_knots=4, degree=3), 
                       Ridge(alpha=1)))
                        ])


# In[ ]:


splines_pipe.fit(X_train3, y_train3)

y_pred = splines_pipe.predict(X_train3)


# In[ ]:


import numpy as np

model_results(model_name = "spline")


# In[ ]:


y3.info()


# In[ ]:


import numpy as np
import matplotlib.pyplot as plt

# Create observations
x_new = np.linspace(X_test3.min(),X_test3.max(), 100)
# Make some predictions
pred = splines_pipe.predict(x_new)

# plot
sns.scatterplot(x=X_train3['median_income'], y=y_train3['median_house_value'])

plt.plot(x_new, pred, label='Cubic spline with degree=3', color='orange')
plt.legend();


# In[ ]:


print('R squared training set', round(splines_pipe.score(X_train3, y_train3)*100, 2))
print('R squared test set', round(splines_pipe.score(X_test3, y_test3)*100, 2))


# ### 3.2 Splines with Patsy

# In[ ]:


# Generating cubic spline with 3 knots at 1, 4 and 7
transformed_x = dmatrix(
            "bs(train, knots=(1,4,7), degree=3, include_intercept=False)", 
                {"train": X_train3},return_type='dataframe')


# In[ ]:


# Fitting generalised linear model on transformed dataset
spline2 = sm.GLM(y_train3, transformed_x).fit()


# In[ ]:


# Training data
pred_train = spline2.predict(dmatrix("bs(train, knots=(1,4,7), include_intercept=False)", {"train": X_train3}, return_type='dataframe'))
rmse_train = mean_squared_error(y_train3, pred_train, squared=False)

# Test data
pred_test = spline2.predict(dmatrix("bs(test, knots=(1,4,7), include_intercept=False)", {"test": X_test3}, return_type='dataframe'))
rmse_test =mean_squared_error(y_test3, pred_test, squared=False)

# Save model results
model_results = pd.DataFrame(
    {
    "model": "Cubic spline (cs)",  
    "rmse_train": [rmse_train], 
    "rmse_test": [rmse_test]
    })

model_results


# In[ ]:


# Create observations
xp = np.linspace(X_test3.min(),X_test3.max(), 100)
# Make some predictions
pred = spline2.predict(dmatrix("bs(xp, knots=(1,4,7), include_intercept=False)", {"xp": xp}, return_type='dataframe'))

# plot
sns.scatterplot(x=X_train3['median_income'], y=y_train3['median_house_value'])

plt.plot(xp, pred, label='Cubic spline with degree=3 (3 knots)', color='orange')
plt.legend();


# ### 3.3 Natural Spline with Patsy & Statsmodels

# In[ ]:


transformed_x3 = dmatrix("cr(train,df = 3)", {"train": X_train3}, return_type='dataframe')

spline3 = sm.GLM(y_train3, transformed_x3).fit()


# In[ ]:


# Training data
pred_train = spline3.predict(dmatrix("cr(train, df=3)", {"train": X_train3}, return_type='dataframe'))
rmse_train = mean_squared_error(y_train3, pred_train, squared=False)

# Test data
pred_test = spline3.predict(dmatrix("cr(test, df=3)", {"test": X_test3}, return_type='dataframe'))
rmse_test = mean_squared_error(y_test3, pred_test, squared=False)

# Save model results
model_results_ns = pd.DataFrame(
    {
    "model": "Natural spline (ns)",  
    "rmse_train": [rmse_train], 
    "rmse_test": [rmse_test]
    })

model_results_ns


# In[ ]:


# Make predictions
pred = spline3.predict(dmatrix("cr(xp, df=3)", {"xp": xp}, return_type='dataframe'))
xp = np.linspace(X_test3.min(),X_test3.max(), 100)
# plot
sns.scatterplot(x=X_train3['median_income'], y=y_train3['median_house_value'])
plt.plot(xp, pred, color='orange', label='Natural spline with df=3')
plt.legend();

