#!/usr/bin/env python
# coding: utf-8

# # Classification Modelle

# # Setup
# ---

# ## Import Libraries & Daten

# In[1]:


import pandas as pd
import numpy as np

import statsmodels.api as sm
import statsmodels.formula.api as smf

from scipy.stats import chi2_contingency, fisher_exact

import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.model_selection import train_test_split

import sklearn.linear_model as skl_lm


# Module für Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.compose import make_column_selector as selector
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler, OneHotEncoder

from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay
from sklearn.metrics import classification_report

sns.set_theme()
get_ipython().run_line_magic('matplotlib', 'inline')


# In[2]:


ROOT = "https://raw.githubusercontent.com/jan-kirenz/project-OliScha/main/"
DATA = "project_data.csv?token=GHSAT0AAAAAABPCEITIYHBIEPRTFMZJXUGKYPKREJQ"

df = pd.read_csv(ROOT + DATA)


# In[3]:


# prüfen ob Import funktioniert hat
df


# ## Erstellen der Pipeline für scikit learn Modelle

# In[4]:


# für numeric features
numeric_transformer = Pipeline(steps=[
    ('imputer', SimpleImputer(strategy='median')),
    ('scaler', StandardScaler())
    ])


# In[5]:


# für categorical features  
categorical_transformer = Pipeline(steps=[
    ('imputer', SimpleImputer(strategy='constant', fill_value='missing')),
    ('onehot', OneHotEncoder(handle_unknown='ignore'))
    ])


# In[6]:


# Erstellen der Pipeline, zusammenführen von cat und numeric transformer
preprocessor = ColumnTransformer(transformers=[
    ('num', numeric_transformer, selector(dtype_exclude="category")),
    ('cat', categorical_transformer, selector(dtype_include="category"))
        ])


# Siehe Kapitel *Regression*

# # Data Split
# ---

# In[7]:


# Data Split
train_dataset = df.sample(frac=0.8, random_state=0)
test_dataset = df.drop(train_dataset.index)

train_dataset


# # Data Transformation
# ---

# ## Anpassung Datatypes & Bereinigung

# Hier werden die gleichen Schritte durchgeführt wie im regression notebook.

# In[8]:


# die identifizierten Observations mit NULL values sollen nun entfernt werden
train_dataset = train_dataset.dropna()


# In[9]:


# change datatype zu string um str.replace transformation durchzuführen
train_dataset['median_house_value'] = train_dataset['median_house_value'].astype("string")
train_dataset['housing_median_age'] = train_dataset['housing_median_age'].astype("string")

# Bereinigung der fehlerhaften Werte
train_dataset.median_house_value = train_dataset.median_house_value.str.replace("$", "", regex =True)
train_dataset.housing_median_age = train_dataset.housing_median_age.str.replace("years", "", regex =True)


# In[10]:


# Anpassung der datatypes
train_dataset['median_house_value'] = train_dataset['median_house_value'].astype("float64")
train_dataset['housing_median_age'] = train_dataset['housing_median_age'].astype("float64")
train_dataset['total_bedrooms'] = train_dataset['total_bedrooms'].astype("int64")
train_dataset['ocean_proximity'] = train_dataset['ocean_proximity'].astype("category")


# In[11]:


#correct price_category values
train_dataset['updated_price_cat'] = np.where(train_dataset.median_house_value >= 150000, 'above', 'below')


# In[12]:


train_dataset['updated_price_cat'] = train_dataset['updated_price_cat'].astype("category")


# In[13]:


train_dataset = train_dataset.drop(columns=['price_category','median_house_value'])


# In[14]:


train_dataset


# ## Feature Engineering

# In[15]:


# Erstellen neuer Variablen
train_dataset=train_dataset.assign(people_per_household=lambda train_dataset: train_dataset.population/train_dataset.households)
train_dataset=train_dataset.assign(bedrooms_per_household=lambda train_dataset: train_dataset.total_bedrooms/train_dataset.households)
train_dataset=train_dataset.assign(rooms_per_household=lambda train_dataset: train_dataset.total_rooms/train_dataset.households)
train_dataset=train_dataset.assign(bedrooms_per_room=lambda train_dataset: train_dataset.total_bedrooms/train_dataset.total_rooms)


# In[16]:


# people_per_household outlier droppen
train_dataset = train_dataset.drop(index=[19006,3364,16669,13034])
# rooms_per_household outlier droppen
train_dataset = train_dataset.drop(index=[1914,1979])
# bedrooms_per_household bereits in anderen enthalten und gedropped


# In[17]:


train_dataset.ocean_proximity = train_dataset.ocean_proximity.str.replace("ISLAND", "NEAR WATER", regex =True)
train_dataset.ocean_proximity = train_dataset.ocean_proximity.str.replace("NEAR BAY", "NEAR WATER", regex =True)
train_dataset.ocean_proximity = train_dataset.ocean_proximity.str.replace("NEAR OCEAN", "NEAR WATER", regex =True)
train_dataset.ocean_proximity = train_dataset.ocean_proximity.str.replace("<1H OCEAN", "NEAR WATER", regex =True)


# In[18]:


# durch die gerade stattgefundene Transformation muss der datatype wieder korrigiert werden
train_dataset['ocean_proximity'] = train_dataset['ocean_proximity'].astype("category")


# # Modelling

# ## Model 1 - Logistic Regression mit statsmodels

# Zu Beginn wird eine Kopie des Trainingsdatensets erstellt, um dieses bearbeiten zu können und das Original Datenset noch für das nächste Model zur Verfügung zu haben ohne es erneut erzeugen zu müssen.

# Zur Vorhersage für das Logistic Regression Modell werden die gleichen Features verwendet, welche sich bereits in der linearen Regression bewährt haben.

# In[19]:


model1 = smf.glm(formula = 'updated_price_cat ~ median_income + ocean_proximity + bedrooms_per_room + people_per_household + housing_median_age', data=train_dataset, family=sm.families.Binomial()).fit()


# In[20]:


print(model1.summary())


# DIe P-Values der Features weisen alle einen Wert von unter 0,05 auf, was bedeutet, dass wir die Null Hypothese verwerfen können und die Features für unser Modell verwenden können.

# ### Predict

# Im nächsten Schritt erstellen wir Vorhersagen auf Basis unseres Modells und erweitern unser dataset um die Angabe, bei welchem Threshold welche Vorhersage gemacht werden würde.

# In[21]:


# Predictions erstellen und die Wahrscheinlichkeit zum dataset hinzufügen
train_dataset['Probability_above'] = model1.predict()


# In[22]:


# Use thresholds to discretize Probability
train_dataset['Threshold 0.4'] = np.where(train_dataset['Probability_above'] > 0.4, 'above', 'below')
train_dataset['Threshold 0.5'] = np.where(train_dataset['Probability_above'] > 0.5, 'above', 'below')
train_dataset['Threshold 0.6'] = np.where(train_dataset['Probability_above'] > 0.6, 'above', 'below')
train_dataset['Threshold 0.7'] = np.where(train_dataset['Probability_above'] > 0.7, 'above', 'below')

train_dataset


# Im nächsten Schritt wird eine Funktion definiert, mit welcher die Confusion Matrizen für dieses Modell mit verschiedenen Thresholds dargestellt werden.

# In[23]:


def print_metrics(train_dataset, predicted):
    # Header
    print('-'*50)
    print(f'Metrics for: {predicted}\n')
    
    # Confusion Matrix
    y_actu = pd.Series(train_dataset['updated_price_cat'], name='Actual')
    y_pred = pd.Series(train_dataset[predicted], name='Predicted')
    train_dataset_conf = pd.crosstab(y_actu, y_pred)
    display(train_dataset_conf)
    
    # Confusion Matrix to variables:
    pop = train_dataset_conf.values.sum()
    tp = train_dataset_conf['above']['above']
    tn = train_dataset_conf['below']['below']
    fp = train_dataset_conf['above']['below']
    fn = train_dataset_conf['below']['above']
    
    # Metrics
    accuracy = (tp + tn) / pop
    precision = tp / (tp + fp)
    recall = tp / (tp + fn)
    f1_score = 2 * ((precision * recall) / (precision + recall))
    print(f'Accuracy:  {accuracy:.4f}')
    print(f'Precision: {precision:.4f}')
    print(f'Recall:    {recall:.4f}')
    print(f'F1 Score:  {f1_score:.4f} \n')


# In[24]:


print_metrics(train_dataset, 'Threshold 0.4')
print_metrics(train_dataset, 'Threshold 0.5')
print_metrics(train_dataset, 'Threshold 0.6')
print_metrics(train_dataset, 'Threshold 0.7')


# Aus der Sicht eines Immobilienunternehmens sollten weder zu viele Fehlinvestitionen getätigt werden noch gute Investitionsgelegenheiten verpassen werden. Deshalb wird der Threshold mit dem besten F1 Score, welche das Mittel zwischen Precision und Recall darstellt, gewählt.
# Demnach wäre das **Modell** **mit** **Threshold** **0,5** am besten geeignet.

# Die zuvor hinzugefügten Spalten zur Evaluation des model1 werden nun wieder entfernt, um das dataset für das zweite Modell zu verwenden.

# In[25]:


# droppen der hinzugefügten Spalten
train_dataset = train_dataset.drop(columns=['Probability_above','Threshold 0.4','Threshold 0.5','Threshold 0.6','Threshold 0.7'])


# In[26]:


train_dataset


# ## Model 2 - Logistic Regression mit scikit learn

# ### Data Split

# Für das scikit learn Modell werden die Daten in das von scikit learn verlangte Format gebracht. Dabei erzeugen wir auch Evaluationsdaten, mit welchenwir unser Modell vor der finalen Evaluation mit den Testdaten noch einmal bewerten können.

# In[27]:


# prepara data for scikit learn 
X2 = train_dataset.drop(columns=['updated_price_cat','ocean_proximity'])
y2 = train_dataset.updated_price_cat


# In[28]:



# Split in Trainings- und Testdaten
X_train, X_val, y_train, y_val = train_test_split(X2, y2, test_size=0.2, random_state=42)


# ### Model 

# In[29]:


# Erstellen des Models mit Pipeline
class_pipe = Pipeline(steps=[
    ('preprocessor', preprocessor),
    ('classification', skl_lm.LogisticRegression())
                        ])


# In[30]:


y_pred = class_pipe.fit(X_train, y_train).predict(X_val)


# In[31]:


# Return the mean accuracy on the given test data and labels:
class_pipe.score(X_val, y_val)


# In[32]:


cm = confusion_matrix(y_val, y_pred)

disp = ConfusionMatrixDisplay(confusion_matrix=cm,
                              display_labels=class_pipe.classes_)
disp.plot()
plt.show()


# In[33]:


print(classification_report(y_val, y_pred, target_names=['above', 'below']))


# ### Change Threshold

# In[34]:


pred_proba = class_pipe.predict_proba(X_val)

y_pred_threshold = np.where(pred_proba[:,0] >= .4, 'above', 'below')

cm = confusion_matrix(y_val, y_pred_threshold)

disp = ConfusionMatrixDisplay(confusion_matrix=cm,
                              display_labels=class_pipe.classes_)
disp.plot()
plt.show()


# In[35]:


print(classification_report(y_val, y_pred_threshold, target_names=['above', 'below']))


# In[36]:


pred_proba = class_pipe.predict_proba(X_val)



y_pred_threshold2 = np.where(pred_proba[:,0] >= .6, 'above', 'below')

cm = confusion_matrix(y_val, y_pred_threshold2)

disp = ConfusionMatrixDisplay(confusion_matrix=cm,
                              display_labels=class_pipe.classes_)
disp.plot()
plt.show()


# In[37]:


print(classification_report(y_val, y_pred_threshold2, target_names=['above', 'below']))


# Mit der gleichen Begründung wie im ersten Modell wird hier das Modell mit dem besten F1-Score, also das Modell mit dem Threshold von 0,5 gewählt.

# # Finale Evaluation mit Testdaten

# ## Anpassung Testdaten

# ### Anpassung datatypes & Bereinigung

# Die Anpassung der Datentypen und das Entfernen der nicht benötigten Spalten muss genauso durchgeführt werden wie bei den Trainingsdaten.

# In[38]:


# droppen von NULL Values
test_dataset = test_dataset.dropna()


# In[39]:


# Anpassung der datatypes
test_dataset['median_house_value'] = test_dataset['median_house_value'].astype("float64")
test_dataset['housing_median_age'] = test_dataset['housing_median_age'].astype("float64")
test_dataset['total_bedrooms'] = test_dataset['total_bedrooms'].astype("int64")
test_dataset['ocean_proximity'] = test_dataset['ocean_proximity'].astype("category")


# In[40]:


#correct price_category values
test_dataset['updated_price_cat'] = np.where(test_dataset.median_house_value >= 150000, 'above', 'below')


# In[41]:


test_dataset['updated_price_cat'] = test_dataset['updated_price_cat'].astype("category")


# In[42]:


test_dataset = test_dataset.drop(columns=['price_category'])
test_dataset = test_dataset.drop(columns=['median_house_value'])


# ### Feature Engineering Testdaten

# Für die Testdaten müssen die gleichen Anpassungen an den Features durchgeführt werden wie bei den Trainingsdaten.

# In[43]:


# neue Varaiablen müssen auch in den Testdaten ergänzt werden
test_dataset=test_dataset.assign(people_per_household=lambda test_dataset: test_dataset.population/test_dataset.households)
test_dataset=test_dataset.assign(bedrooms_per_household=lambda test_dataset: test_dataset.total_bedrooms/test_dataset.households)
test_dataset=test_dataset.assign(rooms_per_household=lambda test_dataset: test_dataset.total_rooms/test_dataset.households)
test_dataset=test_dataset.assign(bedrooms_per_room=lambda test_dataset: test_dataset.total_bedrooms/test_dataset.total_rooms)


# In[44]:


test_dataset.ocean_proximity = test_dataset.ocean_proximity.str.replace("ISLAND", "NEAR WATER", regex =True)
test_dataset.ocean_proximity = test_dataset.ocean_proximity.str.replace("NEAR BAY", "NEAR WATER", regex =True)
test_dataset.ocean_proximity = test_dataset.ocean_proximity.str.replace("NEAR OCEAN", "NEAR WATER", regex =True)
test_dataset.ocean_proximity = test_dataset.ocean_proximity.str.replace("<1H OCEAN", "NEAR WATER", regex =True)


# In[45]:


# durch die gerade stattgefundene Transformation muss der datatype wieder korrigiert werden
test_dataset['ocean_proximity'] = test_dataset['ocean_proximity'].astype("category")


# ## Split Testdaten

# In[46]:


X2 = test_dataset.drop(columns=['updated_price_cat','ocean_proximity'])
y2 = test_dataset.updated_price_cat


# ## Validierung Classification mit statsmodels

# Hier wird das Classification Modell von statsmodels evaluiert. Zur Visualisierung wird die Confusion Matrix und der Classification Report verwendet.

# In[47]:


y_pred_test1 = model1.predict(test_dataset)

y_pred_test3 = np.where(y_pred_test1 >= .5, 'above', 'below')

cm = confusion_matrix(y2, y_pred_test3)

disp = ConfusionMatrixDisplay(confusion_matrix=cm,
                              display_labels=class_pipe.classes_)
disp.plot()
plt.show()


# In[48]:


print(classification_report(y2, y_pred_test3, target_names=['above', 'below']))


# In der Evaluation mit den Testdaten schneidet das Modell leicht schlechter ab als mit den Trainingsdaten, liefert aber immernoch ein zufriedenstellendes Ergebnis.  
# 
# Zum Vergleich die KPIs mit den Trainingsdaten:  
# Accuracy:  0.8523  
# Precision: 0.8667  
# Recall:    0.9058  
# F1 Score:  0.8858

# ## Validierung Classification mit scikit learn

# In[49]:


pred_proba = class_pipe.predict_proba(X2)

y_pred_test2 = np.where(pred_proba[:,0] >= .5, 'above', 'below')

cm = confusion_matrix(y2, y_pred_test2)

disp = ConfusionMatrixDisplay(confusion_matrix=cm,
                              display_labels=class_pipe.classes_)
disp.plot()
plt.show()


# In[50]:


print(classification_report(y2, y_pred_test2, target_names=['above', 'below']))


# Auch dieses Modell schneidet in der Evaluation mit den Testdaten leicht schlechter ab als mit den Trainingsdaten, liefert aber immernoch ein zufriedenstellendes Ergebnis.  
# 
# Zum Vergleich die KPIs mit den Trainingsdaten:  

# In[51]:


# Classification Report mit Trainingsdaten Threshold 0,5
print(classification_report(y_val, y_pred, target_names=['above', 'below']))

