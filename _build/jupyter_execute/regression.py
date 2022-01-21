#!/usr/bin/env python
# coding: utf-8

# # Regression Modelle

# # Setup
# ---

# ### Import Module & Libraries

# Zu Beginn werden die benötigten Komponenten aus den entsprechenden Libraries importiert.

# In[1]:


# Allgemein
get_ipython().run_line_magic('matplotlib', 'inline')
import pandas as pd
import numpy as np
import seaborn as sns 
sns.set_theme(style="ticks", color_codes=True)
import matplotlib.pyplot as plt
import plotly.express as px
from patsy import dmatrix

# statsmodels
from statsmodels.compat import lzip
import statsmodels.api as sm
#from statsmodels.formula.api import ols
from statsmodels.stats.outliers_influence import variance_inflation_factor
import statsmodels.formula.api as smf
from statsmodels.tools.eval_measures import mse, rmse
from statsmodels.tools.tools import add_constant

# sklearn
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
from sklearn.preprocessing import SplineTransformer
from sklearn.pipeline import make_pipeline
from sklearn.linear_model import Ridge


# ### Erstellen einer Pipeline

# Um die numerische und kategoriale Variablen automatisch in ein korrektes Format für die scikit learn Modelle zu transformieren, wird an dieser Stelle eine Pipeline definiert. Diese besteht aus verschiedenen Elementen, welche bspw. fehlende Werte ausfüllt (SimpleImputer) oder kategoriale Variablen in ein binäres Format umwandelt (OneHotEncoder).

# In[2]:


# für numeric features
numeric_transformer = Pipeline(steps=[
    ('imputer', SimpleImputer(strategy='median')),
    ('scaler', StandardScaler())
    ])


# In[3]:


# für categorical features  
categorical_transformer = Pipeline(steps=[
    ('imputer', SimpleImputer(strategy='constant', fill_value='missing')),
    ('onehot', OneHotEncoder(handle_unknown='ignore'))
    ])


# In[4]:


# Erstellen der Pipeline, zusammenführen von cat und numeric transformer
preprocessor = ColumnTransformer(transformers=[
    ('num', numeric_transformer, selector(dtype_exclude="category")),
    ('cat', categorical_transformer, selector(dtype_include="category"))
        ])


# # Import Data
# ---

# Als nächstes wird der Datensatz aus dem GitHub Repository importiert.

# In[5]:


ROOT = "https://raw.githubusercontent.com/jan-kirenz/project-OliScha/main/"
DATA = "project_data.csv?token=GHSAT0AAAAAABPCEITIYHBIEPRTFMZJXUGKYPKREJQ"

df = pd.read_csv(ROOT + DATA)


# Wir werfen einen ersten Blick auf die Daten um zu prüfen, auch um zu prüfen ob der Import funktioniert hat. Hier fällt direkt auf, dass die erste Observation durch Text und ein Währungszeichen verunreigt ist.

# In[6]:


df


# # Data Split
# ---

# Direkt nach dem Import und einem ersten Blick auf die Daten werden die Daten in Traings- und Testdaten aufgeteilt. Alle folgenden Schritt finden auf Basis der Trainingsdaten statt, um möglichst realistische Ergebnisse bei der späteren Evaluation mit Testdaten zu erzielen.
# 
# Dadurch soll ein möglichst realistischer generalization error (or out-of-sample error) erzielt werden.
# 
# Bestimmte Schritte der Datenaufbereitung (datatypes, selbst erstellte Variablen etc.) müssen dann späte rallerdings auch für die Testdaten durchgeführt werden, um diese zur Evaluation nutzen zu können.

# In[7]:


# Data Split
train_dataset = df.sample(frac=0.8, random_state=0)
test_dataset = df.drop(train_dataset.index)

train_dataset


# # Data Inspection
# ---

# ### Anomaly Detection

# In diesem Abschnitt werden die Daten genauer untersucht um Anomalien zu identifizieren und ggfs. zu bereinigen.

# In[8]:


train_dataset.head()


# Bereits beim ersten Blick auf das df ist aufgefallen, dass in der Observation mit Index = 0 die Variablen *housing_median_age* und *median_house_value* nicht dem Format der übrigen Daten in der gleichen Spalte entsprechen. Die Observation ist noch in unserem Traingsdatenset enthalten und muss in den nächsten Schritten bereinigt werden

# In[9]:


train_dataset.loc[0, : ]


# Diese Verunreinigung sorgt auch dafür, dass den Variablen *housing_median_age* und *median_house_value* automatisch der Datatype *object* zugewiesen wurde,
# obwohl man eher einen numerischen Wert erwarten würde (float oder int). Dies muss auch nach der Bereinigung korrigiert werden.
# 
# In der Übersicht .info() kann außerdem bereits erkannt werden, dass einige Observations in der Spalte *total_bedrooms* keinen Wert (NULL) vorliegen haben.
# 
# 

# In[10]:


train_dataset.info()


# In[11]:


# Identifizieren der NULL Werte via Heatmap
sns.heatmap(train_dataset.isnull(), 
            yticklabels=False,
            cbar=False, 
            cmap='viridis');


# In[12]:


# Identifizieren der NULL Werte via Liste
print(train_dataset.isnull().sum())


# Die 160 identifizierten Datensätze mit den NULL Values in der Variablen *total_bedrooms* werden im nächsten Schritt entfernt.  
# Auf ein Ausfüllen der leeren Felder mit Mittelwert o.ä. wird hier verzichtet, da es sich hierbei lediglich um ca. 1% der Daten handelt und ein blindes Ausfüllen per mean auch Schaden anrichten könnte.

# In[13]:


# Droppen der NULL values
train_dataset = train_dataset.dropna()


# Die Variable *price_category* wird direkt aus unserer vorherzusagenden Variable *median_house_value* abgeleitet und eignet sich daher nicht als Feature.

# In[14]:


# Droppen der nicht benötigten Variable
train_dataset = train_dataset.drop(columns=['price_category'])


# Im nächsten Schritt wird der zuvor identifizierte verunreinigte Datensatz korrigiert. Da es sich nur um einen einzigen Datensatz handelt, könnte dieser ebenfalls gelöscht werden.  
# Da allerdings die Daten vollständig vorhanden sind, sollen diese behalten und bereinigt werden.

# In[15]:


# change datatype zu string um str.replace transformation durchzuführen
train_dataset['median_house_value'] = train_dataset['median_house_value'].astype("string")
train_dataset['housing_median_age'] = train_dataset['housing_median_age'].astype("string")


# In[16]:


# Bereinigung der fehlerhaften Werte
train_dataset.median_house_value = train_dataset.median_house_value.str.replace("$", "", regex =True)
train_dataset.housing_median_age = train_dataset.housing_median_age.str.replace("years", "", regex =True)


# ### Schema Definition

# Da nun alle offensichtlichen Anomalien in den Trainingsdaten entfernt wurden, sollen als nächstes die Datentypen der Variablen angepasstwerden.  
# Numerische Variablen werden je nach Variable als *float* oder *int* definiert, kategoriale Varaiblen als *cat*.  
# Die Varaiblen *housing_median_age*, *median_house_value* und *ocean_proximity* passen noch nicht in das Schema und müssen angepasst werden.
# - *housing_median_age* --> float, da das Alter theoretisch auch in nicht gerundeten Werten angegeben werden könnte (bspw. 10,5 Jahre - was in unseren Daten allerdings nicht der Fall ist)
# - *median_house_value* --> float, da Wert auch mit Nachkommastellen angegeben werden könnte
# - *ocean_proximity* --> cat, dem Abstand zum Meer könnte theoretisch eine Reihenfolge zugewiesen werden
# - *total_bedrooms* --> int, wie bei *total_rooms*, da es in der Regel keine halben Räume in Districts gibt

# In[17]:


# Anpassung der datatypes
train_dataset['median_house_value'] = train_dataset['median_house_value'].astype("float64")
train_dataset['housing_median_age'] = train_dataset['housing_median_age'].astype("float64")
train_dataset['total_bedrooms'] = train_dataset['total_bedrooms'].astype("int64")
train_dataset['ocean_proximity'] = train_dataset['ocean_proximity'].astype("category")


# In[18]:


train_dataset.info()


# # Deskriptive Statistik
# ---

# In diesem Abschnitt sollen die Variablen genauer auf statistische Merkmale untersucht werden.  
# Zuerst werden die numerischen Variablen betrachtet.

# In[19]:


# Zusammenfassung für alle numerischen Varaiblen
round(train_dataset.describe(),1).transpose()


# Folgende Auffälligkeiten sind hier zu sehen:
# - Bei den Variablen *total_rooms*,*total_bedrooms*, *population* und *households* liegt der höchste Wert über dem 10fachen des Wertes des 3 Quartils. Das deutet darauf hin, dass es einen oder wenige Distrikte gibt, welche viel größer sind als die Mehrzahl der Distrikte. Dies könnte im Feature Engineering ausgeglichen werden, indem durch Durchschnitsswerte die größe des Distrikts relativiert wird (bspw. Räume pro Person anstatt Räume im gesamten Distrikt)
# - Auch bei der Variablen *median_income* fällt auf, dass der höchste Werte sehr hoch liegt im Vergleich zur Verteilung auf die Quartile (75% liegen unter 4.7, max bei 15.0)
# - An Hand der **Standardabweichung** können wir erkennen, dass bei vielen Variablen eine hohe Streubreite vorliegt (bspw. bei *population*, *total_bedrooms*, *total_rooms*).  
# Bei Longitude und Latitude ist die Streuung am geringsten, was darauf hindeutet, dass die Distrikte nicht über den Globus verteilt sind sondern sich in einem bestimmten Land oder einer Region befinden.
# - Es gibt wohl min. einen Distrikt, welcher nur einen Haushalt beinhaltet, min. einen Distrikt der noch sehr neu ist (1 Jahr *housing_median_age*), min. einen Distrikt in dem nur 3 Personen leben.

# Als nächstes werden die kategorialen Variablen betrachtet.

# In[20]:


train_dataset.describe(include=["category"]).transpose()


# In[21]:


train_dataset['ocean_proximity'].value_counts()


# In[22]:


train_dataset['ocean_proximity'].value_counts(normalize=True)


# Da die kategoriale Variable *price_category* bereits entfernt wurde, existiert nur noch *ocean_proximity* als kategoriale Variable. Diese hat fünf Ausprägungen. Vor allem die Ausprägung *ISLAND* ist mit nur vier Observation sehr schwach vertreten. Diese könnte man ggfs. später mit anderen ähnlichen Ausprägungen zusammenführen.

# In[23]:


# Erstellen von Histogrammen
train_dataset.hist(bins=70, figsize=(20,15))
plt.show()


# Bei einem Blick auf die Histogramme fällt auf, dass es bei der vorherzusagenden Variablen *median_house_value* eine große Anzahl an Observations gibt, die einen extrem hohen Wert besitzen (ca. 500.000$).  
# Auch bei *housing_median_age* gibt es eine auffällig große Menge an Observations mit dem größten Wert (52 Jahre).
# 
# Evtl. wurden bei der Datenerhebung alle Werte größer den hier auffälligen Werten auf diesen Höchstwert gesetzt. Das könnte die Datenlage erklären, lässt sich in diesem Fall allerdings nicht bestätigen.
# 
# Bei den Geokoordinaten fällt auf, dass sich Observations um bestimmte Koordinaten herum häufen. Das kann auf bevölkerungsdichte Regionen mit vielen Distrikten hinweisen, wie bspw. bei großen Städten.

# # Exploratory Data Analysis
# ---

# ### Überblick

# Nachdem die statistischen Besonderheiten berachtet wurden, soll nun die Verteilung der Variablen sowie deren Abhängigkeiten untereinander sowie auf die vorherzusagende Variable untersucht werden.

# In[24]:


df.info()


# In[25]:


# Pairplot für numerische Variablen mit Farbmarkierung für ocean_proximity
sns.pairplot(data=train_dataset, vars=["total_rooms","total_bedrooms","population","households","median_house_value"],     hue="ocean_proximity");


# >**Erkenntnis:** 
# >- Einfluss auf Y Variable: Ein Zusammenhang zwischen *median_income* und *median_house_value* ist zu erkennen. Durch die Einfärbung nach *ocean_proximity* kann man erkennen, dass sich INLAND Distrikte tendenziell eher am unteren Wertebereich der Y-Variable befinden. Ansonsten gibt es erstmal keine weiteren offensichtlichen Zusammenhänge.  
# >- Variablen untereinander: Es gibt starke lineare Zusammenhänge zwischen *total_rooms*, *total_bedrooms*, *households* und *population*. Das ist soweit logisch und muss bei der Modellierung später beachtet werden (Collinearity).

# In[26]:


# Pairplot für numerische Variablen mit Farbmarkierung für ocean_proximity
sns.pairplot(data=train_dataset, vars=["housing_median_age","longitude","latitude","median_income","median_house_value"], hue="ocean_proximity");


# Auch an Hand der Korrelationswerte der numerischen Variablen lässt sich erkennen, dass lediglich *median_income* einen starken Einfluss auf *median_house_value* hat.

# In[27]:


corr = train_dataset.corr()
corr['median_house_value'].sort_values(ascending=False)


# In[28]:


# Create correlation matrix for numerical variables
corr_matrix = train_dataset.corr()
corr_matrix


# In[29]:


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


# Die oben bereits erkannten Zusammenhänge lassen sich hier nochmal auf einen Blick bestätigen.

# ---

# ### Analyse kategorialer Variablen

# Als nächstes wird die kategoriale Variable *ocean_proximity* näher untersucht.

# In[30]:


train_dataset['ocean_proximity'].value_counts()


# In[31]:


# Verteilung von ocean_proximity auf Geokoordinaten visualisieren
sns.jointplot(data=train_dataset, x='longitude', y='latitude', hue="ocean_proximity",height=10);


# Auf Basis der Geokoordinaten und der Form lässt sich feststellen, dass es sich bei unserem Datensatz um Distrikte aus Kalifornien handelt.  
# Auf dem Plot lässt sich erkennen, dass sich die Ausprägungen *ISLAND* und *NEAR BAY* sehr stark auf bestimmte Gebiete beschränken. Die anderen Ausprägungen sind einigermaßen gleichmäßig und nach einer Logik mit Abstand zum Meer verteilt.

# In[32]:


# Visualisierung Dichte
sns.jointplot(data=train_dataset, x='longitude', y='latitude', hue="ocean_proximity",height=10, alpha=0.2 );


# Durch eine kleine Änderung in den Parametern des Plots lassen sich besonders gut die Gebiete erkennen, aus welchen viele Observations vorliegen. Wie bereits vermutet handelt es sich vor allem um Ballungszentren bei San Francisco und Los Angeles.

# In[33]:


train_dataset.plot(kind="scatter", x="longitude", y="latitude", alpha=0.4, figsize=(10,7),
c="median_house_value", cmap=plt.get_cmap("jet"), colorbar=True,)
plt.legend()


# Mit Hilfe eines erweiterten Scatterplots können wir sogar visualisieren, wo die Distrikts mit den hohen *median_house_value* liegen. Wie zu erwarten war handelt es sich ebenfalls um die Gebiete rund um San Francisco und Los Angeles.

# In[34]:


# Untersuchung der kategroialen Variable "ocean_proximity" mit einem erweiterten Boxplot
sns.boxenplot(data=train_dataset, x="ocean_proximity", y="median_house_value");


# Durch ein erweitertes Boxplot Diagramm können wir erkennen, dass die Verteilung von *median_house_value* in den Ausprägungen *<1H OCEAN, NEAR BAY* und *NEAR OCEAN* sehr ähnlich ist. *INLAND* und *ISLAND* unterscheiden sich davon stark. Ggfs. könnte es Sinn machen, die ähnlichen Ausprägungen zusammenzufassen, das soll jedoch zuerst in den Modellen getestet werden.

# In[35]:


# Ergänzung zum Boxplot um Menge und Dichte der Observations zu visualisieren
sns.stripplot(data=train_dataset, x="ocean_proximity", y="median_house_value" , size=1 );


# Durch eine andere Form der Darstellung lässt sich nochmal gut die Menge und Dichte der Observations erkennen - und dass für *ISLAND* nur sehr wenige Observations vorliegen.

# In[36]:


# Analyse von "ocean_proximity" mit displot
sns.displot(data=train_dataset, x="median_house_value", hue = "ocean_proximity", kind="kde" )


# >**Fazit:** Die kategoriale Variable *ocean_proximity* hat definitiv Einfluss auf unsere vorherzusagende Variable. *INLAND* ist bspw. tenedneziell günstiger als die Distrikte, die näher am Meer liegen.   
# Auffällig ist, das es für die Ausprägung *ISLAND* nur sehr wenige Observations gibt und *ISLAND* und *NEAR BAY* stark auf bestimmte Gebiete beschränkt sind.

# ### Analyse numerischer Variablen

# Vor allem die numerische Variable *median_income* sieht auf Basis des zu Beginn erstellen pairplots vielversprechend aus und wird daher nun näher untersucht. Aber auch andere numerische Varaiblen sollen noch genauer betrachtet werden.

# In[37]:


# Analyse der Variablen "median_income"
sns.jointplot(data=train_dataset, x='median_income', y='median_house_value', hue="ocean_proximity" );


# Wie bereits im Parplot lässt sich hier nochmal gut erkennen, dass tendenziell mit steigendem *median_income* auch der *median_house_value* steigt.

# In[38]:


sns.lmplot(x='median_income', y='median_house_value', data=train_dataset, 
line_kws={'color': 'darkred'}, ci=False);


# Analyse "total_rooms"

# In[39]:


sns.scatterplot(data=train_dataset, x='total_rooms', y='median_house_value', hue="ocean_proximity" )


# Auch bei *total_rooms* lääst sich ein leichter Trend erkennen. Da es sich allerdings hier um die Gesamtzahl der Räume in einem Distrikt handelt, welche natürlich direkt abhängig von dessen Größe ist, ist diese Variable möglicherweise nicht besonders gut geeignet. Viel besser wäre eine Variable, die die Anzahl der Räume in Relation setzt zur Anzahl an Haushalten oder Menschen. So eine Variable kann später im Feature Engineering erstellt werden.

# In[40]:


sns.lmplot(x='total_rooms', y='median_house_value', data=train_dataset, 
line_kws={'color': 'darkred'}, ci=False);


# Analyse "housing_median_age"

# In[41]:


sns.scatterplot(data=train_dataset, x='housing_median_age', y='median_house_value', hue="ocean_proximity" )


# Bei der Variable *housing_median_age* könnte man eine Auswirkung auf median_house_value erwarten, da neuere Häuser teurer sein könnten als alte Häuser. Anhand der Analyse lässt sich jedoch erkennen, dass diese Auswirkung nur sehr gering ist. Das könnte man bspw. durch teure Altbauwohungen oder durchgeführte Instandhaltungsmaßnahmen erklären.

# In[42]:


sns.lmplot(x='housing_median_age', y='median_house_value', data=train_dataset, 
line_kws={'color': 'darkred'}, ci=False);


# >**Fazit:** Wie bereits erwartet bietet die Variable *median_income* das höchste Potential zur Vorhersage von *median_house_value*. Auch *ocean_proximity* zeigt eindeutig Auswirkungen auf die Y Variable, kann aber ggfs. noch zusammengefasst werden um später die Modelle zu optimieren. Auch bei weiteren Variablen kann Einfluss auf Y gefunden werden, allerdings ist dieser sehr schwach und sollte im Feauture Engineering optimiert werden.

# # Initial Feature Engineering
# ---

# ## Erstellen eigener Variablen

# In[43]:


# Erstellen neuer Variablen
train_dataset=train_dataset.assign(people_per_household=lambda train_dataset: train_dataset.population/train_dataset.households)
train_dataset=train_dataset.assign(bedrooms_per_household=lambda train_dataset: train_dataset.total_bedrooms/train_dataset.households)
train_dataset=train_dataset.assign(rooms_per_household=lambda train_dataset: train_dataset.total_rooms/train_dataset.households)
train_dataset=train_dataset.assign(bedrooms_per_room=lambda train_dataset: train_dataset.total_bedrooms/train_dataset.total_rooms)


# ## Analyse & Optimierung

# In[44]:


corr = train_dataset.corr()
corr['median_house_value'].sort_values(ascending=False)


# Anhand der Matrix lässt sich erkennen, dass von den neu erstellten Varaiblen vor allem *bedrooms_per_room* und *rooms_per_household* einen Einfluss auf die vorherzusagende Variable *median_house_value* haben.

# In[45]:


#Analyse neu erstellter Variablen
sns.pairplot(data=train_dataset, y_vars=["median_house_value"], 
x_vars=["people_per_household", "bedrooms_per_household", "rooms_per_household", "bedrooms_per_room"], hue="ocean_proximity");


# Hier lassen sich Outlier bei *people_per_household*, *bedrooms_per_household* und *rooms_per_household* erkennen. Diese können später entfernt werden.

# In[46]:


sns.scatterplot(data=train_dataset, x='bedrooms_per_room', y='median_house_value')


# In[47]:


sns.lmplot(data=train_dataset, x='bedrooms_per_room', y='median_house_value');


# In[48]:


sns.scatterplot(data=train_dataset, x='rooms_per_household', y='median_house_value')


# In[49]:


sns.lmplot(data=train_dataset, x='rooms_per_household', y='median_house_value');


# In[50]:


sns.scatterplot(data=train_dataset, x='people_per_household', y='median_house_value')


# In[51]:


sns.lmplot(data=train_dataset, x='people_per_household', y='median_house_value');


# Die zuvor auf dem Scatterplot aufgefallenen Outliers von *people_per_household* sollen nun identifiziert und entfernt werden. Distrikte, in denen im Schnitt mehrere hundert Menschen in einem Haushalt leben klingen nach fehlerhaften Daten (Unterschiedliche Methode in der Erfassung der Daten, Tippfehler etc.)

# In[52]:


train_dataset.nlargest (10,'people_per_household')


# In[53]:


train_dataset.nlargest (10,'rooms_per_household')


# In[54]:


train_dataset.nlargest (10,'bedrooms_per_household')


# In[55]:


# people_per_household outlier droppen
train_dataset = train_dataset.drop(index=[19006,3364,16669,13034])
# rooms_per_household outlier droppen
train_dataset = train_dataset.drop(index=[1914,1979])
# bedrooms_per_household bereits in anderen enthalten und gedropped


# Nach dem Entfernen der Outlier lässt sich eine stärkere Auswirkung der Features auf die Y-Variable erkennen.

# In[56]:


sns.lmplot(data=train_dataset, x='bedrooms_per_room', y='median_house_value');


# In[57]:


sns.lmplot(data=train_dataset, x='rooms_per_household', y='median_house_value');


# In[58]:


sns.lmplot(data=train_dataset, x='people_per_household', y='median_house_value');


# In[59]:


corr = train_dataset.corr()
corr['median_house_value'].sort_values(ascending=False)


# ## Auswahl Features

# Auf Basis der vorhergegangenen Analysen sollen nun vorläufige Features zur Modellierung ausgewählt werden. Da nicht mehrere Features gewählt werden sollten, welche sich aus den selben Variablen berechnen, wird jeweils die Variable mit der vielversprechendsten Correlation zu Y ausgewählt (bspw. *bedrooms_per_room* > *rooms_per_household* ).
# - *median_income*
# - *bedrooms_per_room*
# - *people_per_household*
# - *ocean_proximity*
# - *housing_median_age*
# 
# Obwohl *latitude* auch eine vielversprechende Correlation zeigt, wird diese Varaible nicht als feature für die Modelle ausgewählt, da geografische Informationen bereits in *ocean_proximity* enthalten sind.

# In[60]:



# choose features and add constant
features = add_constant(train_dataset[['median_house_value', 'median_income','bedrooms_per_room','people_per_household','housing_median_age']])
# create empty DataFrame
vif = pd.DataFrame()
# calculate vif
vif["VIF Factor"] = [variance_inflation_factor(features.values, i) for i in range(features.shape[1])]
# add feature names
vif["Feature"] = features.columns

vif.round(2)


# Um unsere ausgewählten Variablen auf Multikollinearität zu untersuchen, berechnen wir den Variance Inflation Factor (VIF). Da sich dieser bei allen Variablen unter 5 befindet, stellt die Kollinearität bei diesen Variablen kein Problem dar.

# ## Ergänzung

# Nach Durchlauf mehrerer Modelle wird folgendes ergänzt:  
# Die Ausprägungen *ISLAND*, *NEAR* *BAY* und *NEAR* *OCEAN* der Variablen *ocean_proximity* wurden in verschiedenen Modellen (Classification & Regression) durch P-Values oder Koeffizienten als unwichtig eingestuft.
# Daher werden diese nun zu einer Ausprägung zusammengefasst.

# In[61]:


train_dataset.ocean_proximity = train_dataset.ocean_proximity.str.replace("ISLAND", "NEAR WATER", regex =True)
train_dataset.ocean_proximity = train_dataset.ocean_proximity.str.replace("NEAR BAY", "NEAR WATER", regex =True)
train_dataset.ocean_proximity = train_dataset.ocean_proximity.str.replace("NEAR OCEAN", "NEAR WATER", regex =True)
train_dataset.ocean_proximity = train_dataset.ocean_proximity.str.replace("<1H OCEAN", "NEAR WATER", regex =True)


# In[62]:


# durch die gerade stattgefundene Transformation muss der datatype wieder korrigiert werden
train_dataset['ocean_proximity'] = train_dataset['ocean_proximity'].astype("category")


# In[63]:


train_dataset['ocean_proximity'].value_counts()


# In[64]:


# Verteilung von ocean_proximity auf Geokoordinaten visualisieren
sns.jointplot(data=train_dataset, x='longitude', y='latitude', hue="ocean_proximity",height=10);


# In[65]:


# Untersuchung der kategroialen Variable "ocean_proximity" mit einem erweiterten Boxplot
sns.boxenplot(data=train_dataset, x="ocean_proximity", y="median_house_value");


# # Modelling
# ---

# ## 1. Linear OLS Regression
# ---

# ### Model 1.1 - Linear OLS Regression mit scikit learn

# #### Modellierung

# Mit den zuvor ausgewählten Variablen testen wir nun das Lineare Regressionsmodell von scikit learn.  
# Um die Daten in das für scikit learn benötigte Format zu bringen, führen wir erneut einen datasplit durch. Da wir bereits vom Anfang noch ein Testdatenset haben, wird bei diesem Split ein Validierungsdatenset erzeugt, mit welchem das Modell erst validiert wird, bevor es dann am Ende mit den Testdaten evaluiert wird.

# In[66]:


# Auswahl der Features und Split der Trainingsdaten in X und Y Varaiblen
features = ['median_income', 'ocean_proximity', 'bedrooms_per_room', 'people_per_household','housing_median_age']
X1 = train_dataset[features]
y1 = train_dataset["median_house_value"]


# In[67]:


# Data Split für Modell Scikitlearn
X_train1, X_val1, y_train1, y_val1 = train_test_split(X1, y1, test_size=0.2, random_state=0)


# In[68]:


# Create pipeline with model
lin_pipe = Pipeline(steps=[
    ('preprocessor', preprocessor),
    ('lin', LinearRegression())
                        ])


# In[69]:


# show pipeline
set_config(display="diagram")
# Fit model
lin_pipe.fit(X_train1, y_train1)


# In[70]:


# Obtain model coefficients
lin_pipe.named_steps['lin'].coef_


# In[71]:


list_numerical = X1.drop(['ocean_proximity'], axis=1).columns


# In[72]:


features_names = np.concatenate((list_numerical.to_numpy(), lin_pipe.named_steps['preprocessor'].transformers_[1][1]['onehot'].get_feature_names_out()))
features_names


# Mit Hilfe der Koeffizienten der ausgewählten Variablen können wir uns die Relevanz grafisch darstellen lassen. So können beim Optimieren des Models gezielt weniger wichtige Features weggelassen werden. So wie aktuell dargestellt haben alle Features zumindest eine gewisse Relevanz. Ein Durchlauf des Models ohne die schwächste Variable *housing_median_age* erzielt kein besseres Ergebnis (Siehe "Evaluation mit Trainingsdaten" unten). Zuvor wurde hier ebenfalls Ausprägungen von *ocean_proximity* als irrelevant dargestellt, was unter anderem Anlass war, im Feature Engineering die Ausprägungen zu *NEAR* *WATER* zusammengefasst wurden.

# In[73]:


# get absolute values of coefficients
importance = np.abs(lin_pipe.named_steps['lin'].coef_)

sns.barplot(x=importance, 
            y=features_names
            );


# #### Evaluation mit Trainingsdaten

# In[74]:


y_pred1train = lin_pipe.predict(X_train1)


# In[75]:


y_pred1train


# In[76]:


print('r2 in %:', r2_score(y_train1, y_pred1train)*100)
print('MSE:', mean_squared_error(y_train1, y_pred1train))
print('RMSE:', mean_squared_error(y_train1, y_pred1train, squared=False))


# Ein Ergebnis von r2 = 62,62% und einem RMSE von 70712 ist kein besonders gutes Ergebnis. Das Modell weicht im Schnitt um 70.712 vom eigentlichen *median_house_value* ab, welcher sich meistens zwischen 100.000 und 500.000 Dollar bewegt. Als Versuch wurde das schwächste Feature *housing_median_age* entfernt, das Ergebnis fiel dabei leicht schlechter aus:  
# 
# Modell ohne *housing_median_age*:  
# - r2 in %: 61.742949278293366  
# - MSE: 5117358064.557615  
# - RMSE: 71535.71181275556  
# 

# #### Validierung mit Testdaten

# Nun wird das Modell mit dem Validation dataset validiert.

# In[77]:


y_pred1val = lin_pipe.predict(X_val1)


# In[78]:


print('r2 in %:', r2_score(y_val1, y_pred1val)*100)
print('MSE:', mean_squared_error(y_val1, y_pred1val))
print('RMSE:', mean_squared_error(y_val1, y_pred1val, squared=False))


# Das Ergebnis der Validierung fällt noch schlechter aus.
# Das Modell ist also nicht gut genug gefittet, also underfitted. Nun kann entweder das Modell angepasst werden (bspw. andere Variablen gewählt werden) oder ein anderes Modell getestet werden um auf ein besseres Ergebnis zu kommen. Da nach verschiedenen Tests kein bessere Kombination an Features gefunden werden konnte, werden zunächst andere Modelle getestet.

# ### Model 1.2 - Linear OLS Regression with Statsmodels

# #### Modellierung

# Für das Linear Regression Modell von statsmodels ist kein erneuter datasplit notwendig, das Modell kann mit dem train_dataset arbeiten, welches X und Y Variablen enthält. Auf die seperate Erstellung eines Validation Datasets wird hier verzichtet, das Modell wird lediglich mit den Traingsdaten und schließlich mit den Testdaten evaluiert.

# In[79]:



lm1 = smf.ols(formula='median_house_value ~ median_income + ocean_proximity + bedrooms_per_room + people_per_household + housing_median_age', data=train_dataset).fit()


# In[80]:


# Short summary
lm1.summary().tables[1]


# In[81]:


# Full summary
lm1.summary()


# In[82]:


print("RMSE", np.sqrt(lm1.mse_resid))


# Das Modell weißt schon einen ähnlichen r2 value auf wie das scikit learn Modell. Die gewählten Features scheinen sinnvoll gewählt zu sein, da keine einen P-Value von größer 0,05 besitzt. Damit können wir die Null Hypothese verwerfen und davon ausgehen, dass die Features Einfluss auf unsere Y Variable haben.  
#  
# Mehrere Tests ergeben, dass das Löschen von Variablen (getestet wurde ohne *housing_median_age* und *people_per_household*) negativen Einfluss auf den Erfolg des Modells hat.  
# 
# Bevor die Ergebnisse weiter interpretiert werden, soll zunächste ein Optimierung durch die Entfernung von Outliern stattfinden. Die Outlier werden durch die Methode "Cook's Distance" identifiziert.

# In[83]:


# Visualisierung der Outlier nach Cook's Distance
fig = sm.graphics.influence_plot(lm1, criterion="cooks")
fig.tight_layout(pad=1.0)


# In[84]:


# Berechnen der Outlier nach Cook's Distance
# obtain Cook's distance 
lm1_cooksd = lm1.get_influence().cooks_distance[0]

# get length of df to obtain n
n = len(train_dataset["median_income"])

# calculate critical d
critical_d = 4/n
print('Critical Cooks distance:', critical_d)

# identification of potential outliers with leverage
out_d = lm1_cooksd > critical_d

# output potential outliers with leverage
print(train_dataset.index[out_d], "\n", 
    lm1_cooksd[out_d])


# Um das Trainingsdatenset noch für weitere Modelle verwenden zu können ohne es neu erzeugen zu müssen, wird an dieser Stelle ein neues Datenset erstellt.

# In[85]:


# droppen der Outlier Observations
train_dataset2=train_dataset.drop(train_dataset.index[out_d])


# Nun wird der Algorithmus erneut durchgeführt, auf Basis des bereinigten datasets.

# In[86]:


lm2 = smf.ols(formula='median_house_value ~ median_income + ocean_proximity + bedrooms_per_room + people_per_household + housing_median_age', data=train_dataset2).fit()


# In[87]:


lm2.summary()


# In[88]:


print("RMSE", np.sqrt(lm2.mse_resid))


# Folgende Erkenntnisse lassen sich hier gewinnen:
# - Der r2 hat sich um 12% verbessert. 74,2% Abweichung kann durch das Modell erklärt werden.
# - F-Statistic gibt uns einen Überblick über das Modell im Vehältnis zum Fehler im Modell ( Verhältnis von "wie sehr hat das Modell die Vorhersage des Ergebnisses verbessert vs. Abweichungen/Fehler des Modells). Wenn die Vebesserung überwiegt, entsteht ein großer F Wert, was hier der Fall ist.
# - AIC und BIC haben sich auch verbessert nach der Entferung der Outlier.
# - Die Normalverteilung der residuals kann über Omnibus und Jarque-Bera analysiert werden, ist in diesem Fall mit einem Datensatz mit weit mehr als 50 Observations allerdings zu vernachlässigen.
# - Aus einem Durbin-Watson Wert nahe 2 kann man erkennen, dass keine autocorrelation vorliegt.
# - der RMSE hat sich auch verbessert und ist um einiges besser als der RMSE im scikit learn Modell

# ### Regression Diagnostics

# In diesem Abschnitt soll nun noch einmal verstärkt die Auswirkung der einzelnen Features auf unser Modell sowie mögliche Outlier untersucht werden.

# In[89]:


fig = sm.graphics.plot_partregress_grid(lm2)
fig.tight_layout(pad=1.0)


# Auf dem Partial Regression Plot können wir jeweils einen mehr oder weniger linearen Zusammenhang der Features auf die Y Variable erkennen.

# In[90]:


sns.lmplot(x='median_income', y='median_house_value', data=train_dataset2, 
line_kws={'color': 'darkred'}, ci=False);


# In[91]:


sns.lmplot(x='bedrooms_per_room', y='median_house_value', data=train_dataset2, 
line_kws={'color': 'darkred'}, ci=False);


# Da bereits im Feature Engineering sowie durch Cook's Distance starke Outlier entfernt wurden, sind hier keine Outlier mehr mit bspw. besonders starkem Hebel zu erkennen.

# In[92]:


y_pred2 = lm2.predict(train_dataset2)


# In[93]:


sns.residplot(x=y_pred2, y="median_house_value", data=train_dataset2, scatter_kws={"s": 80});


# Im residplot sollten im Optimalfall die residuals einigermaßen gleichmäßig zufällig um die horizontale Nulllinie verteilt sein.
# Da dies nicht ganz der Fall ist, könnte es auf heteroscedasticity hindeuten. Das bedeutet, dass die Error im Modell nicht gleichmäßig verteilt sind.
# 
# Um das zu überprüfen, wird der Breusch-Pagan Lagrange Multiplier Test angewendet. Die Null Hypothese hierbei ist homoscedasticity.

# In[94]:


name = ['Lagrange multiplier statistic', 'p-value', 'f-value', 'f p-value']
test = sm.stats.het_breuschpagan(lm2.resid, lm2.model.exog)
lzip(name, test)


# Zwei der drei P-Values liegen unter 0,05, was auf heteroscedasticity hindeutet.

# Im Detail lassen sich auch noch einzelne Variablen untersuchen, was hier lediglich an der einflussreichsten Variablen *median_income* durchgeführt werden soll. Die folgende Funktion liefert mehrere Plots für eine Variable:
# - Y and Fitted vs. X: Die auf Basis von *median_income* gefitteten Werte bewegen sich in einem ähnlichen Bereich wie unsere Y Variable
# - Partial Regression & CCPR Plot deuten auf ein lineares Verhältnis von *median_income* auf Y hin

# In[95]:


# Regression diagnostics für Variable "median_income"
fig = sm.graphics.plot_regress_exog(lm2, "median_income")
fig.tight_layout(pad=0.2)


# ## 2. Lasso Regression mit scikit learn
# ---

# Für das Lasso Modell muss wieder nach scikit learn Logik ein datasplit durchgeführt werden.

# #### Lasso - Split Data

# In[96]:


# Erstellen der X und Y Variablen
y2 = train_dataset2['median_house_value']
features = ['median_income', 'people_per_household', 'ocean_proximity', 'bedrooms_per_room', 'housing_median_age']
X2 = train_dataset2[features]


# In[97]:


# Data split
X_train2, X_val2, y_train2, y_val2 = train_test_split(X2, y2, test_size=0.3, random_state=0)


# #### Lasso - Model

# Bei der ersten Durchführung des Modells setzen wir den Alpha Wert standardmäßig auf 1. Dieser wird später noch optimiert.

# In[98]:



# Erstellen der Pipeline mit Lasso Modell
lasso_pipe = Pipeline(steps=[
    ('preprocessor', preprocessor),
    ('lasso', Lasso(alpha=1))
                        ])


# In[99]:


# Fitten von Pipeline/Modell
lasso_pipe.fit(X_train2, y_train2)


# Um die Importance der einzelnen Features im Lasso Algorithmus zu visualisieren, wird an dieser Stelle eine Liste mit allen Feature Namen erstellt.

# In[100]:


# Erstellen einer Liste der numerical features
list_numerical = X2.drop(['ocean_proximity'], axis=1).columns
list_numerical


# In[101]:


# Erstellen einer Liste aller Feature Namen
feature_names = np.concatenate((list_numerical.to_numpy(), lasso_pipe.named_steps['preprocessor'].transformers_[1][1]['onehot'].get_feature_names_out()))
feature_names


# In[102]:


# get absolute values of coefficients
importance = np.abs(lasso_pipe.named_steps['lasso'].coef_)

sns.barplot(x=importance, 
            y=feature_names);


# #### Lasso - Model Evaluation

# Bevor es an die Optimierung geht, soll zuerst das Modell noch mit dem Wert alpha = 1 evaluiert werden.

# In[103]:


pred_train = lasso_pipe.predict(X_train2)
pred_val = lasso_pipe.predict(X_val2)


# In[104]:


print('R squared training set', round(lasso_pipe.score(X_train2, y_train2)*100, 2))
print('R squared validation set', round(lasso_pipe.score(X_val2, y_val2)*100, 2))
print('RMSE train set', round(mean_squared_error(y_train2, pred_train, squared=False),2))
print('RMSE validation set', round(mean_squared_error(y_val2, pred_val, squared=False),2))


# In[105]:


# Training data
pred_train = lasso_pipe.predict(X_train2)
mse_train = mean_squared_error(y_train2, pred_train)
print('MSE training set', round(mse_train, 2))

# Validation data
pred_test = lasso_pipe.predict(X_val2)
mse_test =mean_squared_error(y_val2, pred_val)
print('MSE validation set', round(mse_test, 2))


# #### Lasso - k-fold cross validation

# Nun soll mit Hilfe der k-fold Cross Validation Methode der optimale Wert für alpha ermittelt werden, um unser Modell weiter zu optimieren.

# In[106]:


# Erstellen von Modell mit Pipeline
lassoCV_pipe = Pipeline(steps=[
    ('preprocessor', preprocessor),
    ('lassoCV', LassoCV(cv=5, random_state=0, max_iter=10000))
                        ])


# In[107]:


# Fit model
lassoCV_pipe.fit(X_train2, y_train2)


# In[108]:


lassoCV_pipe.named_steps['lassoCV'].alpha_


# #### Lasso - Lasso Best

# Der ermittelte Optimalwert für alpha wird nun in das Modell eingesetzt.

# In[109]:


# Create pipeline with model
lassobest_pipe = Pipeline(steps=[
    ('preprocessor', preprocessor),
    ('lassobest', Lasso(alpha=lassoCV_pipe.named_steps['lassoCV'].alpha_))
                            ])


# In[110]:


# Set best alpha

lassobest_pipe.fit(X_train2, y_train2)


# Zur Berechnung des RMSE werden auch mit dem Modell lassobest_pipe Predictions berechnet.
# 

# In[111]:


pred_trainbest = lassobest_pipe.predict(X_train2)
pred_valbest = lassobest_pipe.predict(X_val2)


# In[112]:


#Werte mit optimalem alpha:
print('mit lasso best:')
print('R squared training set ', round(lassobest_pipe.score(X_train2, y_train2)*100, 2))
print('R squared validation set ', round(lassobest_pipe.score(X_val2, y_val2)*100, 2))
print('RMSE train set', round(mean_squared_error(y_train2, pred_trainbest, squared=False),2))
print('RMSE validation set', round(mean_squared_error(y_val2, pred_valbest, squared=False),2))


# In[113]:


# Werte ohne optimales alpha:
print('ohne Lasso best:')
print('R squared training set lasso ', round(lasso_pipe.score(X_train2, y_train2)*100, 2))
print('R squared validation set lasso ', round(lasso_pipe.score(X_val2, y_val2)*100, 2))
print('RMSE train set', round(mean_squared_error(y_train2, pred_train, squared=False),2))
print('RMSE validation set', round(mean_squared_error(y_val2, pred_val, squared=False),2))


# >**Fazit**: Durch die best alpha Optimierung hat sich das Ergebnis unseres Modells nicht verbessert, lediglich der RMSE ist kaum merklich besser geworden. Ein r2 von ca. 73% und ein RMSE von ca 53.000 ist ähnlich gut wie das Linear Regression Modell mit statsmodels.

# ## 3. Splines
# ---

# ### 3.1 Splines mit scikit learn

# Für das scikit leran Modell wird wieder der übliche datasplit durchgeführt.

# In[114]:


y3 = train_dataset[['median_house_value']]
X3 = train_dataset[['median_income', 'ocean_proximity', 'bedrooms_per_room', 'people_per_household', 'housing_median_age']]


# In[115]:


# data split
X_train3, X_val3, y_train3, y_val3 = train_test_split(X3, y3, test_size=0.3, random_state=0)


# Für das Modell definieren wir die Anzahl der zu platzierenden Knots auf 4. Die Komponente SplineTransformer wird diese Knots dann automatisch platzieren.

# In[116]:


# Erstellen des Models mit Pipeline. Der Spline Transformer soll 4 Knots platzieren.
splines_pipe = Pipeline(steps=[
    ('preprocessor', preprocessor),
    ('splines', make_pipeline(SplineTransformer(n_knots=4, degree=3), 
                       Ridge(alpha=1)))
                        ])


# In[117]:


splines_pipe.fit(X_train3, y_train3)

y_pred = splines_pipe.predict(X_train3)


# Nach dem Erstellen und fitten des Modells wird eine Funktion definiert, um den RMSE auszugeben.

# In[118]:


# Erstellen der Funktion model_results um KPIs auszugeben
def model_results(model_name):

    # Training data
    pred_train = splines_pipe.predict(X_train3)
    rmse_train = round(mean_squared_error(y_train3, pred_train, squared=False),4)

    # Test data
    pred_val = splines_pipe.predict(X_val3)
    rmse_val =round(mean_squared_error(y_val3, pred_val, squared=False),4)

    # Print model results
    result = pd.DataFrame(
        {"model": model_name, 
        "rmse_train": [rmse_train], 
        "rmse_val": [rmse_val]}
        )
    
    return result;


# In[119]:


model_results(model_name = "spline sklearn")


# In[120]:


print('R squared training set', round(splines_pipe.score(X_train3, y_train3)*100, 2))
print('R squared validation set', round(splines_pipe.score(X_val3, y_val3)*100, 2))


# Sowohl der r2 Wert als auch der RMSE des Modells liegen hinter dem der vorherigen zwei Modelle.

# In[121]:


# Create observations
x_new = np.linspace(X_val3.min(),X_val3.max(), 100)
x_new = pd.DataFrame(x_new, columns=X3.drop(["ocean_proximity"], axis=1).columns)
x_new = x_new.assign(ocean_proximity=lambda x_new: "dummy")


# In[122]:


pred = splines_pipe.predict((x_new))

sns.scatterplot(x=X_train3['median_income'], y=y_train3["median_house_value"])

plt.plot(x_new["median_income"], pred, label='Cubic spline with degree=3', color='orange')
plt.legend();


# #### Test mit Cook's Distance bereinigten Dataset

# Da dieses Splines Modell das vielversprechendste ist, wird das Modell als Test noch einmal mit den per Cook's Distance bereinigten Trainingsdaten gefittet und evaluiert.

# In[123]:


y5 = train_dataset2[['median_house_value']]
X5 = train_dataset2[['median_income', 'ocean_proximity', 'bedrooms_per_room', 'people_per_household', 'housing_median_age']]


# In[124]:


# data split
X_train5, X_val5, y_train5, y_val5 = train_test_split(X5, y5, test_size=0.3, random_state=0)


# In[125]:


# Erstellen des Models mit Pipeline. Der Spline Transformer soll 4 Knots platzieren.
splinesCD_pipe = Pipeline(steps=[
    ('preprocessor', preprocessor),
    ('splines', make_pipeline(SplineTransformer(n_knots=4, degree=3), 
                       Ridge(alpha=1)))
                        ])


# In[126]:


splinesCD_pipe.fit(X_train5, y_train5)

y_pred = splinesCD_pipe.predict(X_train5)


# In[127]:


# Erstellen der Funktion model_results um KPIs auszugeben
def model_results(model_name):

    # Training data
    pred_train = splinesCD_pipe.predict(X_train5)
    rmse_train = round(mean_squared_error(y_train5, pred_train, squared=False),4)

    # Test data
    pred_val = splinesCD_pipe.predict(X_val5)
    rmse_val =round(mean_squared_error(y_val5, pred_val, squared=False),4)

    # Print model results
    result = pd.DataFrame(
        {"model": model_name, 
        "rmse_train": [rmse_train], 
        "rmse_val": [rmse_val]}
        )
    
    return result;


# In[128]:


model_results(model_name = "spline sklearn")


# In[129]:


print('R squared training set', round(splinesCD_pipe.score(X_train5, y_train5)*100, 2))
print('R squared validation set', round(splinesCD_pipe.score(X_val5, y_val5)*100, 2))


# Das Resultat ist um einiges besser als im vorherigen Versuch. Ob das Modell mit den originalen Testdaten auch so gut abschneidet muss allerdings erst evaluiert werden.

# ### 3.2 Cubic Spline with Patsy & Statsmodels

# In[130]:


y4 = train_dataset[['median_house_value']]
X4 = train_dataset[['median_income']]


# In[131]:


# data split
X_train4, X_val4, y_train4, y_val4 = train_test_split(X4, y4, test_size=0.3, random_state=0)


# In[132]:


# Erstellen eines cubic spline mit 3 manuell platzierten knots bei 1, 4 und 7
transformed_x = dmatrix(
            "bs(train, knots=(1,4,7), degree=3, include_intercept=False)", 
                {"train": X_train4},return_type='dataframe')


# In[133]:


# Fitten des linearen models an das transformierte dataset
spline2 = sm.GLM(y_train4, transformed_x).fit()


# In[134]:


# Training data
pred_train = spline2.predict(dmatrix("bs(train, knots=(1,4,7), include_intercept=False)", {"train": X_train4}, return_type='dataframe'))
rmse_train = mean_squared_error(y_train4, pred_train, squared=False)

# Validation data
pred_val = spline2.predict(dmatrix("bs(val, knots=(1,4,7), include_intercept=False)", {"val": X_val4}, return_type='dataframe'))
rmse_val =mean_squared_error(y_val4, pred_val, squared=False)

# Save model results
model_results = pd.DataFrame(
    {
    "model": "Cubic spline",  
    "rmse_train": [rmse_train], 
    "rmse_val": [rmse_val]
    })

model_results


# In[135]:


print('R squared training set', round(r2_score(y_train4, pred_train)*100, 2))
print('R squared validation set', round(r2_score(y_val4, pred_val)*100, 2))


# In diesem Modell ist sowohl der r2 Wert als auch der RMSE sehr viel schlechter als in den übrigen Modellen. Dies überrascht nicht, da in diesem Modell nur eine X Variable zur Vorhersage verwendet wurde.

# In[136]:


# Create observations
xp = np.linspace(X_val4.min(),X_val4.max(), 100)
# Make some predictions
pred = spline2.predict(dmatrix("bs(xp, knots=(1,4,7), include_intercept=False)", {"xp": xp}, return_type='dataframe'))

# plot
sns.scatterplot(x=X_train4['median_income'], y=y_train4['median_house_value'])

plt.plot(xp, pred, label='Cubic spline with degree=3 (3 knots)', color='orange')
plt.legend();


# ### 3.3 Natural Spline with Patsy & Statsmodels

# In[137]:


transformed_x3 = dmatrix("cr(train,df = 3)", {"train": X_train4}, return_type='dataframe')

spline3 = sm.GLM(y_train3, transformed_x3).fit()


# In[138]:


# Training data
pred_train = spline3.predict(dmatrix("cr(train, df=3)", {"train": X_train4}, return_type='dataframe'))
rmse_train = mean_squared_error(y_train4, pred_train, squared=False)

# Validation data
pred_val = spline3.predict(dmatrix("cr(val, df=3)", {"val": X_val4}, return_type='dataframe'))
rmse_val = mean_squared_error(y_val4, pred_val, squared=False)

# Save model results
model_results_ns = pd.DataFrame(
    {
    "model": "Natural spline (ns)",  
    "rmse_train": [rmse_train], 
    "rmse_val": [rmse_val]
    })

model_results_ns


# In[139]:


print('R squared training set', round(r2_score(y_train4, pred_train)*100, 2))
print('R squared test set', round(r2_score(y_val4, pred_val)*100, 2))


# Auch dieses Modell liefert keine guten Werte.

# In[140]:


# Make predictions
pred = spline3.predict(dmatrix("cr(xp, df=3)", {"xp": xp}, return_type='dataframe'))
xp = np.linspace(X_val4.min(),X_val4.max(), 100)
# plot
sns.scatterplot(x=X_train4['median_income'], y=y_train4['median_house_value'])
plt.plot(xp, pred, color='orange', label='Natural spline')
plt.legend();


# # Abschließende Evaluation mit Testdaten

# ## Transformation Testdaten

# Um die Testdaten verwenden zu können, müssen die gleichen Schritte zur Transformation durchgeführt werden wie mit den Traindaten (Anpassung Datentypen, droppen NULL values, Erzeugen von selbst erstellten Features).

# In[141]:


test_dataset


# In[142]:


# droppen von NULL Values
test_dataset = test_dataset.dropna()


# In[143]:


# nicht benötigte Variable droppen
test_dataset = test_dataset.drop(columns=['price_category'])


# In[144]:


# Anpassung der datatypes
test_dataset['median_house_value'] = test_dataset['median_house_value'].astype("float64")
test_dataset['housing_median_age'] = test_dataset['housing_median_age'].astype("float64")
test_dataset['total_bedrooms'] = test_dataset['total_bedrooms'].astype("int64")
test_dataset['ocean_proximity'] = test_dataset['ocean_proximity'].astype("category")


# In[145]:


# erzeugen der selbst erstellten Features
test_dataset=test_dataset.assign(people_per_household=lambda test_dataset: test_dataset.population/test_dataset.households)
test_dataset=test_dataset.assign(bedrooms_per_household=lambda test_dataset: test_dataset.total_bedrooms/test_dataset.households)
test_dataset=test_dataset.assign(rooms_per_household=lambda test_dataset: test_dataset.total_rooms/test_dataset.households)
test_dataset=test_dataset.assign(bedrooms_per_room=lambda test_dataset: test_dataset.total_bedrooms/test_dataset.total_rooms)


# In[146]:


# Zusammenfassung des Features ocean_proximity
test_dataset.ocean_proximity = test_dataset.ocean_proximity.str.replace("ISLAND", "NEAR WATER", regex =True)
test_dataset.ocean_proximity = test_dataset.ocean_proximity.str.replace("NEAR BAY", "NEAR WATER", regex =True)
test_dataset.ocean_proximity = test_dataset.ocean_proximity.str.replace("NEAR OCEAN", "NEAR WATER", regex =True)
test_dataset.ocean_proximity = test_dataset.ocean_proximity.str.replace("<1H OCEAN", "NEAR WATER", regex =True)


# In[147]:


# Korrektur des datatype von ocean_proximity
test_dataset['ocean_proximity'] = test_dataset['ocean_proximity'].astype("category")


# ## Test Data Split

# Um die Testdaten in der Evaluation für bestimmte Modelle verwenden zu können, müssen diese nach Features und vorherzusagender Varaible aufgeteilt werden.

# In[148]:


y = test_dataset['median_house_value']
features = ['median_income', 'people_per_household', 'ocean_proximity', 'bedrooms_per_room', 'housing_median_age']
X = test_dataset[features]


# ## Finale Evaluation

# Zur finalen Evaluation mit den Testdaten werden die besten Modelle ausgewählt:
# 1. Lineare Regression mit statsmodels
# 2. Lasso Regression mit scikit learn
# 3. Splines mit scikit learn

# ### Evaluation Lineares Regressionsmodell mit statsmodels

# In[149]:


pred_test = lm2.predict(X)


# In[150]:


print('R squared test set', round(r2_score(y, pred_test)*100, 2))
print('RMSE test set', round(mean_squared_error(y, pred_test, squared=False),2))


# Die Ergebnisse mit den Testdaten sind schlechter als die Resultate mit den bearbeiteten Trainingsdaten. Dies liegt vermutlich am Entfernen vieler Outlier durch Cook's Distance.

# ### Evaluation Lasso Regression mit scikit learn

# In[151]:


pred_testfinal = lassobest_pipe.predict(X)


# In[152]:


print('Testergebnis mit lasso best:')
print('R squared test set ', round(lassobest_pipe.score(X, y)*100, 2))
print('RMSE test set', round(mean_squared_error(y, pred_testfinal, squared=False),2))


# Das Ergebnis ist sehr schlecht, das liegt vermutlich an dem datenset mit entfernten Outliern nach Cook's Distance. Das mit dem "originalen" train_dataset gefittete Modell hätte vermutlich besser Abgeschnitten.

# ### Evaluation Splines mit scikit learn

# In[153]:


y_pred = splines_pipe.predict(X)


# In[154]:


# Erstellen der Funktion model_results um KPIs auszugeben
def model_results(model_name):

    # Training data
    pred_train = splinesCD_pipe.predict(X_train5)
    rmse_train = round(mean_squared_error(y_train5, pred_train, squared=False),4)

    # Test data
    pred_val = splinesCD_pipe.predict(X)
    rmse_val =round(mean_squared_error(y, pred_val, squared=False),4)

    # Print model results
    result = pd.DataFrame(
        {"model": model_name, 
        "rmse_train": [rmse_train], 
        "rmse_test": [rmse_val]}
        )
    
    return result;


# In[155]:


model_results(model_name = "spline sklearn")


# In[156]:


print('R squared test set', round(splinesCD_pipe.score(X, y)*100, 2))


# Das Modell, welches mit dem Cook's Distance optimierten Traindataset gefitted wurde, erzielt in der Evaluation mit den Testdaten leicht bessere Ergebnisse.
# 
# Zum Vergleich die KPIs ohne Cook's Distance Optimierung:  
# - RMSE: 66017.6385  
# - r2: 66.57  

# >**Fazit**: In der Evaluation der Testdaten schneidet das Splines Modell von scikit learn am besten ab.
