#!/usr/bin/env python
# coding: utf-8

# In[1]:


import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import scipy

from functions import *
import statsmodels.api as sm
import statsmodels.formula.api as smf
from fanalysis.pca import PCA
from sklearn.cluster import KMeans
from sklearn import preprocessing, decomposition
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import confusion_matrix, precision_score, recall_score, plot_confusion_matrix
from sklearn.datasets import make_classification
from sklearn.model_selection import train_test_split
from sklearn.svm import SVC
from scipy.spatial.distance import cdist
from sklearn import model_selection

data = pd.read_csv("notes_billets.csv")


# # SOMMAIRE
# 
# ## [MISSION 0 - Description des données](#ch0)
# 
# ## [MISSION 1 - ACP](#ch1)
# 
# ## [MISSION 2 - Application d'un algorithme de classification](#ch2)
# 
# ## [MISSION 3 - Régression logistique](#ch3)

# <a id="ch0"></a>
# # MISSION 0 : Description des données

#   

# In[2]:


data


# In[3]:


# Vérification des NaN
data[data.isna().any(axis=1)]


# Pas de NaN dans le fichier

# In[4]:


data.info()


# In[5]:


data.describe()


# Je n'observe pas de valeur aberrante.

# In[6]:


# Vérification des valeurs de la colonne booléenne
print(data.is_genuine.unique())
print(data.is_genuine.value_counts())


# Il s'agit bien uniquement de Booléens.  
# Il y a environ 3/5 de valeurs vraies et 2/5 de valeurs fausses.
#   
# Les données n'ont pas à être nettoyées

# In[7]:


# Analyse univariée
print(data.median())
print(data.mode())


# In[8]:


# Pairplot
sns.pairplot(data, hue="is_genuine")
plt.savefig("images_soutenance/01_pairplot.jpeg", format="jpeg")
# Affichage d'une heatmap
plt.figure(figsize=(16,8))
sns.heatmap(data.corr(), annot=True)
plt.savefig("images_soutenance/01-2_matrice_correl.jpeg", format="jpeg")
plt.show()


# On observe une corrélation nette entre la hauteur du côté droit et la hauteur du côté gauche (height_right et height_left), ce qui n'est pas surprenant, et pas nécessairement utile pour notre analyse de validité des billets.
# On peut songer à regrouper ces deux variables, car il y a redondance.  
# 
# On observe une séparation nette en deux grands groupes de points entre la longueur des billets (length) et trois variables : la marge du bas (margin_low), la hauteur du côté gauche et la hauteur du côté droit. De même entre la marge en haut du billet et la marge en bas du billet (margin_up et margin_low).  
# 
# La marge basse du billet est corrélée très négativement à sa validité, et présente un caractère discriminant très intéressant. La longueur du billet est quant à elle corrélée très positivement à la validité du billet : ces deux variables nous semblent les plus intéressantes sur lesquelles nous pencher.  
# 
# On cherche à préciser ces intuitions avec un boxplot.

# In[9]:


# Boxplots

meanprops = {'marker':'o', 'markeredgecolor':'black',
            'markerfacecolor':'firebrick', "markersize":'7'}

for columns in data.set_index("is_genuine") : 
    plt.figure(figsize=(20,5))
    sns.boxplot(x=columns, y="is_genuine", data=data, showmeans=True, meanprops=meanprops, orient="h")
    plt.title('Variation de {}'.format(columns), fontsize='23')
    plt.xlabel('Longueur (cm)', fontsize='18')
    plt.ylabel("Validité du billet", fontsize='18')
    plt.xticks(fontsize=13)
    plt.yticks(fontsize=13)
    plt.savefig("images_soutenance/02_boxplot{}.jpeg".format(columns), format="jpeg")


# Par observation graphique, les variables qui permettent le mieux de distinguer les billets vrais des billets faux sont les suivantes : margin_low et length de manière très prononcée ; height_left, height_right et margin_up de manière moins prononcée. La variable diagonal ne permet de pas distinguer les billets vrais des billets faux.  
# 
# Cela confirme nos observations précédentes.

# <a id="ch1"></a>
# # MISSION 1 : ACP

# J'utilise la bibliothèque scikit-learn pour afficher le cercle des corrélations et le nuage des individus, et la bibliothèque fanalysis pour l'éboulis des valeurs propres, la qualité de représentation des variables et des individus.  
# Cela parce que fanalysis ne sait pas afficher de couleurs différentes selon le nom de l'individu, ce qui empêche de comprendre visuellement clairement le nuage.

# In[10]:


# Création d'une ACP avec fanalysis

acp = PCA(std_unit=True, row_labels=data.is_genuine.values, 
          col_labels=data.set_index("is_genuine").columns.values[0:6]) #std_unit = True ➔ ACP normée

# Calculs sur les données
selected_variables = data.set_index("is_genuine").columns
X = data[selected_variables].values
acp.fit(X)

# Propriétés de l'objet généré
print(dir(acp))


# In[11]:


# Préparation du graphique
fig, ax = plt.subplots(figsize=(5,5))
ax.plot(range(1,7),acp.eig_[0],".-")
ax.set_xlabel("Nb. facteurs")
ax.set_ylabel("Val. propres")
plt.title("Eboulis des valeurs propres")

#rajout du seuil du Kaiser
ax.plot([1,6],[1,1],"r--",linewidth=1)

plt.savefig("images_soutenance/03_eboulis.jpeg", format="jpeg")


# Le seuil de Kaiser nous indique qu'il faut prendre en compte les trois premiers axes factoriels.  
# Je choisis d'afficher le deuxième plan factoriel en entier, avec F4, pour plus de clarté, mais j'y étudierai seulement F3, soit l'axe des abscisses.

#   

# In[12]:


n_comp = 6 # calcul des 6 composantes pour l'ACP, de manière standard

X = data[selected_variables].values

# préparation des données pour l'ACP
features = selected_variables

# Centrage et Réduction
std_scale = preprocessing.StandardScaler().fit(X)
X_scaled = std_scale.transform(X)

# Calcul des composantes principales
pca = decomposition.PCA(n_components=n_comp)
pca.fit(X_scaled)


# In[13]:


# Cercle des corrélations
pcs = pca.components_
display_circles(pcs, n_comp, pca, [(0,1),[2,3]], labels = np.array(features)) # Je n'affiche que les deux
                                                                              # premiers plans (seuil de Kaiser)
# Projection des individus
X_projected = pca.transform(X_scaled)
display_factorial_planes(X_projected, n_comp, pca, [(0,1),(2,3)], illustrative_var = data.is_genuine)


# F1/F2
# 
# F1 synthétise près de 50% de l'information et le plan factoriel F1/F2 synthétise 70% de l'information. On s'intéresse donc essentiellement à F1, et à ce premier plan factoriel. On analysera F2 en complément seulement.
# 
# Les variables bien correctement représentées sur F1 sont la longueur (length), les deux hauteurs (height_left et height_right) et la marge basse (margin_low). On peut éliminer la diagonale, très mal représentée, et la marge haut (margin_up), mal représentée aussi (à moins de 0,35).
# Sur F2, seule la variable "diagonale" est bien représentée. Ce qui, compte tenu des observations précédentes, nous indique le peu d'intérêt de F2 pour discriminer les billets Vrais des billets Faux.
# 
# De plus, la représentation des individus fait apparaître de manière nette une séparation entre les Vrais billets et les Faux billets, sur F1, et pas du tout sur F2. Quelques individus néanmoins échappent à la règle et demandent de pousser plus loin l'analyse.  
#   
# Je fais la supposition que les variables non discriminantes gênent l'analyse, il conviendrait de les supprimer dans une nouvelle ACP.
# 
# F3/F4
# 
# Il n'est guère besoin d'observer le deuxième cercle des corrélations quand on observe le deuxième plan factoriel, qui mélange absolument les individus correspondant à des billets vrais et les individus correspondant à des billets faux.  
# On n'utilisera pas ce deuxième plan factoriel.

# In[14]:


# Vérification du fonctionnement de l'ACP avec le test du cos2

# Qualité de la représentation des variables : cos2
data_cos_var = acp.col_cos2_
data_cos_var = pd.DataFrame(data_cos_var, index = data.set_index("is_genuine").columns,
                           columns = ["F1", "F2", "F3", "F4", "F5", "F6"])
cos_var_plan_factoriel = data_cos_var[['F1', 'F2']]
cos_var_plan_factoriel["F1_F2"] = cos_var_plan_factoriel['F1'] + cos_var_plan_factoriel['F2']

cos_var_plan_factoriel


# Qualité de représentation des variables :  Une variable est mal représentée sur l'ensemble du premier plan factoriel : margin_up.

# In[15]:


# Qualité de la représentation des individus : cos2
data_cos_indiv = acp.row_cos2_
data_cos_indiv = pd.DataFrame(data_cos_indiv, columns = ["F1", "F2", "F3", "F4", "F5", "F6"])
cos_indiv_plan_factoriel = data_cos_indiv[['F1', 'F2']]
cos_indiv_plan_factoriel["F1_F2"] = cos_indiv_plan_factoriel['F1'] + cos_indiv_plan_factoriel['F2']

cos_indiv_plan_factoriel.F1_F2.describe()


# Pas de valeurs aberrantes a priori.

# In[16]:


# Contribution des individus au premier plan
cos_indiv_plan_factoriel['F1_F2_pourcentage'] = (cos_indiv_plan_factoriel.F1_F2
                                                 /cos_indiv_plan_factoriel.F1_F2.sum())*100
cos_indiv_plan_factoriel.F1_F2_pourcentage


# In[17]:


# Recherche d'individus trop mal ou trop bien représentés
indiv_atypiques = cos_indiv_plan_factoriel.loc[(cos_indiv_plan_factoriel.F1_F2_pourcentage>1.5) |
                            (cos_indiv_plan_factoriel.F1_F2_pourcentage<0.2)].F1_F2_pourcentage
print(indiv_atypiques)
data.iloc[indiv_atypiques.index]


# Les seules valeurs atypiques sont celles qui sont mal représentées sur F1/F2. Pas de valeur atypique trop bien représentée.
# 7 individus ont des valeurs atypiques, mais qui ne sont pas significativement différentes des autres. Après observation de leurs caractéristiques, il n'est pas nécessaire de les supprimer.

# <a id="ch2"></a>
# # MISSION 2 : Application d'un algorithme de classification

#   

# Je choisis d'appliquer la méthode du K-means plutôt que le dendrogramme, car ici on connaît le nombre de clusters que l'on recherche (2 clusters, 1 pour les billets vrais et 1 pour les billets faux).  
# Le K-means permet d'obtenir immédiatement des informations sur ces clusters, notamment en faisant apparaître le centre de gravité de chaque groupe.

# In[18]:


# Méthode du coude, pour vérifier le nombre de clusters pertinent

X = X_scaled.copy()

distortions = []
K = range(1,10)
for k in K:
    kmeanModel = KMeans(n_clusters=k).fit(X)
    kmeanModel.fit(X)
    distortions.append(sum(np.min(cdist(X, kmeanModel.cluster_centers_, 'euclidean'), axis=1)) / X.shape[0])

# Représentation graphique
plt.plot(K, distortions, 'bx-')
plt.xlabel('k')
plt.ylabel('Distortion')
plt.title('The Elbow Method showing the optimal k')

plt.savefig("images_soutenance/05_elbow.jpeg", format="jpeg")


# On observe un coude net pour k=2, et un coude moins net pour k=3.  
# On peut donc a priori choisir entre deux ou trois clusters.

# In[19]:


# Application de l'algorithme K-means

# Nombre de clusters souhaités
n_clust = 2

# Réduire n'est ici pas nécessaire car les variables sont exprimées dans la même unité

# Clustering par K-means
km = KMeans(n_clusters=n_clust)
km.fit(X)

# Récupération des clusters attribués à chaque individu
clusters = km.labels_

# Affichage du clustering par projection des individus sur le premier plan factoriel
pca = decomposition.PCA(n_components=3).fit(X)
X_projected = pca.transform(X)
plt.scatter(X_projected[:, 0], X_projected[:, 1], c=clusters.astype(np.float), cmap = 'jet', alpha=.2)
plt.title("Projection des {} individus sur le 1e plan factoriel pour k=2".format(X_projected.shape[0]))

plt.savefig("images_soutenance/06_kmeans1.jpeg", format="jpeg")


# In[20]:


# Création d'une colonne établissant la validité des billets d'après le K-means
data["cluster"]=clusters
data["is_genuine_cluster"]=data["cluster"].map({0:True, 1:False})

# Matrice de confusion
X = data.is_genuine.values
y = data.is_genuine_cluster.values
X = X.reshape(-1,1)
y = y.reshape(-1,1)

clf = SVC(random_state=0)
clf.fit(X, y)
plot_confusion_matrix(clf, X, y) 
plt.title("Matrice de confusion des {} individus pour k=2".format(X_projected.shape[0]))

plt.savefig("images_soutenance/07_matrice_conf_1.jpeg", format="jpeg")

# Précision et rappel
print(precision_score(data.is_genuine, data.is_genuine_cluster))
print(recall_score(data.is_genuine, data.is_genuine_cluster))


# In[21]:


del X, y


# ###  K-means pour k=3.

# In[22]:


# Application de l'algorithme K-means

# Nombre de clusters souhaités
n_clust = 3

# Clustering par K-means
X = X_scaled.copy()
km = KMeans(n_clusters=n_clust)
km.fit(X)

# Récupération des clusters attribués à chaque individu
clusters = km.labels_

# Affichage du clustering par projection des individus sur le premier plan factoriel
pca = decomposition.PCA(n_components=4).fit(X)
X_projected = pca.transform(X)
plt.scatter(X_projected[:, 0], X_projected[:, 1], c=clusters.astype(np.float), cmap = 'jet', alpha=.2)
plt.title("Projection des {} individus sur le 1e plan factoriel pour k=3".format(X_projected.shape[0]))

plt.savefig("images_soutenance/08_kmeans2.jpeg", format="jpeg")


# In[23]:


# Création d'une matrice de confusion

# Création d'une colonne établissant la validité des billets d'après le K-means
data["cluster"]=clusters
data[["is_genuine", "cluster"]].value_counts()


# In[24]:


# Création d'une fonction pour relier chaque cluster à sa valeur dominante
temp = data[["is_genuine", "cluster"]]

liste_bool_cluster = []
for i in range(0,n_clust):
    liste_bool_cluster.append(temp.loc[temp.cluster==i].is_genuine.value_counts().idxmax())

print(liste_bool_cluster)

data["is_genuine_cluster"]=data["cluster"].map({0:liste_bool_cluster[0],
                                                1:liste_bool_cluster[1],
                                                2:liste_bool_cluster[2]})


# In[25]:


# Matrice de confusion

X = data.is_genuine.values
y = data.is_genuine_cluster.values
X = X.reshape(-1,1)
y = y.reshape(-1,1)

clf = SVC(random_state=0)
clf.fit(X, y)
plot_confusion_matrix(clf, X, y)
plt.title("Matrice de confusion des {} individus pour k=3".format(X_projected.shape[0]))

plt.savefig("images_soutenance/09_matrice_conf_2.jpeg", format="jpeg")


# Précision et rappel
print(precision_score(data.is_genuine, data.is_genuine_cluster))
print(recall_score(data.is_genuine, data.is_genuine_cluster))

Il y a moins d'erreurs avec le second K-means, je continue donc avec les résultats du K-means dans lequel k=3.
# <a id="ch3"></a>
# # MISSION 3 : Régression logistique

#   

# In[26]:


# Avec statsmodels

y = data.is_genuine.values
y = y.reshape(-1,1)
X = data.drop(['is_genuine', "cluster", "is_genuine_cluster"], axis=1)
X_scaled_logit = pd.DataFrame(X_scaled, columns = X.columns)
logit_model=sm.Logit(y,X_scaled_logit)
result=logit_model.fit(method='lbfgs')
print(result.summary2())


# Le coefficient R carré, coefficient de corrélation, est très élevé, mais la régression logistique généralise-t-elle ? En rajoutant des données, le modèle va-t-il se tromper ? 
# Probablement, car les p-value de chaque variable sont toutes très supérieures au seuil de 5% (visible sur la colonne centrale).
# 
# Les deux colonnes de droite sont les bornes de l'intervalle de confiance. Or la valeur 0 est comprise dans cet intervalle, ce qui implique qu'un billet peut ressortir soit vrai soit faux.
# 
# Cela signifie que l'on a trop de variables : il faut en supprimer jusqu'à ce qu'on ait des p-value inférieures à 5% pour chaque variable.

# In[27]:


#Backward Elimination/Selection
pmax = 1
cols = list(X.columns)
X_1 = X_scaled_logit.copy()

while (len(cols)>0):
    p= []
    logit_model=sm.Logit(y,X_1)
    result=logit_model.fit(method='lbfgs')
    p = pd.Series(result.pvalues.values[0:],index = X_1.columns)
    pmax = max(p)
    feature_with_pmax = p.loc[p==pmax].index
    if(pmax>0.05):
        X_1 = X_1.drop(feature_with_pmax, axis=1)
    else:
        break
reg_log_colonnes = X_1.columns
print(result.summary2())


# Les p-value sont inférieures à 5%.  
# Le pseudo R carré est plus faible, mais reste très bon. Le modèle généralise mieux. 

# In[28]:


# Matrice de confusion
X = data.is_genuine.values
X = X.reshape(-1,1)
y = np.where(result.fittedvalues > 0, 1, 0)

clf = SVC(random_state=0)
clf.fit(X, y)
plot_confusion_matrix(clf, X, y)
plt.title("Matrice de confusion des {} individus avec la régression logistique sm".format(X_projected.shape[0]))

plt.savefig("images_soutenance/10_matrice_conf_sm.jpeg", format="jpeg")


# Pandas semble convertir automatiquement les True en 1 et les False en 0.

# In[29]:


# Régression logistique avec scikit-learn

# Création des jeux de données d'entrainement et de test
y = data.is_genuine.values
X_train, X_test, y_train, y_test = train_test_split(X_1, y, test_size=0.25, random_state=42)

# Modèle de régression logistique
reg_log_scikit = LogisticRegression().fit(X_train, y_train)
reg_log_scikit


# In[30]:


# Cross-validation du modèle
modele_prediction_billets = model_selection.cross_val_score(reg_log_scikit,X_train,y_train,cv=10,
                                                            scoring='accuracy')
print(modele_prediction_billets)
print(modele_prediction_billets.mean())


# Les résultats des différentes validations varient entre deux valeurs qui ont entre elles un écart de 8 points, ce qui est moyennement important.
# Néanmoins la moyenne est très bonne, ce qui permet de valider la cross-validation.

# In[31]:


# Test du modèle sur le set de test
prediction_test = reg_log_scikit.predict(X_test)

# Matrice de confusion
clf = SVC(random_state=0)
y_test = pd.DataFrame(y_test)
clf.fit(y_test, prediction_test)
plot_confusion_matrix(clf, y_test, prediction_test)
plt.title("Matrice de confusion des {} individus du set de tests".format(y_test.shape[0]))

plt.savefig("images_soutenance/11_matrice_conf_scikit_test.jpeg", format="jpeg")


#   

#   

# ### On fait une deuxième régression logistique, cette fois sur les axes factoriels dégagés par l'ACP, et on compare les deux modèles.

# In[32]:


# Régression logistique sur les axes factoriels

# Création d'un dataframe avec les coordonnées des individus selon les axes factoriels
coord_indiv_axes = acp.row_coord_
coord_indiv_axes = pd.DataFrame(coord_indiv_axes)

noms_colonnes = [] 
for i in coord_indiv_axes.columns:
    noms_colonnes.append("F{}".format(i+1))

coord_indiv_axes.columns = noms_colonnes
coord_indiv_axes.drop(["F3", "F4", "F5", "F6"], axis=1, inplace=True) # on ne garde que F1/F2

# Régression logistique sur statsmodels

y = data.is_genuine.values
y = y.reshape(-1,1)
X = coord_indiv_axes.copy()
X_logit = pd.DataFrame(X, columns = X.columns)
logit_model_acp=sm.Logit(y,X_logit)
result=logit_model_acp.fit(method='lbfgs')


print(result.summary2())


# In[33]:


# Matrice de confusion

X = data.is_genuine.values
X = X.reshape(-1,1)
y = np.where(result.fittedvalues > 0, 1, 0)

clf = SVC(random_state=0)
clf.fit(X, y)
plot_confusion_matrix(clf, X, y)
plt.title("Matrice de confusion des {} individus avec la régression logistique sm".format(X_projected.shape[0]))


# In[34]:


# Régression logistique avec scikit-learn

# Création des jeux de données d'entrainement et de test
y = data.is_genuine.values
X_train, X_test, y_train, y_test = train_test_split(X_1, y, test_size=0.25, random_state=42)

# Modèle de régression logistique
reg_log_scikit_acp = LogisticRegression().fit(X_train, y_train)

# Cross-validation du modèle
modele_prediction_billets = model_selection.cross_val_score(reg_log_scikit_acp,X_train,y_train,cv=10,
                                                            scoring='accuracy')
print(modele_prediction_billets)
print(modele_prediction_billets.mean())


# In[35]:


# Test du modèle sur le set de test
prediction_test = reg_log_scikit_acp.predict(X_test)

# Matrice de confusion
clf = SVC(random_state=0)
y_test = pd.DataFrame(y_test)
clf.fit(y_test, prediction_test)
plot_confusion_matrix(clf, y_test, prediction_test)
plt.title("Matrice de confusion des {} individus du set de tests".format(y_test.shape[0]))
plt.savefig("images_soutenance/11_matrice_conf_F1_F2.jpeg", format="jpeg")
plt.show()


# Le second modèle donne un résultat un peu meilleur, mais pas de manière significative. De plus le R2 est nettement inférieur. On préfère donc garder le premier modèle.

# In[36]:


# Ligne de code pour la soutenance

fichier_test = pd.read_csv("example.csv")

fichier_test.set_index("id", inplace=True)
fichier_test_scaled = std_scale.transform(fichier_test)
fichier_test_scaled = pd.DataFrame(fichier_test_scaled)
fichier_test_scaled.columns = selected_variables
fichier_test_scaled = fichier_test_scaled[reg_log_colonnes]

pourcentages_validite = pd.DataFrame(reg_log_scikit.predict_proba(fichier_test_scaled)*100)
colonnes = pourcentages_validite.columns
colonnes_bool = colonnes.astype('bool')
pourcentages_validite.columns = colonnes_bool
pourcentages_validite


# In[37]:


# Exportation pour l'API

import joblib
joblib.dump(reg_log_scikit, "reg_log.model")
joblib.dump(std_scale, "std_scale.model")


# In[ ]:




