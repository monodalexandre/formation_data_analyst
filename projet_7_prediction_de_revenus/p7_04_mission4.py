#!/usr/bin/env python
# coding: utf-8

# ## MISSION 4

# In[1]:


import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import scipy.stats as st

# Calcul de l'ANOVA
import statsmodels.api as sm
import statsmodels.formula.api as smf
from statsmodels.stats.stattools import durbin_watson
from sklearn import preprocessing
import scipy

data = pd.read_csv("data_mission_4_2.csv")


# # SOMMAIRE
# 
# ## [I - ANOVA(s)](#ch1)
# 
# ### [1) ANOVA paramétrique](#ch1-1)
# 
# ### [2) ANOVA non paramétrique](#ch1-2)
# 
# ## [II - Régressions linéaires](#ch2)
# 
# ### [1) Sans c_parent](#ch2-1)
# 
# ### [2) Avec c_parent](#ch2-2)
# 

# <a id="ch1"></a>
# # I - ANOVA(s)
# 
# <a id="ch1-1"></a>
# ## 1) ANOVA paramétrique

# NOTE POUR MOI-MÊME
# Cas riche pour comprendre GROUPBY et APPLY.
# Je croyais qu'un groupby sans .agg ne marchait pas vraiment ; mais si ! Simplement il n'affiche pas de df.
# Mais il doit bien emmagasiner un df puisqu'avec apply, il en renvoie un.
# Après un groupby comme celui-là, un groupby simple sans agg derrière, un apply(lambda : x.sample()) permet d'avoir seulement un certain nb de lignes, de manière très propre. On pourrait mettre ici, au lieu de (frac=0.5), (500) tout simplement, c-à-d le nb de lignes qu'on veut.

# In[2]:


# J'ai besoin de 500 individus or j'en ai créé 1000 (pour des facilités de calcul)
# Pour réduire à 500, je fais un random sur mon df

grouped = data.groupby('country')
data = grouped.apply(lambda x: x.sample(frac=0.5)).reset_index(drop=True)


# In[3]:


# Pour accélérer le calcul, je fais un groupby qui réunit les 1000 individus enfants créés lors de la Mission 3
# En effet, ici je m'intéresse à la variance de l'income des individus enfants, qui représentent chacun une
# classe de revenu, je ne m'intéresse pas à la variance de l'income des parents de ces enfants, qui serait
# différent parce qu'il n'y pas le même nombre de parents par quantile que d'enfants

# Création d'un df temporaire, avec income et country
temp_country_income = data[["country","income"]].drop_duplicates()


# In[4]:


# Modélisation des boxplots

X = "country"
Y = "income" 

meanprops = {'marker':'o', 'markeredgecolor':'black', 'markerfacecolor':'firebrick'}

plt.figure(figsize=(16, 6))
sns.boxplot(x=temp_country_income[X], y=temp_country_income[Y], showmeans=True, meanprops=meanprops)
plt.title("Variance du revenu moyen selon le pays", fontsize = '25')
plt.xlabel("Income", fontsize=17)
plt.ylabel("Pays", fontsize=17)
plt.xticks(fontsize=7, rotation="vertical")
plt.yticks(fontsize=15)

plt.savefig("images_soutenance/mission4_anova_boxplots.jpeg", format="jpeg")


# Ce graphe, qui affiche les variances des income des différents pays, n'est pas très lisible, en raison des nombreux outliers. On lui préfèrera un graphe qui affiche seulement les moyennes.

# In[5]:


# Modélisation des MOYENNES des boxplots

# Création d'un df temporaire, avec country
country_mean = data.groupby("country").agg({"income" : "mean"})
country_mean = pd.DataFrame(country_mean)
country_mean.reset_index(inplace=True)
country_mean.head(2)

# Modélisation des boxplots

X = "country"
Y = "income" 

meanprops = {'marker':'o', 'markeredgecolor':'black', 'markerfacecolor':'firebrick'}

plt.figure(figsize=(16, 6))
sns.boxplot(x=country_mean[X], y=country_mean[Y], showmeans=True, meanprops=meanprops)
plt.title("Variance du revenu moyen selon le pays", fontsize = '25')
plt.xlabel("Pays", fontsize=17)
plt.ylabel("Income", fontsize=17)
plt.xticks(fontsize=7, rotation="vertical")
plt.yticks(fontsize=15)

plt.savefig("images_soutenance/mission4_anova.jpeg", format="jpeg")


# In[6]:


# Calcul de l'ANOVA

linear_model = smf.ols('income ~ C(country)', data=temp_country_income).fit()
table = sm.stats.anova_lm(linear_model, typ=2)
print(table)

pvalue = linear_model.f_pvalue
print('Ici la p-value est égale à {}'.format(pvalue))


# La pvalue est <5%, donc l'hypothèse H0 est rejetée : les moyennes sont différentes.

# In[7]:


# Affichage de la régression linéaire
print(linear_model.summary2())


# In[8]:


# Mise en place des tests de conformité des régressions linéaires

# Test de normalité

w, pvalue=st.shapiro(linear_model.resid)
print(f"Shapiro-Wilk :{w}, pvalue :{pvalue}")


# Normalité : La pvalue est inférieure à 5% : les résidus NE SUIVENT PAS UNE LOI NORMALE !  
#   
# On a donc ici une ANOVA non paramétrique.

# <a id="ch1-2"></a>
# ## 2) ANOVA non paramétrique

# In[9]:


# Je mets en place un test de Kruskal-Wallis

noms_pays = temp_country_income["country"].unique()
liste_income_pays = [] # Création d'une liste contenant les income de chaque pays
for i in noms_pays:
    liste_income_pays.append(temp_country_income
                             [temp_country_income["country"]==i]["income"])

st.kruskal(*liste_income_pays)


# La p-valeur est < 5% : on rejette H0 : les médianes sont différentes selon chaque pays.  
#   
# On en conclut que le pays a une influence sur l'income.

# <a id="ch2"></a>
# # II - Régressions linéaires
# 
# <a id="ch2-1"></a>
# ## 1) Sans c_parent

# On cherche à établir une relation entre le revenu des enfants et deux variables, que l'on étudie ensemble : le gdpppp, un indice de pouvoir d'achat, et l'indice de Gini de chaque pays

# In[10]:


# Création d'une colonne log
data["gdpppp_log"] = np.log(data.gdpppp)
data["income_log"] = np.log(data.income)


# In[11]:


# Centrage et réduction - création d'un df avec des colonnes centrées-réduites

# Centrage et Réduction

features_to_scale = data[["gdpppp", "gini", "c_i_parent", "gdpppp_log"]]
std_scale = preprocessing.StandardScaler().fit(features_to_scale)
features_scaled = std_scale.transform(features_to_scale)

features_scaled


# In[12]:


# Création d'un dataframe

features_scaled = pd.DataFrame(features_scaled)
features_scaled.columns = ["gdpppp", "gini", "c_i_parent", "gdpppp_log"]

# Ajout des colonnes income, non-scalées

features_scaled[["income", "income_log"]] = data[["income", "income_log"]]
features_scaled


# In[13]:


# Calcul de la régression linéaire multiple

lr = smf.ols('income~gdpppp+gini', data=features_scaled).fit()
print(lr.summary2())

# Affichage de a et b, coefficients de ma droite de régression linéaire
print(lr.params)


# Les deux variables ont une p-value nulle : on peut donc les conserver.
# Le R2 ajusté est faible : or il correspond au pourcentage de variance expliquée ; il faut procéder autrement.  
# On procède au passage de la colonne "income" au logarithme.

# In[14]:


# Mise en place des tests de conformité des régressions linéaires

# Test de normalité

w, pvalue=st.shapiro(lr.resid)
print(f"Shapiro-Wilk :{w}, pvalue :{pvalue}")

# Test d'homoscédasticité

levene, pvalue = st.levene(*liste_income_pays) # Test de Levene
print("Levene : {}, pvalue : {}".format(levene, pvalue))

# Test d'indépendance

print(f"Durbin-Watson : {durbin_watson(lr.resid)}")


# Le notebook affiche un message : la p-value avec autant de données n'est pas significative pour savoir si le modèle suit la loi normale : il faut donc passer à un test graphique : la droite de Henry, ci-dessous.  
# Le test d'homoscédasticité a une pvalue faible, voire nulle : les variances selon les pays diffèrent.
# Le test de Durbin-Watson, proche de 2, indique une indépendance relative, mais suffisante, des individus.

# In[15]:


# Droite de Henry

sm.qqplot(lr.resid, dist=scipy.stats.distributions.norm, line='s')
plt.title("Droite de Henry des résidus".format(lr.resid), fontsize='x-large')
plt.savefig("images_soutenance/mission4_droite_henry_1.jpeg", format="jpeg")


# La droite de Henry ne suit pas la loi normale.

# In[16]:


# Calcul de la régression linéaire multiple avec le logarithme

lr = smf.ols('income_log~gdpppp_log+gini', data=features_scaled).fit()
print(lr.summary2())

# Affichage de a et b, coefficients de ma droite de régression linéaire
print(lr.params)


# In[17]:


# Mise en place des tests de conformité des régressions linéaires

# Test de normalité - droite de Henry (plus besoin de Shapiro)

sm.qqplot(lr.resid, dist=scipy.stats.distributions.norm, line='s')
plt.title("Droite de Henry des résidus".format(lr.resid), fontsize='x-large')
plt.savefig("images_soutenance/mission4_droite_henry_2.jpeg", format="jpeg")

# Test d'homoscédasticité

levene, pvalue = st.levene(*liste_income_pays) # Test de Levene
print("Levene : {}, pvalue : {}".format(levene, pvalue))

# Test d'indépendance

print(f"Durbin-Watson : {durbin_watson(lr.resid)}")


# Cette seconde modélisation, avec le lograithme, suit la loi normale de manière imparfaite mais satisfaisante.  
# Le test d'homoscédasticité est bon, les variances des résidus diffèrent.
# Le DW indique qu'il y a une corrélation moyenne (mais attendue, car on a créé ces données) entre les résidus.
#   
# Le revenu d'une personne est expliqué à 49,3% par le pays, à travers le gdpppp et l'indice de gini.

# <a id="ch2-2"></a>
# ## 2) Avec c_parent

# In[18]:


# Calcul de la régression linéaire multiple

lr = smf.ols('income~gdpppp+gini+c_i_parent', data=features_scaled).fit()
print(lr.summary2())

# Affichage de a et b, coefficients de ma droite de régression linéaire
print(lr.params)


# Le R2 est plus élevé que pour la première régression linéaire multiple.

# In[19]:


# Mise en place des tests de conformité des régressions linéaires

# Test de normalité - droite de Henry

sm.qqplot(lr.resid, dist=scipy.stats.distributions.norm, line='s')
plt.title("Droite de Henry des résidus".format(lr.resid), fontsize='x-large')
plt.savefig("images_soutenance/mission4_droite_henry_3.jpeg", format="jpeg")

# Test d'homoscédasticité

levene, pvalue = st.levene(*liste_income_pays) # Test de Levene
print("Levene : {}, pvalue : {}".format(levene, pvalue))

# Test d'indépendance

print(f"Durbin-Watson : {durbin_watson(lr.resid)}")


# La droite de Henry ne suit pas la loi normale... le test de normalité n'est pas concluant.

# In[20]:


# Calcul de la régression linéaire multiple avec le logarithme

lr = smf.ols('income_log~gdpppp_log+gini+c_i_parent', data=features_scaled).fit()
print(lr.summary2())

# Affichage de a et b, coefficients de ma droite de régression linéaire
print(lr.params)


# In[21]:


# Mise en place des tests de conformité des régressions linéaires

# Test de normalité - droite de Henry

sm.qqplot(lr.resid, line='s')
plt.title("Droite de Henry des résidus".format(lr.resid), fontsize='x-large')
plt.savefig("images_soutenance/droite_henry_4.jpeg", format="jpeg")

# Test d'homoscédasticité

levene, pvalue = st.levene(*liste_income_pays) # Test de Levene
print("Levene : {}, pvalue : {}".format(levene, pvalue))

# Test d'indépendance

print(f"Durbin-Watson : {durbin_watson(lr.resid)}")


# Cette seconde modélisation, avec le logarithme, suit la loi normale de manière imparfaite mais satisfaisante. La modélisation est la meilleure obtenue jusque-là.
# Le test d'homoscédasticité est bon (pvalue < 5%, on rejette H0), les variances des résidus diffèrent.
# En revanche le test DW indique une corrélation plutôt positive des résidus (ce qui est inévitable puisqu'ils ont été artificiellement créés à partir des mêmes données).
#   
# Le R2 à 54% indique que la classe parent explique environ 5% du revenu d'une personne (puisqu'on a vu plus haut que le pays expliquait 49% du revenu --> on fait 54-49 =5%)

# CONCLUSION FINALE
# 
# La classe parent a finalement une influence relative sur le revenu des enfants, qu'on peut observer de manière nette. Néanmoins, son influence n'est pas déterminante, surtout comparée à celle du pays.
#   
# La recommandation Data qui s'impose est donc de cibler les pays ayant un gdpppp élevé, et un indice de Gini faible en premier lieu, puis les pays ayant un gdpppp élevé malgré un indice de Gini faible en second lieu ET les pays ayant un indice de Gini TRES élevé malgré un indice de Gini faible, s'il y en a.
