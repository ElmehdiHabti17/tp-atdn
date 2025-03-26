import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from scipy import stats
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

# Chargement des données
data = pd.read_csv("rendement_mais.csv")

# Etape 1 : Compréhension du problème
print("\nAperçu des données :")
print(data.head())
print("\nDescription des variables :")
print(data.info())

# Identification des variables
cible = "RENDEMENT_T_HA"  # Corrigé le nom de la colonne
variables_explicatives = ["SURFACE_HA", "ENGRAIS_KG_HA", "PRECIPITATIONS_MM", "TEMPERATURE_C", "TYPE_SOL"]

# Etape 2 : Analyse statistique descriptive
print("\nStatistiques descriptives :")
print(data.describe())

# 2.1 Mesures de tendance centrale
print("\nMoyenne du rendement:", data[cible].mean())
print("Médiane du rendement:", data[cible].median())
print("Mode du rendement:", data[cible].mode()[0])

# 2.2 Mesures de dispersion
print("\nEcart-type du rendement:", data[cible].std())
print("Variance du rendement:", data[cible].var())
print("Etendue du rendement:", data[cible].max() - data[cible].min())

# 2.3 Visualisation des données (Histogrammes améliorés)
plt.figure(figsize=(15,5))

# Histogramme du rendement
plt.subplot(1, 3, 1)
plt.hist(data[cible], bins=15, color='skyblue', edgecolor='black', alpha=0.8)
plt.title("Distribution du rendement", fontsize=14)
plt.xlabel("Rendement (tonnes/ha)", fontsize=12)
plt.ylabel("Fréquence", fontsize=12)

# Histogramme des précipitations
plt.subplot(1, 3, 2)
plt.hist(data["PRECIPITATIONS_MM"], bins=15, color='lightgreen', edgecolor='black', alpha=0.8)
plt.title("Distribution des précipitations", fontsize=14)
plt.xlabel("Précipitations (mm)", fontsize=12)
plt.ylabel("Fréquence", fontsize=12)

# Histogramme de la température
plt.subplot(1, 3, 3)
plt.hist(data["TEMPERATURE_C"], bins=15, color='salmon', edgecolor='black', alpha=0.8)
plt.title("Distribution de la température", fontsize=14)
plt.xlabel("Température (°C)", fontsize=12)
plt.ylabel("Fréquence", fontsize=12)

plt.tight_layout()  # Ajuster l'espacement pour un affichage propre
plt.show()

# 2.4 Visualisation des boxplots
plt.figure(figsize=(12,6))

# Boxplot du rendement
plt.subplot(1, 3, 1)
sns.boxplot(data=data[cible], color='lightblue')
plt.title("Boxplot du rendement", fontsize=14)

# Boxplot des précipitations
plt.subplot(1, 3, 2)
sns.boxplot(data=data["PRECIPITATIONS_MM"], color='lightgreen')
plt.title("Boxplot des précipitations", fontsize=14)

# Boxplot de la température
plt.subplot(1, 3, 3)
sns.boxplot(data=data["TEMPERATURE_C"], color='salmon')
plt.title("Boxplot de la température", fontsize=14)

plt.tight_layout()  # Ajuster l'espacement pour un affichage propre
plt.show()

# 2.5 Corrélations
corr_matrix = data.corr(numeric_only=True)
plt.figure(figsize=(8,6))
sns.heatmap(corr_matrix, annot=True, cmap="coolwarm", fmt=".2f")
plt.title("Matrice de corrélation", fontsize=14)
plt.show()

# Etape 3 : Analyse de la variance (ANOVA)
anova = stats.f_oneway(
    data[data["TYPE_SOL"] == "Argileux"][cible],
    data[data["TYPE_SOL"] == "Sableux"][cible],
    data[data["TYPE_SOL"] == "Limoneux"][cible]
)
print("\nRésultat de l'ANOVA :", anova)
if anova.pvalue < 0.05:
    print("Le type de sol a une influence significative sur le rendement.")
else:
    print("Le type de sol n'a pas d'influence significative sur le rendement.")

# Etape 4 : Modélisation
# Transformation des variables catégoriques
data = pd.get_dummies(data, columns=["TYPE_SOL"], drop_first=True)

# Séparation des données
X = data.drop(columns=[cible])
y = data[cible]
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Entraînement des modèles
models = {
    "Régression Linéaire": LinearRegression(),
    "Random Forest": RandomForestRegressor(n_estimators=100, random_state=42)
}

for name, model in models.items():
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)
    print(f"\n{name} :")
    print("MAE:", mean_absolute_error(y_test, y_pred))
    print("RMSE:", np.sqrt(mean_squared_error(y_test, y_pred)))
    print("R²:", r2_score(y_test, y_pred))

# Etape 5 : Interprétation et recommandations
print("\nInterprétation et recommandations :")
print("- Ajuster la quantité d'engrais en fonction des conditions climatiques et du type de sol.")
print("- Favoriser le type de sol qui maximise le rendement.")
print("- Améliorer la collecte de données pour affiner les prédictions.")
