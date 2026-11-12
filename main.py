import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.model_selection import train_test_split
import joblib

df = pd.read_csv ("smartphone_dataset.csv")
# print(df.head())

# compteur_marques = df['brand'].value_counts()
# print(compteur_marques)  

top_10_brands = df['brand'].value_counts().head(11).index[1:]         # Nothing est la marque prédominante et est donc supprimée du top 5%
print (top_10_brands)   # Index(['Xiaomi', 'Vivo', 'Apple', 'Google', 'OnePlus', 'Realme', 'Motorola','Samsung', 'Oppo'],dtype='str', name='brand')

# Remplacement des marques qui ne sont pas dans le Top 5 par "Other"
df['brand_nettoyee'] = df['brand'].apply(lambda x: x if x in top_10_brands else 'Other')

print(df['brand_nettoyee'].value_counts())

df = pd.get_dummies(df, columns=['brand_nettoyee'], drop_first=True)

features_finales = ['launch_year', 'storage_gb', 'ram_gb', 'battery_mah'] + [
    col for col in df.columns if col.startswith('brand_nettoyee_')
]
X = df[features_finales]
y = df['price_inr']

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

modele = RandomForestRegressor(n_estimators=50, random_state=42, n_jobs=-1)
print("Entraînement du modèle en cours...")
modele.fit(X_train, y_train)
y_pred = modele.predict(X_test)

print(f'Score R2 : {r2_score(y_test, y_pred):.4f}')
print(f'MAE : {mean_absolute_error(y_test, y_pred):.2f}')
rmse = np.sqrt(mean_squared_error(y_test, y_pred))
print(f'RMSE : {rmse:.2f}')
"""print(
    f'RMSE : {mean_squared_error(y_test, y_pred, squared=False):.2f}'
)"""

""" Visualisation des données"""

plt.figure(figsize=(8, 6))
# On trace les points (Prix réel vs Prix prédit)
plt.scatter(y_test, y_pred, alpha=0.3, color='teal')

# Ligne diagonale parfaite (où Prix prédit = Prix réel)
plt.plot(
    [y_test.min(), y_test.max()],
    [y_test.min(), y_test.max()],
    'r--',
    lw=2,
    label='Prédiction parfaite',
)

plt.xlabel('Prix Réels (INR)')
plt.ylabel('Prix Prédits (INR)')
plt.title('Performance du modèle : Prix Réels vs Prédits')
plt.legend()
plt.show()

"""
y = df ["price_inr"]

features = ["brand", "launch_year", "ram_gb", "battery_mah"]
X = df [features]
X1 = df [features[0]]       # brand
X2 = df [features[1]]       # launch_year
X3 = df [features[2]]       # ram_gb
X4 = df [features[3]]       # battery_mah
# print (X.head())






"""

# Création d'un DataFrame temporaire pour comparer les valeurs réelles et prédites sur le set de test
df_test = X_test.copy()
df_test['prix_reel'] = y_test
df_test['prix_predit'] = y_pred

# Définition des variables numériques à tracer
features_a_tracer = ['launch_year', 'storage_gb', 'ram_gb', 'battery_mah']

fig, axes = plt.subplots(2, 2, figsize=(14, 10))
axes = axes.ravel()

for i, feature in enumerate(features_a_tracer):
    # Nuage de points pour les prix réels
    axes[i].scatter(
        df_test[feature],
        df_test['prix_reel'],
        alpha=0.3,
        color='blue',
        label='Prix Réels',
    )
    # Nuage de points pour les prix prédits par le modèle
    axes[i].scatter(
        df_test[feature],
        df_test['prix_predit'],
        alpha=0.3,
        color='orange',
        label='Prix Prédits',
    )

    axes[i].set_xlabel(feature)
    axes[i].set_ylabel('Prix (INR)')
    axes[i].set_title(f'Impact de {feature} sur le prix')
    axes[i].legend()

plt.tight_layout()
plt.show()


# Sauvegarde du modèle et de la liste exacte des colonnes de X_train
joblib.dump(modele, 'modele_rf_smartphone.pkl')
joblib.dump(X.columns.tolist(), 'colonnes_features.pkl')