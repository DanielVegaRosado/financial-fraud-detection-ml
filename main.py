import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score
from sklearn.preprocessing import StandardScaler

# Configuración visual
sns.set(style="whitegrid")

# --- 1. GENERACIÓN DE DATOS SINTÉTICOS ---
# Simulación de un escenario real donde el fraude es muy poco común
print("Generando dataset de transacciones...")
np.random.seed(42)
n_samples = 10000
n_fraud = int(n_samples * 0.02) # 2% de fraude

# Generación de las legítimas suponiendo que las transacciones normales tienen montos más bajos y patrones estables (Clase 0)
X_legit = np.random.normal(loc=50, scale=10, size=(n_samples - n_fraud, 2)) 

# Generación de las transacciones fraudulentas suponiendo montos más altos o patrones erráticos (Clase 1)
X_fraud = np.random.normal(loc=80, scale=20, size=(n_fraud, 2))

# Unión de los datos
X = np.vstack((X_legit, X_fraud))
y = np.hstack((np.zeros(n_samples - n_fraud), np.ones(n_fraud)))

# Creación el DataFrame
df = pd.DataFrame(X, columns=['transaction_amount', 'user_activity_score'])
df['is_fraud'] = y

print(f"Dataset creado: {df.shape[0]} registros.")
print(f"Distribución de clases:\n{df['is_fraud'].value_counts()}")

# --- 2. PREPROCESAMIENTO ---
# Escalamiento de los datos
scaler = StandardScaler()
X_scaled = scaler.fit_transform(df[['transaction_amount', 'user_activity_score']])

# Separación en Train (80%) y Test (20%)
# 'stratify=y' asegura que haya la misma proporción de fraude en train y test
X_train, X_test, y_train, y_test = train_test_split(
    X_scaled, y, test_size=0.2, random_state=42, stratify=y
)

# --- 3. ENTRENAMIENTO DEL MODELO ---
print("\nEntrenando Random Forest...") # Uso de Random Forest por su buen funcionamiento con datos tabulares (Filas y columnas)
model = RandomForestClassifier(n_estimators=100, random_state=42)
model.fit(X_train, y_train)

# --- 4. EVALUACIÓN ---
print("Evaluando modelo...")
y_pred = model.predict(X_test)

# Métricas
acc = accuracy_score(y_test, y_pred)
print(f"\nAccuracy Global: {acc:.4f}") #La precisión puede causar fallos visuales a simple vista con datos desbalanceados
#Lo ideal es mirar el Classification Report.

print("\n--- Reporte de Clasificación ---")
print(classification_report(y_test, y_pred, target_names=['Legit', 'Fraud']))

# --- 5. VISUALIZACIÓN ---
plt.figure(figsize=(8, 6))
cm = confusion_matrix(y_test, y_pred)
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', cbar=False,
            xticklabels=['Pred: Legit', 'Pred: Fraud'],
            yticklabels=['Real: Legit', 'Real: Fraud'])
plt.title('Matriz de Confusión: Detección de Fraude')
plt.xlabel('Predicción del Modelo')
plt.ylabel('Valor Real')
plt.tight_layout()
plt.show()

plt.savefig('confusion_matrix.png')
print("\nProceso finalizado. Gráfico guardado.")