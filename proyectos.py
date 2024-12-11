import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import GaussianNB
from sklearn.metrics import classification_report, confusion_matrix, ConfusionMatrixDisplay
from sklearn.metrics import silhouette_score
from wordcloud import WordCloud

# Cargar el dataset
dataset = pd.read_excel("SaludMental.xlsx")
dataset.columns = dataset.columns.str.strip()  # Limpiar nombres de columnas

# Limpieza básica de datos
dataset = dataset.dropna()  # Elimina filas con valores nulos
print(f"Datos después de la limpieza: {dataset.shape}")

# Separar columnas numéricas
numeric_cols = [
    "¿Cuántas horas duermes en promedio cada noche?",
    "En una escala del 1 al 10, ¿cómo calificarías tu nivel de estrés durante el último mes?",
    "¿Cuántas horas al día dedicas al estudio fuera del horario de clases?",
]
num_data = dataset[numeric_cols]

# Estadísticas descriptivas
print("Estadísticas descriptivas:")
print(num_data.describe())

# Visualización inicial: histogramas
for col in numeric_cols:
    plt.figure()
    sns.histplot(num_data[col], kde=True)
    plt.title(f"Distribución de {col}")
    plt.show()

# Nube de palabras
text_data = dataset.drop(columns=numeric_cols + ["Marca temporal"])
all_text = " ".join(text_data.apply(lambda x: " ".join(x.astype(str)), axis=1))
wordcloud = WordCloud(width=800, height=400, background_color="white").generate(all_text)

plt.figure(figsize=(10, 5))
plt.imshow(wordcloud, interpolation="bilinear")
plt.axis("off")
plt.title("Nube de palabras")
plt.show()

# Normalización de datos para clustering
scaler = StandardScaler()
scaled_data = scaler.fit_transform(num_data)

# Determinación del número óptimo de clústeres (Método del Codo y Silhouette Score)
inertia = []
silhouette_scores = []
K_range = range(2, 11)

for k in K_range:
    kmeans = KMeans(n_clusters=k, random_state=42)
    kmeans.fit(scaled_data)
    inertia.append(kmeans.inertia_)
    silhouette_scores.append(silhouette_score(scaled_data, kmeans.labels_))

# Gráfico del método del codo
plt.figure()
plt.plot(K_range, inertia, marker="o")
plt.title("Método del Codo")
plt.xlabel("Número de Clústeres")
plt.ylabel("Inercia")
plt.show()

# Gráfico del Silhouette Score
plt.figure()
plt.plot(K_range, silhouette_scores, marker="o", color="green")
plt.title("Silhouette Score para cada número de Clústeres")
plt.xlabel("Número de Clústeres")
plt.ylabel("Silhouette Score")
plt.show()

# Aplicación de K-means con 3 clústeres (puedes ajustar el número según los gráficos)
kmeans = KMeans(n_clusters=3, random_state=42)
clusters = kmeans.fit_predict(scaled_data)
num_data["Cluster"] = clusters

plt.figure()
sns.histplot(data=num_data, x="¿Cuántas horas duermes en promedio cada noche?", hue="Cluster", multiple="stack")
plt.title("Distribución de horas de sueño por clúster")
plt.show()

# Diagrama de dispersión para explorar relaciones
plt.figure()
sns.scatterplot(
    x=num_data["¿Cuántas horas duermes en promedio cada noche?"],
    y=num_data["En una escala del 1 al 10, ¿cómo calificarías tu nivel de estrés durante el último mes?"],
    hue=num_data["Cluster"],
    palette="viridis"
)
plt.title("Relación entre Horas de Sueño y Nivel de Estrés por Clúster")
plt.xlabel("Horas de Sueño")
plt.ylabel("Nivel de Estrés")
plt.legend(title="Clúster")
plt.show()

# Clasificación con Naive Bayes
num_data["Estrés Alto"] = num_data[
    "En una escala del 1 al 10, ¿cómo calificarías tu nivel de estrés durante el último mes?"
].apply(lambda x: 1 if x >= 7 else 0)

X = num_data.drop(["Cluster", "Estrés Alto"], axis=1)
y = num_data["Estrés Alto"]

# Introducir algo de ruido para simular datos reales
X_noisy = X + np.random.normal(0, 0.5, X.shape)
X_train, X_test, y_train, y_test = train_test_split(X_noisy, y, test_size=0.3, random_state=42)

nb_model = GaussianNB()
nb_model.fit(X_train, y_train)
y_pred = nb_model.predict(X_test)

# Reporte de clasificación
print("Resultados de la clasificación")
print(classification_report(y_test, y_pred))

# Matriz de confusión
conf_matrix = confusion_matrix(y_test, y_pred)
disp = ConfusionMatrixDisplay(confusion_matrix=conf_matrix, display_labels=["Bajo Estrés", "Alto Estrés"])
disp.plot(cmap="Blues")
plt.title("Matriz de Confusión - Naive Bayes")
plt.show()
