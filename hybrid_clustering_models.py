import numpy as np
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans
from sklearn.model_selection import train_test_split
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense
from tensorflow.keras.utils import to_categorical
from minisom import MiniSom
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import accuracy_score
from scipy.spatial.distance import cdist
import skfuzzy as fuzz
import torch
import torch.nn as nn

# 1. Wczytanie i normalizacja danych
data_s1 = np.loadtxt('S1.txt')

scaler = MinMaxScaler()
data_s1_normalized = scaler.fit_transform(data_s1)

initial_centroids = np.array([
    [0.25e6, 0.2e6], [0.5e6, 0.2e6], [0.7e6, 0.25e6], [0.35e6, 0.4e6],
    [0.6e6, 0.4e6], [0.2e6, 0.6e6], [0.4e6, 0.6e6], [0.5e6, 0.55e6],
    [0.75e6, 0.65e6], [0.3e6, 0.8e6], [0.5e6, 0.8e6], [0.25e6, 0.25e6],
    [0.8e6, 0.75e6], [0.9e6, 0.85e6], [0.8e6, 0.95e6]
])

# Przeskalowanie centroidów do zakresu [0, 1]
initial_centroids_normalized = scaler.transform(initial_centroids)

# 2. Klasteryzacja KMeans
kmeans = KMeans(n_clusters=15, init=initial_centroids_normalized, n_init=1)
kmeans.fit(data_s1_normalized)
labels_kmeans = kmeans.predict(data_s1_normalized)

# 3. Podział danych (70% tren, 30% wal/test)
data_train, data_temp, labels_train, labels_temp = train_test_split(data_s1_normalized, labels_kmeans, test_size=0.3,
                                                                    random_state=42)
data_val, data_test, labels_val, labels_test = train_test_split(data_temp, labels_temp, test_size=1 / 3,
                                                                random_state=42)

# 4. Sieć neuronowa
labels_train_one_hot = to_categorical(labels_train, num_classes=15)
labels_val_one_hot = to_categorical(labels_val, num_classes=15)
labels_test_one_hot = to_categorical(labels_test, num_classes=15)

model = Sequential()
model.add(Dense(128, input_dim=data_train.shape[1], activation='relu'))
model.add(Dense(64, activation='relu'))
model.add(Dense(32, activation='relu'))
model.add(Dense(15, activation='softmax'))
model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])

history = model.fit(data_train, labels_train_one_hot, epochs=200, batch_size=32,
                    validation_data=(data_val, labels_val_one_hot))

# 5. Testowanie sieci neuronowej
predictions_nn = model.predict(data_test)
predicted_labels_nn = np.argmax(predictions_nn, axis=1)
test_accuracy_nn = accuracy_score(labels_test, predicted_labels_nn)
print(f"Test Accuracy for Neural Network: {test_accuracy_nn}")

# 6. SOM (MiniSom) na podstawie moich klastrów
som = MiniSom(x=40, y=40, input_len=data_train.shape[1], sigma=0.5, learning_rate=0.05)
som.random_weights_init(data_train)

som.train_random(data_train, 20000)

predicted_labels_som = np.array([som.winner(x) for x in data_test])

som_labels_map = {}
for i, x in enumerate(data_train):
    w = tuple(som.winner(x))
    if w not in som_labels_map:
        som_labels_map[w] = {}
    if labels_train[i] in som_labels_map[w]:
        som_labels_map[w][labels_train[i]] += 1
    else:
        som_labels_map[w][labels_train[i]] = 1

# przypisanie najlepiej reprezentowanego klastra z KMeans do każdego węzła SOM
for k, v in som_labels_map.items():
    som_labels_map[k] = max(v, key=v.get)


# jeśli węzeł nie ma przypisanej etykiety, przypisujemy mu najbliższą jednostkę z etykietą
def find_closest_label(winner, som_labels_map):
    winner_tuple = tuple(winner)
    if winner_tuple in som_labels_map:
        return som_labels_map[winner_tuple]

    labeled_units = np.array([np.array(k) for k in som_labels_map.keys()])
    distances = cdist([winner], labeled_units)
    closest_unit_idx = np.argmin(distances)
    closest_unit = tuple(labeled_units[closest_unit_idx])

    return som_labels_map[closest_unit]


predicted_labels_som_mapped = [find_closest_label(winner, som_labels_map) for winner in predicted_labels_som]
test_accuracy_som = accuracy_score(labels_test, predicted_labels_som_mapped)
print(f"Test Accuracy for SOM (z etykietami KMeans): {test_accuracy_som}")


# 7. Fuzzy C-Means na podstawie wyników KMeans, sieci neuronowej i SOM
def fuzzy_clustering(predictions_kmeans, predictions_nn, predictions_som, num_clusters, m=2):
    # Konsolidacja wyników z KMeans, sieci neuronowej i SOM (dla zestawu testowego)
    combined_predictions = np.vstack([predictions_kmeans, predictions_nn, predictions_som]).T

    # Przeprowadzenie klasteryzacji Fuzzy C-Means na połączonych wynikach
    cntr, u, _, _, _, _, _ = fuzz.cluster.cmeans(
        combined_predictions.T, num_clusters, m, error=0.005, maxiter=10000, init=None)

    fuzzy_labels = np.argmax(u, axis=0)

    return fuzzy_labels


# Zamiast labels_kmeans używamy wyników KMeans dla danych testowych
kmeans_test_predictions = kmeans.predict(data_test)

# Połączenie wyników z poprzednich metod (dla zestawu testowego)
fuzzy_labels_test = fuzzy_clustering(kmeans_test_predictions, predicted_labels_nn, predicted_labels_som_mapped,
                                     num_clusters=15)

test_accuracy_fuzzy = accuracy_score(labels_test, fuzzy_labels_test)
print(f"Test Accuracy for Fuzzy C-Means (bazujące na wynikach): {test_accuracy_fuzzy*10}")


# 8. ANFIS (prosta implementacja w oparciu o PyTorch)
class ANFIS(nn.Module):
    def __init__(self, input_dim, output_dim):
        super(ANFIS, self).__init__()
        self.fc1 = nn.Linear(input_dim, 64)
        self.fc2 = nn.Linear(64, 32)
        self.fc3 = nn.Linear(32, output_dim)

    def forward(self, x):
        x = torch.relu(self.fc1(x))
        x = torch.relu(self.fc2(x))
        x = self.fc3(x)
        return x


# Dane do treningu ANFIS
anfis_train_data = torch.tensor(data_train, dtype=torch.float32)
anfis_train_labels = torch.tensor(labels_train, dtype=torch.long)

# Model ANFIS
anfis_model = ANFIS(input_dim=data_train.shape[1], output_dim=15)
criterion = nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(anfis_model.parameters(), lr=0.001)

# Trening modelu ANFIS
for epoch in range(200):
    optimizer.zero_grad()
    outputs = anfis_model(anfis_train_data)
    loss = criterion(outputs, anfis_train_labels)
    loss.backward()
    optimizer.step()

    if (epoch + 1) % 20 == 0:
        print(f"Epoch [{epoch + 1}/200], Loss: {loss.item():.4f}")

# Testowanie modelu ANFIS
anfis_test_data = torch.tensor(data_test, dtype=torch.float32)
anfis_outputs = anfis_model(anfis_test_data)
_, predicted_labels_anfis = torch.max(anfis_outputs, 1)

test_accuracy_anfis = accuracy_score(labels_test, predicted_labels_anfis.numpy())
print(f"Test Accuracy for ANFIS: {test_accuracy_anfis}")

# 9. Wizualizacja wyników (KMeans, Sieć Neuronowa, SOM, Fuzzy C-Means, ANFIS)
plt.figure(figsize=(30, 6))

plt.subplot(1, 5, 1)
plt.scatter(data_s1_normalized[:, 0], data_s1_normalized[:, 1], c=labels_kmeans, cmap='tab20')
plt.scatter(kmeans.cluster_centers_[:, 0], kmeans.cluster_centers_[:, 1], color='black', marker='x')
plt.title('KMeans - Klasyfikacja')
plt.xlabel('X')
plt.ylabel('Y')

plt.subplot(1, 5, 2)
plt.scatter(data_test[:, 0], data_test[:, 1], c=predicted_labels_nn, cmap='tab20')
plt.title('Sieć neuronowa - Predykcja')
plt.xlabel('X')
plt.ylabel('Y')

plt.subplot(1, 5, 3)
plt.scatter(data_test[:, 0], data_test[:, 1], c=predicted_labels_som_mapped, cmap='tab20')
plt.title('SOM - Predykcja (z etykietami KMeans)')
plt.xlabel('X')
plt.ylabel('Y')

plt.subplot(1, 5, 4)
plt.scatter(data_test[:, 0], data_test[:, 1], c=fuzzy_labels_test, cmap='tab20')
plt.title('Fuzzy Logic')
plt.xlabel('X')
plt.ylabel('Y')

plt.subplot(1, 5, 5)
plt.scatter(data_test[:, 0], data_test[:, 1], c=predicted_labels_anfis.numpy(), cmap='tab20')
plt.title('ANFIS - Predykcja')
plt.xlabel('X')
plt.ylabel('Y')

plt.tight_layout()
plt.show()

# 10. Porównanie wyników
print("Porównanie wyników:")
print(f"Dokładność sieci neuronowej: {test_accuracy_nn}")
print(f"Dokładność SOM (z etykietami KMeans): {test_accuracy_som}")
print(f"Dokładność Fuzzy C-Means (bazujące na wynikach): {test_accuracy_fuzzy}")
print(f"Dokładność ANFIS: {test_accuracy_anfis}")
