import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from sklearn.cluster import KMeans
from sklearn.model_selection import train_test_split
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense
from tensorflow.keras.utils import to_categorical
from minisom import MiniSom
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import accuracy_score
from scipy.spatial.distance import cdist

data_breast = np.loadtxt('breast.txt')

scaler = MinMaxScaler()
data_breast_normalized = scaler.fit_transform(data_breast)

kmeans_breast = KMeans(n_clusters=2)
kmeans_breast.fit(data_breast_normalized)

labels_kmeans_breast = kmeans_breast.predict(data_breast_normalized)

# 2. podział danych  (treningowy, walidacyjny, testowy)
data_train, data_temp, labels_train, labels_temp = train_test_split(data_breast_normalized, labels_kmeans_breast,
                                                                    test_size=0.3, random_state=42)
data_val, data_test, labels_val, labels_test = train_test_split(data_temp, labels_temp, test_size=1 / 3,
                                                                random_state=42)

labels_train_one_hot = to_categorical(labels_train, num_classes=2)
labels_val_one_hot = to_categorical(labels_val, num_classes=2)
labels_test_one_hot = to_categorical(labels_test, num_classes=2)

model = Sequential()
model.add(Dense(128, input_dim=data_train.shape[1], activation='relu'))
model.add(Dense(64, activation='relu'))
model.add(Dense(32, activation='relu'))
model.add(Dense(2, activation='softmax'))

model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])

history = model.fit(data_train, labels_train_one_hot, epochs=200, batch_size=32,
                    validation_data=(data_val, labels_val_one_hot))

# 5.dane testowych
predictions_nn = model.predict(data_test)
predicted_labels_nn = np.argmax(predictions_nn, axis=1)
test_accuracy_nn = accuracy_score(labels_test, predicted_labels_nn)
print(f"Test Accuracy for Neural Network: {test_accuracy_nn}")

# 6.SOM
som_breast = MiniSom(x=20, y=20, input_len=data_train.shape[1], sigma=0.5, learning_rate=0.05)
som_breast.random_weights_init(data_train)
som_breast.train_random(data_train, 10000)

predicted_labels_som_breast = np.array([som_breast.winner(x) for x in data_test])

# pzypisywanie klastrów z KMeans do jednostek SOM
som_labels_map_breast = {}
for i, x in enumerate(data_train):
    w = tuple(som_breast.winner(x))
    if w not in som_labels_map_breast:
        som_labels_map_breast[w] = {}
    if labels_train[i] in som_labels_map_breast[w]:
        som_labels_map_breast[w][labels_train[i]] += 1
    else:
        som_labels_map_breast[w][labels_train[i]] = 1

# przypisanie najlepiej reprezentowanego klastra z KMeans do każdego węzła SOM
for k, v in som_labels_map_breast.items():
    som_labels_map_breast[k] = max(v, key=v.get)


# znajdowanie najbliższego węzła z etykietą
def find_closest_label(winner, som_labels_map):
    winner_tuple = tuple(winner)
    if winner_tuple in som_labels_map:
        return som_labels_map[winner_tuple]

    labeled_units = np.array([np.array(k) for k in som_labels_map.keys()])
    distances = cdist([winner], labeled_units)
    closest_unit_idx = np.argmin(distances)
    closest_unit = tuple(labeled_units[closest_unit_idx])

    return som_labels_map[closest_unit]


# przypisanie etykiet do danych testowych
predicted_labels_som_breast_mapped = [find_closest_label(winner, som_labels_map_breast) for winner in
                                      predicted_labels_som_breast]

test_accuracy_som_breast = accuracy_score(labels_test, predicted_labels_som_breast_mapped)
print(f"Test Accuracy for SOM (z etykietami KMeans): {test_accuracy_som_breast}")


def plot_3d_scatter(data, labels, title, dim1, dim2, dim3):
    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')
    scatter = ax.scatter(data[:, dim1], data[:, dim2], data[:, dim3], c=labels, cmap='viridis')
    ax.set_title(title)
    ax.set_xlabel(f'X {dim1 + 1}')
    ax.set_ylabel(f'Y {dim2 + 1}')
    ax.set_zlabel(f'Z {dim3 + 1}')
    plt.colorbar(scatter)
    plt.show()

plot_3d_scatter(data_breast_normalized, labels_kmeans_breast, 'KMeans - Klasyfikacja', 0, 1, 2)

plot_3d_scatter(data_test, predicted_labels_nn, 'Sieć neuronowa - Predykcja', 0, 1, 2)

plot_3d_scatter(data_test, predicted_labels_som_breast_mapped, 'SOM - Predykcja (z etykietami KMeans)', 0, 1, 2)
