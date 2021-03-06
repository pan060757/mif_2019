from sklearn.neighbors import LSHForest


X_train = [[5, 5, 2], [21, 5, 5], [1, 1, 1], [8, 9, 1], [6, 10, 2]]
X_test = [[9, 1, 6], [3, 1, 10], [7, 10, 3]]

lshf = LSHForest(random_state=42)
lshf.fit(X_train)

distances, indices = lshf.kneighbors(X_test, n_neighbors=2)
print(distances,indices)