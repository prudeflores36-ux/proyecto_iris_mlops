import joblib
from sklearn.datasets import load_iris
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier


def train():
    # 1. Cargar el dataset Iris (150 instancias, 4 características, 3 clases)
    iris = load_iris()
    X, y = iris.data, iris.target

    # 2. Dividir en conjuntos de entrenamiento y prueba
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )

    # 3. Entrenar el algoritmo de clasificación
    clf = DecisionTreeClassifier(max_depth=3, random_state=42)
    clf.fit(X_train, y_train)

    # 4. Evaluar la precisión
    predictions = clf.predict(X_test)
    acc = accuracy_score(y_test, predictions)
    print(f"Modelo entrenado exitosamente. Exactitud en prueba: {acc * 100:.2f}%")

    # 5. Exportar el artefacto del modelo y los nombres de las clases
    artifact = {
        "model": clf,
        "target_names": iris.target_names.tolist(),
    }
    joblib.dump(artifact, "iris_model.pkl")
    print("Artefacto 'iris_model.pkl' guardado correctamente.")


if __name__ == "__main__":
    train()
