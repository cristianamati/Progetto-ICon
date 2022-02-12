import pandas as pd
from sklearn import preprocessing
import numpy as np
from sklearn.metrics import classification_report

from sklearn.naive_bayes import GaussianNB
from sklearn.linear_model import SGDClassifier, LogisticRegression
from sklearn.neighbors import KNeighborsClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC

from sklearn.metrics import accuracy_score
from sklearn import preprocessing
from sklearn.model_selection import train_test_split


# Fit del modello con i dati di training, predizione su nuovi valori di test e stampa dell'accuratezza
def model_build(model, title="Default"):
    model.fit(X_train, y_train)
    pred = model.predict(X_test)
    print('Accuracy', title, ':{:.2%}'.format(round(accuracy_score(y_test, pred), 5)), '\n')
    return pred


# Calcolo del miglior valore di k per il K-Nearest-Neighbor
def bestk():
    error_rate = []
    for i in range(1, 40):
        knn = KNeighborsClassifier(n_neighbors=i)
        knn.fit(X_train, y_train)
        prediction_i = knn.predict(X_test)
        error_rate.append(np.mean(prediction_i != y_test))
    error_rate.pop(0)
    k = (error_rate.index(min(error_rate))) + 2
    print("Best k neighbors number is: ", k)
    return k


def genre_dict(genres):
    genres = {i: genres[i] for i in range(0, len(genres))}
    result = {}
    i = 0
    for key, value in genres.items():
        if value not in result.values():
            result[i] = value
            i = i + 1
    return result


if __name__ == '__main__':
    # Acquisiszione dataset e visualizzazione
    print("GTZAN Dataset\n")
    df = pd.read_csv('features_3_sec.csv')
    print(df.head())

    genres = genre_dict(df['label'].tolist())

    # Data Preprocessing

    # Trasformazione colonna target in valori numerici ed eliminazione attributi non necessari
    label_encoder = preprocessing.LabelEncoder()
    df['label'] = label_encoder.fit_transform(df['label'])
    X = df.drop(['label', 'filename'], axis=1)
    y = df['label']

    cols = X.columns
    min_max_scaler = preprocessing.MinMaxScaler()
    np_scaled = min_max_scaler.fit_transform(X)

    # Nuovo data frame con dati scalati.
    X = pd.DataFrame(np_scaled, columns=cols)

    # Split del dataset in training e test sul quale effetuare le predizioni
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=111)

    # Calcolo del miglior numero di neighbors per KNN
    k = bestk()

    # Costruzione modelli

    # Naive Bayes
    nb = GaussianNB()
    nb_pred = model_build(nb, "Naive Bayes")

    # Stochastic Gradient Descent
    sgd = SGDClassifier(max_iter=5000, random_state=0)
    sgd_pred = model_build(sgd, "Stochastic Gradient Descent")

    # Decission trees
    tree = DecisionTreeClassifier()
    tree_pred = model_build(tree, "Decision tree")

    # Random Forest
    rforest = RandomForestClassifier(n_estimators=1000, max_depth=10, random_state=0)
    rforest_pred = model_build(rforest, "Random Forest")

    # Support Vector Machine
    svm = SVC(decision_function_shape="ovo")
    svm_pred = model_build(svm, "Support Vector Machine")

    # Logistic Regression
    lg = LogisticRegression(random_state=0, solver='lbfgs', multi_class='multinomial', max_iter=400)
    lg_pred = model_build(lg, "Logistic Regression")

    # KNN
    knn = KNeighborsClassifier(n_neighbors=k)
    knn_pred = model_build(knn, "KNN")

    # Final Model

    preds = knn.predict(X_test)
    print('Accuracy of final model', ':', round(accuracy_score(y_test, preds), 5), '\n')

    print(genres, "\n")

    # Stampa del classification report
    print(classification_report(y_test, preds))
