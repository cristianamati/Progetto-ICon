from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import confusion_matrix, auc, roc_curve
from sklearn.metrics import classification_report
from sklearn.metrics import accuracy_score
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

# Ricerca del miglior valore per K
def k_finder(Xtrain, Xtest, Ytrain, Ytest):
    error_rate = []
    for i in range(1, 40):
        knn = KNeighborsClassifier(n_neighbors=i)
        knn.fit(Xtrain, Ytrain)
        prediction_i = knn.predict(Xtest)
        error_rate.append(np.mean(prediction_i != Ytest))

    print("\nVuoi mostrare il tasso di errore per i valori di K? Y/N")
    answer = input()
    if answer == 'Y' or answer == 'y' or answer == 'yes' or answer == 'Yes':
        plt.figure(figsize=(10, 4))
        plt.plot(range(1, 40), error_rate, color='orange', linestyle='dashed', marker='o', markerfacecolor='yellow',
                 markersize=10)
        plt.title('Tasso errore vs. K-Values')
        plt.xlabel('K-Values')
        plt.ylabel('Tasso di errore')
        plt.show()
    else:
        print("OK!")

    error_rate.pop(0)
    bestk = (error_rate.index(min(error_rate))) + 2
    return bestk


# ROC curve
def plot_multiclass_roc(clf, X_test, y_test, n_classes, figsize=(17, 6)):
    y_score = clf.predict_proba(X_test)

    fpr = dict()
    tpr = dict()
    roc_auc = dict()

    y_test_dummies = pd.get_dummies(y_test, drop_first=False).values
    for i in range(n_classes):
        fpr[i], tpr[i], _ = roc_curve(y_test_dummies[:, i], y_score[:, i])
        roc_auc[i] = auc(fpr[i], tpr[i])

    # roc per ogni classe
    fig, ax = plt.subplots(figsize=figsize)
    ax.plot([0, 1], [0, 1], 'k--')
    ax.set_xlim([0.0, 1.0])
    ax.set_ylim([0.0, 1.05])
    ax.set_xlabel('False Positive Rate')
    ax.set_ylabel('True Positive Rate')
    ax.set_title('Receiver operating characteristic example')
    for i in range(n_classes):
        ax.plot(fpr[i], tpr[i], label='Curva ROC (area = %0.2f) per label %i' % (roc_auc[i], i))
    ax.legend(loc="best")
    ax.grid(alpha=.4)
    # sns.despine()
    plt.show()


# Acquisizione dati nuovo pinguino da classificare
def pred():
    temp = []
    cl = input("Inserire lunghezza becco in mm:   ")
    temp.append(float(cl))
    cd = input("Inserire profondità becco in mm:   ")
    temp.append(float(cd))
    fl = input("Inserire lunghezza pinna in mm:   ")
    temp.append(float(fl))
    bm = input("Inserire peso del pinguino in kg:   ")
    temp.append(float(bm))
    sex = input("Inserire sesso M/F:   ")
    if sex == 'M' or sex == 'm':
        temp.append(0)
        temp.append(1)
    elif sex == 'F' or sex == 'f':
        temp.append(1)
        temp.append(0)

    island = input("Inserire isola di provenienza (Biscoe,Dream,Torgersen):   ")
    if island == 'biscoe' or island == 'Biscoe':
        temp.append(1)
        temp.append(0)
        temp.append(0)
    elif island == 'dream' or island == 'Dream':
        temp.append(0)
        temp.append(1)
        temp.append(0)
    elif island == 'torgersen' or island == 'Torgersen':
        temp.append(0)
        temp.append(0)
        temp.append(1)

    df = pd.DataFrame(temp, index=['culmen_length_mm', 'culmen_depth_mm', 'flipper_length_mm', 'body_mass_g', 'FEMALE',
                                   'MALE', 'Biscoe', 'Dream', 'Torgersen'])
    return df.T


if __name__ == '__main__':
    penguin = pd.read_csv('out.csv')

    X = penguin[['culmen_length_mm', 'culmen_depth_mm',
                 'flipper_length_mm', 'body_mass_g', 'FEMALE', 'MALE', 'Biscoe',
                 'Dream', 'Torgersen']]
    Y = penguin['species']
    Xtrain, Xtest, Ytrain, Ytest = train_test_split(X, Y, test_size=0.3, random_state=101)

    DTmodel = DecisionTreeClassifier(random_state=18)
    DTmodel.fit(Xtrain, Ytrain)
    m_pred = DTmodel.predict(Xtest)
    print("\nReport modello Decision Tree:")
    print(classification_report(Ytest, m_pred))
    print(confusion_matrix(Ytest, m_pred))
    # accuracy score
    print("Accuratezza", accuracy_score(m_pred, Ytest))

    bestk = k_finder(Xtrain, Xtest, Ytrain, Ytest)

    Kmodel = KNeighborsClassifier(n_neighbors=bestk)
    Kmodel.fit(Xtrain, Ytrain)
    K_pred = Kmodel.predict(Xtest)
    print("Report modello K-Neighbors:\n")
    print(classification_report(Ytest, K_pred))
    print(confusion_matrix(Ytest, K_pred))

    print("Accuratezza", accuracy_score(K_pred, Ytest))

    print("\nVuoi mostrare la curva ROC per il modello Albero di decisione? Y/N")
    answer = input()
    if answer == 'Y' or answer == 'y' or answer == 'yes' or answer == 'Yes':
        plot_multiclass_roc(DTmodel, Xtest, Ytest, n_classes=3, figsize=(16, 10))

    print("\nVuoi mostrare la curva ROC per il modello KNeighbors? Y/N")
    answer = input()
    if answer == 'Y' or answer == 'y' or answer == 'yes' or answer == 'Yes':
        plot_multiclass_roc(Kmodel, Xtest, Ytest, n_classes=3, figsize=(16, 10))

    print("\nVuoi predire la specie per un nuovo pinguino? Y/N")
    answer = input()
    if answer == 'Y' or answer == 'y' or answer == 'yes' or answer == 'Yes':
        print(DTmodel.predict(pred()))
    else:
        print("Ciao!")
