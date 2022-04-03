import pandas as pd

import matplotlib.pyplot as plt
import seaborn as sns

#Botplox
def botpl(data):
    penguin_b = data[['culmen_length_mm', 'culmen_depth_mm', 'flipper_length_mm']]
    plt.figure(figsize=(8, 4))
    plt.title("Boxplot")
    sns.boxplot(data=penguin_b, width=0.5, fliersize=5)
    plt.show()

#Countplox
def countpl(data):
    plt.figure(figsize=(8, 4))
    plt.title("Pinguini per specie")
    sns.countplot(x="specie", data=data);
    plt.show()

#Pinguini per isola
def peng_isl(data):
    # penguins = pd.read_csv('penguins_size.csv')
    plt.figure(figsize=(8, 4))
    plt.title("Nnumero di pinguini per isola")
    sns.countplot(x="isola", hue="specie", data=data)
    plt.show()

#Distribuzione e covarianza
def distr(data):
    Vpenguin = data[['species', 'culmen_length_mm', 'culmen_depth_mm', 'flipper_length_mm', 'body_mass_g']]
    plt.figure(figsize=(8, 4))
    sns.pairplot(data=Vpenguin, hue="species", height=3, diag_kind="hist")
    plt.savefig('species.png')
    print('Mappa di correlazione tra caratteristiche per specie salvata come species.png!')
    print('\nCovarianza:')

    Vpenguin.select_dtypes("number")
    print(Vpenguin.cov().to_string())

#Correlation matrix
def corr_m(data):
    Vpenguin = data[['species', 'culmen_length_mm', 'culmen_depth_mm', 'flipper_length_mm', 'body_mass_g']]
    plt.figure()
    sns.heatmap(Vpenguin.corr(), annot=True, cmap='inferno')
    plt.title("Matrice di correlazione")
    plt.show()


if __name__ == '__main__':
    penguin = pd.read_csv('out.csv')
    print(penguin.head())

    print("Vuoi mostrare il botplox per la lunghezza becco, profondità becco, lunghezza pinna? Y/N")
    answer = input()
    if answer == 'Y' or answer == 'y' or answer == 'yes' or answer == 'Yes':
        botpl(penguin)
    else:
        print("OK!")

    print("Vuoi mostrare il numero di pinguini per specie? Y/N")
    answer = input()
    if answer == 'Y' or answer == 'y' or answer == 'yes' or answer == 'Yes':
        countpl(penguin)
    else:
        print("OK!")

    print("Vuoi mostrare la distribuzione dei pinguini per isola? Y/N")
    answer = input()
    if answer == 'Y' or answer == 'y' or answer == 'yes' or answer == 'Yes':
        peng_isl(penguin)
    else:
        print("OK!")

    print("Vuoi mostrare la matrice di correlazione? Y/N")
    answer = input()
    if answer == 'Y' or answer == 'y' or answer == 'yes' or answer == 'Yes':
        corr_m(penguin)
    else:
        print("OK!")

    distr(penguin)
