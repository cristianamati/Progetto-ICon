import pandas as pd

#Riempimento valori nulli con la media della proprietà
def nul_mean(penguin):
    mean_CL = penguin['culmen_length_mm'].mean()
    penguin['culmen_length_mm'].fillna(value=mean_CL, inplace=True)
    penguin.isnull().sum()

    mean_CL = penguin['culmen_depth_mm'].mean()
    penguin['culmen_depth_mm'].fillna(value=mean_CL, inplace=True)
    penguin.isnull().sum()

    mean_CL = penguin['flipper_length_mm'].mean()
    penguin['flipper_length_mm'].fillna(value=mean_CL, inplace=True)
    penguin.isnull().sum()

    mean_CL = penguin['body_mass_g'].mean()
    penguin['body_mass_g'].fillna(value=mean_CL, inplace=True)
    penguin.isnull().sum()

    penguin.loc[336, 'sex'] = "FEMALE"

#Gestione valori nulli relativi al sesso
def sex_na(penguin):
    penguin['sex'] = penguin['sex'].fillna("N/A")
    dummies = pd.get_dummies(penguin.sex)
    merge = pd.concat([penguin, dummies], axis='columns')
    penguin_data = merge.drop(['sex', 'N/A'], axis='columns')

    dummies2 = pd.get_dummies(penguin.island)
    penguin_data = pd.concat([penguin_data, dummies2], axis='columns')
    penguin.drop('island', axis=1, inplace=True)
    return penguin_data


if __name__ == '__main__':
    penguin = pd.read_csv('penguins_size.csv')
    print("Input dataset")
    print(penguin.head())
    nul_mean(penguin)
    #print(penguin.head())
    penguin = sex_na(penguin)
    print("Dataset finale")
    print(penguin.head().to_string())

    penguin.to_csv('out.csv')
