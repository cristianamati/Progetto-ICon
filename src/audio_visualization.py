import librosa
import librosa.display
import pandas as pd
import seaborn as sns
import numpy as np
from matplotlib import pyplot as plt


def plot_sound(path, sr):
    x, sr = librosa.load(path, sr)
    print("Audio length--> {}, sample rate--> {}".format(x.shape, sr))
    return x


def waveplot(audio):
    librosa.display.waveplot(audio, sr=22050)
    plt.xlabel("Time")
    plt.ylabel("Amplitude")
    plt.show()


def spectrogram(audio):
    X = librosa.stft(audio)
    Xdb = librosa.amplitude_to_db(abs(X))
    librosa.display.specshow(Xdb, sr=22050)
    plt.xlabel("Time")
    plt.ylabel("Hz")
    plt.colorbar()
    plt.show()


def mfccs(audio):
    mfccs = librosa.feature.mfcc(audio, n_mfcc=20)
    librosa.display.specshow(mfccs, sr=22050)
    plt.xlabel("Time")
    plt.ylabel("Mfcc")
    plt.colorbar()
    plt.show()


def correlation_matrix():
    spike_cols = [col for col in df.columns if 'mean' in col]
    corr = df[spike_cols].corr()

    mask = np.triu(np.ones_like(corr, dtype=bool))

    # Settaggio figura matplotlib
    f, ax = plt.subplots();

    cmap = sns.diverging_palette(0, 25, as_cmap=True, s=90, l=45, n=5)

    # Disegno la heatmap
    sns.heatmap(corr, mask=mask, cmap=cmap, vmax=.3, center=0,
                square=True, linewidths=.5, cbar_kws={"shrink": .5})

    plt.title('Correlation Heatmap', fontsize=20)
    plt.xticks(fontsize=10)
    plt.yticks(fontsize=10)
    plt.show()


if __name__ == '__main__':
    print("Write the audio file name to analyze (if the file is not inside the current folder write the full path)")
    print("Example of input--->namefile.wav")
    path = input()
    
    #path = 'C:/Users/crist/Desktop/Data/genres_original/blues/blues.00000.wav'
    audio = plot_sound(path, 22050)

    # Waveplot
    choose = input("Do you want to show the audio waveplot? y/n")
    if (choose == 'y'):
        waveplot(audio)
    elif (choose == 'n'):
        print("OK")

    # Spectrogram
    choose = input("Do you want to show the audio spectrogram? y/n")
    if (choose == 'y'):
        spectrogram(audio)
    elif (choose == 'n'):
        print("OK")

    # MFCCs
    choose = input("Do you want to show the Mfccs? y/n")
    if (choose == 'y'):
        mfccs(audio)
    elif (choose == 'n'):
        print("OK")

    # Ispezione del dataset
    df = pd.read_csv('features_3_sec.csv')
    print(df.info())

    choose = input("Do you want to show the Correlation Matrix of the dataset features? y/n")
    if (choose == 'y'):
        correlation_matrix()
    elif (choose == 'n'):
        print("OK")

    genres = df['label'].tolist()
    genres = list(dict.fromkeys(genres))
    print("Music genres in the dataset are-->", genres)
