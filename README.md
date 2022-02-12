# Progetto Ingegneria della conoscenza

Studente: Cristian Amati

Matricola: 707126

E-mail: c.amati9@studenti.uniba.it

## 	Introduzione
Il sistema progettato consiste nell'applicazione di una serie di classificatori ad un dataset musicale (GTZAN dataset) per addestrarli e testarli su esempi non considerati durante il train, effettuando un apprendimento supervisionato.
Lo scopo dell'apprendimento è quello della classificazione del genere musicale di appartenenza per tracce musicali dati alcuni parametri tecnici estratti dalle tracce stesse.

## Il Dataset
Link al dowanload del dataset completo (1 GB circa)[https://www.kaggle.com/andradaolteanu/gtzan-dataset-music-genre-classification/download]

Il dataset utilizzato è il GTZAN dataset, il più famoso dataset usato per applicazione di apprendimento musicale.
Esso contiene 1000 tracce della durata di 30 secondi ognuna divise per generi musicali (in totale 10 generi).
Le tracce sono tutte in formato .wav, dunque senza perdita di qualità.
Allegato al dataset troviamo anche una rappresentazione visuale per ogni traccia in formato .png.

Infine 2 file CSV con le feature di tutte le tracce estratte e rispettive label per il genere di appartenenza.
Il primo file considera ogni traccia splittata in 10 pezzi da 3 secondi ciascuno arrivando cosi ad ottenere circa 10000 dati totali,l'altro considera l'intera song sacrificando però in questo caso il numero di tracce analizzate a 1000 circa.



## 	Algoritmi utilizzati
I classificatori utilizzati sono:
- Naive Bayes, classificatore basato sull'applicazione del teorema di Bayes modellando relazioni probabilistiche tra gli attributi  e l'attributo di classificazione
- Stochastic Gradient Descent, variante della discesa di gradiente che ad ogni iterazione sostituisce il valore del gradiente della funzione di costo con una stima. Converge cosi più velocemente anche senza minimizzare perfettamente la funzione di costo.
- K-Nearest-Neighbor, algoritmo per la classificazione di oggetti basandosi sulle caratteristiche degli oggetti vicini a quello considerato. L'input è costituito dai k esempi di addestramento più vicini nello spazio delle funzionalità. Questo algoritmo ha dimostrato di essere il meglio performante per in dataset considerato.

- Albero di Decisione, modello predittivo dove ogni nodo interno rappresenta una variabile, un arco verso un nodo figlio rappresenta un possibile valore per quella proprietà e una foglia il valore predetto per la variabile obiettivo a partire dai valori delle altre proprietà. Partendo dalla radice, ogni condizione incontrata viene valutata e si segue l'arco
corrispondente al risultato
raggiunta una foglia si assegna la classe corrispondente

- Random Forest, classificatore formato da un insieme di
classificatori semplici (Alberi Decisionali), rappresentati come vettori random
indipendenti e identicamente distribuiti, ognuno di essi vota per la classe più
popolare in input.
Ciascun albero presente all'interno di una Random Forest è costruito e addestrato a
partire da un sottoinsieme casuale dei dati presenti nel training set, gli alberi non
utilizzano quindi il set completo.

- Support Vector Machine, algoritmo con una funzione kernel lineare che rappresenta gli esempi come punti nello spazio, mappati in modo tale che gli esempi appartenenti alle diverse categorie siano chiaramente separati da uno spazio il più possibile ampio, i nuovi esempi sono quindi mappati nello stesso spazio e la predizione della categoria alla quale appartengono viene fatta sulla base del lato nel quale ricade.

- Regressione logistica: Nella fase di addestramento l'algoritmo di regressione logistica prende in input n esempi da un insieme di training. Durante l'addestramento l'algoritmo elabora una distribuzione di pesi che permetta di classificare correttamente gli esempi con le classi corrette. Poi calcola la combinazione lineare z del vettore dei pesi W e degli attributi. La combinazione lineare z viene passata alla funzione logistica (sigmoid) che calcola la probabilità di appartenenza del campione alle classi del modello.


## 	Processo di sviluppo
1)  Effettuo il preprocessing sul training set (encoding dell'attributo target in valori numerici, separazione di attributi dall'attributo target, trasformazione delle features in range [0,1])

2)	Divisione del dataset in 2 parti, una parte (Training-set) usata per addestrare i classificatori e la l'altra utilizzata per testare questi ultimi e le loro performance (Test-set).

3)	Valutazione delle performance dei classificatori tramite il calcolo dell'accuratezza di ciascuna predizione e il confronto di questa con il rispettivo valore reale.

Si è scelto di lavorare sul file "features_3_sec.csv".
L'algoritmo che si è dimostrato più performante durante i test è stato il K-Nearest-Neighbor.


![Esempio](https://github.com/cristianamati/Progetto-ICon/blob/main/report.jpg)

## 	Altre funzioni
Il programma audio_visual.py permette di caricare una traccia audio in formato .wav e successivamente di visualizzarne alcuni aspetti tecnici sotto forma di grafico (waveplot, spettrogramma, MFCCs).
Permette inoltre una visualizzazione completa delle colonne del dataset e la possibilità di creare e mostrare una mappa di correlazione tra esse.

## 	Librerie utilizzate
Per questo sistema sono state utilizzate alcune librerie tipiche delle applicazioni di ML in python:
- sklearn (https://scikit-learn.org/stable/) che permette la definizione e gestione di classificatori e metriche
- pandas (https://pandas.pydata.org/) per la gestione dei trainingset, strutture dati e operazioni su essere
- librosa (https://librosa.org/doc/latest/index.html) , per le operazioni di visualizzazione e manipolazione dei file audio
- numpy (https://numpy.org/), per le numerose funzioni matematiche predisponibili
