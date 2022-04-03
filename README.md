# Progetto Ingegneria della conoscenza

Studente: Cristian Amati

Matricola: 707126

E-mail: c.amati9@studenti.uniba.it

## 	Introduzione
Il sistema progettato consiste nella creazione di un modello di apprendimento supervisionato e non supervisionato per la classificazione dei pinguini in Antartide in 3 specie diverse. E' poi possibile effettuare una predizione basata sul modello supervisionato per la classificazione di un nuovo pinguino dando in input i dati corporei dello stesso.


## Il Dataset

Il dataset utilizzato è il Palmer Penguins dataset, un database di dati raccolti dal  Dr. Kristen Gorman e la Palmer Station (Antartide).
Esso contiene 343 voci per altrettanti pinguini, suddivisi per sesso, specie e isola di provenienza con annessi dati corporei raccolti dalle rilevazioni.

![Esempio](dataset.png)

Gli attributi corporei di ciascun pinguino sono:
- Lunghezza del becco (in mm)
- Profondità del becco (in mm)
- Lunghezza della pinna (in mm)
- Massa (in kg)

![Esempio](body.png)

Il dataset presenza alcuni valori mancanti (NaN) che verrano trattati opportunamente durante l'analisi dei dati nell'algoritmo




## 	Operazioni sul dataset
Per prima cosa dopo aver visualizzato il dataset si procede a verificare la presenza di dati nulli o assenti.

![Esempio](null.png)

Per gestire questi dati si può procedere in diversi modi:

Ignorarli escludendoli dal modello, contattare un esperto, in questo caso l'associazione e/o ricercatori che hanno raccolto i dati per ottendere i valori mancanti o tentare di completare da noi queste lacune.
Si è optato per l'ultima strada

I valori assenti per le 4 colonne relative alle misure corporee vengono dunque rimpiazziati dalla media dei valori nell'intero dataset per quella propietà

Il sesso di un pinguino è inoltre classificato come '.', probabilmente a causa di errori nella compilazione dei dati.
Vengono quindi confrontati i valori dei suoi attributi con le medie di quelli dei pinguini maschi e femmine e classificato come 'femmina'.

Infine circa 10 pinguini pur avendo i dati corporei completi non hanno un sesso assegnato.
Dato che per l'apprendimento è necessario trasformare le colonne con valori testuali in numerici si è scelto di eliminare la colona ralativa al sesso e sostituirla con due colonne 'MALE' e 'FEMALE' con valori binari a seconda che il pinguino sia maschio o femmina.
I pinguini senza sesso assegnato avranno 0 come valore in entrambe le nuove colonne

![Esempio](sex.png)

Infine si è ripetuto lo stesso processo per le isole di provenienza ottenendo cosi il database finale sul quale trainare i modelli

![Esempio](final.png)

Questo dataset è salvato in formato .csv nella stessa directory dello script Pyhton.





## 	Visualizzare e comprendere i dati
Iniziando ad analizzare i dati del dataset, grazie agli strumenti della libreria seaborn è possibile ricavare alcune info sui dati.

Alcuni dati preliminari di facile interpretazione come il numero di pinguini per specie.


![Esempio](count.png)


O la loro ripartizione per isola.

![Esempio](island.png)

Più interessante è il grafico che mostra la correlazione tra le gli attributi fisici dei pinguini

![Esempio](pairplot.png)

Alcune osservazioni:
- La massa coporea non sembra essere una buona variante sulla quale classificare i pinguini

- Le specie "Adelie" e "Chinstrap" hanno alcune caratteristiche simili tra loro
- La specie "Gentoo" è quella che maggiormente si distingue per caratteristiche fisiche
- La lunghezza e la larghezza del becco sembrano essere misure promettenti per la classificazione dei pinguini

Per verificare le dipendenze tra le caratteristiche è utile calcolare la covarianza tra esse

![Esempio](cov.png)

Si nota che la lunghezza del becco, della pinna e la massa variano nella stessa direzione, solo la profondità del becco in direzione opposta

Quanto forti sono queste relazioni è verificabile tramite una apposita matrice

![Esempio](matrix.png)

Si vede ovviamente una forte correlazione tra la massa e la lunghezza della pinna, più sarà pesante il pinguino più la pinna dovrebbe essere lunga

Inoltre la profondità del becco diminuisce all'aumentare della lunghezza dello stesso

## 	Modello supervisionato
Per la costruzione di un modello di apprendimento supervisionato il dataset viene caricato dal file .csv precedentemente creato e splittato in una parte di train e una di test.

 I classificatori usati sono:
- K-Nearest-Neighbor, algoritmo per la classificazione di oggetti basandosi sulle caratteristiche degli oggetti vicini a quello considerato. L'input è costituito dai k esempi di addestramento più vicini nello spazio delle funzionalità.

- Albero di Decisione, modello predittivo dove ogni nodo interno rappresenta una variabile, un arco verso un nodo figlio rappresenta un possibile valore per quella proprietà e una foglia il valore predetto per la variabile obiettivo a partire dai valori delle altre proprietà. Partendo dalla radice, ogni condizione incontrata viene valutata e si segue l'arco corrispondente al risultato raggiunta una foglia si assegna la classe corrispondente

Questi i risultati per il modello basato sull'albero di decisione
![Esempio](tree.png)

E' stato poi calcolato il numero ottimale di neighbor per l'algoritmo K-Nearest-Neighbor
Con k=2 si sono ottenuti questi risultati:

![Esempio](knn.png)

Come è possibile notare anche dai grafici della curva ROC il primo modello è sostanzialmente migliore del secondo.
(Curve più vicine al valore 1 sulle ascisse indicano modelli migliori)

![Esempio](roc1.png)
![Esempio](roc2.png)

Infine è possibile eseguendo lo script predire la classe target, ovvero la specie, di un nuovo pinguino basandosi sul primo modello previo inserimento delle caratteristiche fisiche dell'esemplare.

## 	Modello non supervisionato
Per la creazione di un modello non supervisionato viene eliminata la colonna target relativa alle specie.

L'algoritmo scelto è il K-Means Clustering.
K-means si basa sul concetto di centroide. Il centroide è un punto appartenente allo spazio delle features che media le distanze tra tutti i dati appartenenti al cluster ad esso associato. Rappresenta quindi una sorta di baricentro del cluster.

Si scelgono in modo casuale K centroidi appartenenti allo spazio delle features.
Si calcola la distanza di ogni punto del dataset rispetto ad ogni centroide.
Ogni punto del dataset viene associato al cluster collegato al centroide più vicino.
Si ricalcola la posizione di ogni centroide facendo la media delle posizioni di tutti i punti del cluster associato
Si itera fino a quando non ci sarà più alcun ingresso che cambia di cluster.

E’ possibile verificare il numero ottimale di K tramite il metodo del gomito.

Si itera il K-means per diversi valori di K ed ogni volta si calcola la somma delle distanze al quadrato tra ogni centroide ed i punti del proprio cluster.

![Esempio](elbow.png)

Si nota dal grafico che il numero di K ottimale sarebbe 3 o 2.
Nel nostro caso scegliamo di procedere con K=3.

Viene poi trainato un modello per ogni coppia di proprietà di un pinguino al fine di trovare la combinazione migliore.
A tal proposito è utile notare la distribuzione tramite il grafico

![Esempio](kmeans.png)

Clusterizzare secondo la profondità del becco e la lunghezza della pinna restituisca 3 cluster ben definiti con uno dei 3 molto distaccato dagli altri 2.
Anche clusterizzare secondo la lunghezza del becco e quella della pinna dia tutto sommato buoni risultati

Infine per ogni clusterizzazione viene calcolato il numero di esempi classificati in ognuno dei 3 cluster.

Esempio con clusterizzazione secondo lunghezza della pinna e profondità becco
![Esempio](k_ex.png)

Si nota come questa clusterizzazione distingue molto bene praticamente tutti i pinguini 'Gentoo' mentre ha risultati minori con le altre due specie, che ricordiamo avere maggiori caratteristiche in comune tra loro.

## 	Librerie utilizzate
Per questo sistema sono state utilizzate alcune librerie tipiche delle applicazioni di ML in python:
- sklearn (https://scikit-learn.org/stable/) che permette la definizione e gestione di classificatori e metriche
- pandas (https://pandas.pydata.org/) per la gestione dei trainingset, strutture dati e operazioni su essere
- numpy (https://numpy.org/), per le numerose funzioni matematiche predisponibili
- seaborn e matplotlib (https://seaborn.pydata.org/) (https://matplotlib.org/) per la visualizzazione graphica dei dati
