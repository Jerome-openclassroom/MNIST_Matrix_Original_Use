```python
import datetime
from tensorflow.keras.callbacks import TensorBoard
import os
print(os.getcwd()) # pour connaitre mon répertoire de travail
os.environ["HF_HUB_DISABLE_SYMLINKS_WARNING"] = "1" #désactive un message d'erreur résultat de l'interaction avec HuggingFace
```

    C:\Users\jerom
    


```python
from datasets import load_dataset
ds = load_dataset("ylecun/mnist")

```


```python
print(ds)
```

    DatasetDict({
        train: Dataset({
            features: ['image', 'label'],
            num_rows: 60000
        })
        test: Dataset({
            features: ['image', 'label'],
            num_rows: 10000
        })
    })
    


```python
#importation des librairies
import tensorflow as tf 
from tensorflow import keras
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay
import numpy as np
import matplotlib.pyplot as plt
import os, sys

```


```python
print("TensorFlow version :", tf.__version__)

```

    TensorFlow version : 2.19.0
    


```python
# 2. Extraction des données brutes
x_train = np.stack([np.array(ex['image']) for ex in ds['train']])
y_train = np.array([ex['label'] for ex in ds['train']])
x_test  = np.stack([np.array(ex['image']) for ex in ds['test']])
y_test  = np.array([ex['label'] for ex in ds['test']])
```


```python
#normalisation des données
x_train = x_train / 255.0
x_test = x_test / 255.0

```


```python
#affichage d'un échantillon 
def plot_multiple_images(x_data, y_data, indices, columns=12):
    rows = len(indices) // columns + int(len(indices) % columns > 0)
    plt.figure(figsize=(columns, rows * 1.5))
    for i, idx in enumerate(indices):
        plt.subplot(rows, columns, i + 1)
        plt.imshow(x_data[idx].reshape(28,28), cmap='gray')
        plt.title(str(y_data[idx]), fontsize=10)
        plt.axis('off')
    plt.tight_layout()
    plt.show()

plot_multiple_images(x_train, y_train, range(5, 41), columns=12)

```


    
![png](output_7_0.png)
    



```python


model = keras.models.Sequential()

# Entrée : image 28x28x1
model.add( keras.layers.Input(shape=(28, 28, 1)) )

# Bloc convolutionnel 1
model.add( keras.layers.Conv2D(8, kernel_size=(3,3), activation='relu') )
model.add( keras.layers.MaxPooling2D(pool_size=(2,2)) )
model.add( keras.layers.Dropout(0.2) )

# Bloc convolutionnel 2
model.add( keras.layers.Conv2D(16, kernel_size=(3,3), activation='relu') )
model.add( keras.layers.MaxPooling2D(pool_size=(2,2)) )
model.add( keras.layers.Dropout(0.2) )

# Flatten + Dense
model.add( keras.layers.Flatten() )
model.add( keras.layers.Dense(100, activation='relu') )
model.add( keras.layers.Dropout(0.5) )

# Sortie : 10 classes (digits 0 à 9)
model.add( keras.layers.Dense(10, activation='softmax') )

```


```python
model.summary()

```


<pre style="white-space:pre;overflow-x:auto;line-height:normal;font-family:Menlo,'DejaVu Sans Mono',consolas,'Courier New',monospace"><span style="font-weight: bold">Model: "sequential_2"</span>
</pre>




<pre style="white-space:pre;overflow-x:auto;line-height:normal;font-family:Menlo,'DejaVu Sans Mono',consolas,'Courier New',monospace">┏━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━┳━━━━━━━━━━━━━━━━━━━━━━━━┳━━━━━━━━━━━━━━━┓
┃<span style="font-weight: bold"> Layer (type)                    </span>┃<span style="font-weight: bold"> Output Shape           </span>┃<span style="font-weight: bold">       Param # </span>┃
┡━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━╇━━━━━━━━━━━━━━━━━━━━━━━━╇━━━━━━━━━━━━━━━┩
│ conv2d_4 (<span style="color: #0087ff; text-decoration-color: #0087ff">Conv2D</span>)               │ (<span style="color: #00d7ff; text-decoration-color: #00d7ff">None</span>, <span style="color: #00af00; text-decoration-color: #00af00">26</span>, <span style="color: #00af00; text-decoration-color: #00af00">26</span>, <span style="color: #00af00; text-decoration-color: #00af00">8</span>)      │            <span style="color: #00af00; text-decoration-color: #00af00">80</span> │
├─────────────────────────────────┼────────────────────────┼───────────────┤
│ max_pooling2d_4 (<span style="color: #0087ff; text-decoration-color: #0087ff">MaxPooling2D</span>)  │ (<span style="color: #00d7ff; text-decoration-color: #00d7ff">None</span>, <span style="color: #00af00; text-decoration-color: #00af00">13</span>, <span style="color: #00af00; text-decoration-color: #00af00">13</span>, <span style="color: #00af00; text-decoration-color: #00af00">8</span>)      │             <span style="color: #00af00; text-decoration-color: #00af00">0</span> │
├─────────────────────────────────┼────────────────────────┼───────────────┤
│ dropout_6 (<span style="color: #0087ff; text-decoration-color: #0087ff">Dropout</span>)             │ (<span style="color: #00d7ff; text-decoration-color: #00d7ff">None</span>, <span style="color: #00af00; text-decoration-color: #00af00">13</span>, <span style="color: #00af00; text-decoration-color: #00af00">13</span>, <span style="color: #00af00; text-decoration-color: #00af00">8</span>)      │             <span style="color: #00af00; text-decoration-color: #00af00">0</span> │
├─────────────────────────────────┼────────────────────────┼───────────────┤
│ conv2d_5 (<span style="color: #0087ff; text-decoration-color: #0087ff">Conv2D</span>)               │ (<span style="color: #00d7ff; text-decoration-color: #00d7ff">None</span>, <span style="color: #00af00; text-decoration-color: #00af00">11</span>, <span style="color: #00af00; text-decoration-color: #00af00">11</span>, <span style="color: #00af00; text-decoration-color: #00af00">16</span>)     │         <span style="color: #00af00; text-decoration-color: #00af00">1,168</span> │
├─────────────────────────────────┼────────────────────────┼───────────────┤
│ max_pooling2d_5 (<span style="color: #0087ff; text-decoration-color: #0087ff">MaxPooling2D</span>)  │ (<span style="color: #00d7ff; text-decoration-color: #00d7ff">None</span>, <span style="color: #00af00; text-decoration-color: #00af00">5</span>, <span style="color: #00af00; text-decoration-color: #00af00">5</span>, <span style="color: #00af00; text-decoration-color: #00af00">16</span>)       │             <span style="color: #00af00; text-decoration-color: #00af00">0</span> │
├─────────────────────────────────┼────────────────────────┼───────────────┤
│ dropout_7 (<span style="color: #0087ff; text-decoration-color: #0087ff">Dropout</span>)             │ (<span style="color: #00d7ff; text-decoration-color: #00d7ff">None</span>, <span style="color: #00af00; text-decoration-color: #00af00">5</span>, <span style="color: #00af00; text-decoration-color: #00af00">5</span>, <span style="color: #00af00; text-decoration-color: #00af00">16</span>)       │             <span style="color: #00af00; text-decoration-color: #00af00">0</span> │
├─────────────────────────────────┼────────────────────────┼───────────────┤
│ flatten_2 (<span style="color: #0087ff; text-decoration-color: #0087ff">Flatten</span>)             │ (<span style="color: #00d7ff; text-decoration-color: #00d7ff">None</span>, <span style="color: #00af00; text-decoration-color: #00af00">400</span>)            │             <span style="color: #00af00; text-decoration-color: #00af00">0</span> │
├─────────────────────────────────┼────────────────────────┼───────────────┤
│ dense_4 (<span style="color: #0087ff; text-decoration-color: #0087ff">Dense</span>)                 │ (<span style="color: #00d7ff; text-decoration-color: #00d7ff">None</span>, <span style="color: #00af00; text-decoration-color: #00af00">100</span>)            │        <span style="color: #00af00; text-decoration-color: #00af00">40,100</span> │
├─────────────────────────────────┼────────────────────────┼───────────────┤
│ dropout_8 (<span style="color: #0087ff; text-decoration-color: #0087ff">Dropout</span>)             │ (<span style="color: #00d7ff; text-decoration-color: #00d7ff">None</span>, <span style="color: #00af00; text-decoration-color: #00af00">100</span>)            │             <span style="color: #00af00; text-decoration-color: #00af00">0</span> │
├─────────────────────────────────┼────────────────────────┼───────────────┤
│ dense_5 (<span style="color: #0087ff; text-decoration-color: #0087ff">Dense</span>)                 │ (<span style="color: #00d7ff; text-decoration-color: #00d7ff">None</span>, <span style="color: #00af00; text-decoration-color: #00af00">10</span>)             │         <span style="color: #00af00; text-decoration-color: #00af00">1,010</span> │
└─────────────────────────────────┴────────────────────────┴───────────────┘
</pre>




<pre style="white-space:pre;overflow-x:auto;line-height:normal;font-family:Menlo,'DejaVu Sans Mono',consolas,'Courier New',monospace"><span style="font-weight: bold"> Total params: </span><span style="color: #00af00; text-decoration-color: #00af00">42,358</span> (165.46 KB)
</pre>




<pre style="white-space:pre;overflow-x:auto;line-height:normal;font-family:Menlo,'DejaVu Sans Mono',consolas,'Courier New',monospace"><span style="font-weight: bold"> Trainable params: </span><span style="color: #00af00; text-decoration-color: #00af00">42,358</span> (165.46 KB)
</pre>




<pre style="white-space:pre;overflow-x:auto;line-height:normal;font-family:Menlo,'DejaVu Sans Mono',consolas,'Courier New',monospace"><span style="font-weight: bold"> Non-trainable params: </span><span style="color: #00af00; text-decoration-color: #00af00">0</span> (0.00 B)
</pre>




```python
model.compile(
    optimizer='adam',
    loss='sparse_categorical_crossentropy',
    metrics=['accuracy']
)

```


```python
#création d'un horodatage pour utilisation de tensorboard
# Dossier de logs horodaté
log_dir = os.path.join("logs", "fit", datetime.datetime.now().strftime("%Y%m%d-%H%M%S"))
tensorboard_callback = TensorBoard(log_dir=log_dir, histogram_freq=1)

# Fit avec callback
history = model.fit(
    x_train, y_train,
    epochs=5,
    batch_size=128,
    validation_data=(x_test, y_test),
    callbacks=[tensorboard_callback],
    verbose=2
)
```

    Epoch 1/5
    469/469 - 6s - 12ms/step - accuracy: 0.7849 - loss: 0.6645 - val_accuracy: 0.9599 - val_loss: 0.1395
    Epoch 2/5
    469/469 - 4s - 8ms/step - accuracy: 0.9256 - loss: 0.2428 - val_accuracy: 0.9724 - val_loss: 0.0910
    Epoch 3/5
    469/469 - 4s - 8ms/step - accuracy: 0.9433 - loss: 0.1891 - val_accuracy: 0.9778 - val_loss: 0.0699
    Epoch 4/5
    469/469 - 4s - 8ms/step - accuracy: 0.9520 - loss: 0.1581 - val_accuracy: 0.9809 - val_loss: 0.0591
    Epoch 5/5
    469/469 - 4s - 8ms/step - accuracy: 0.9579 - loss: 0.1400 - val_accuracy: 0.9823 - val_loss: 0.0547
    


```python
test_loss, test_accuracy = model.evaluate(x_test, y_test, verbose=2)
print(f"Final test accuracy : {test_accuracy:.4f}")

```

    313/313 - 1s - 2ms/step - accuracy: 0.9856 - loss: 0.0406
    Final test accuracy : 0.9856
    


```python
# Accuracy
plt.figure(figsize=(10,4))
plt.subplot(1,2,1)
plt.plot(history.history['accuracy'], label='Train acc')
plt.plot(history.history['val_accuracy'], label='Val acc')
plt.title("Accuracy over epochs")
plt.xlabel("Epoch")
plt.ylabel("Accuracy")
plt.legend()
plt.grid(True)

# Loss
plt.subplot(1,2,2)
plt.plot(history.history['loss'], label='Train loss')
plt.plot(history.history['val_loss'], label='Val loss')
plt.title("Loss over epochs")
plt.xlabel("Epoch")
plt.ylabel("Loss")
plt.legend()
plt.grid(True)

plt.tight_layout()
plt.show()

```


    
![png](output_14_0.png)
    



```python
# Prédiction du modèle sur le jeu de test
y_pred_probs = model.predict(x_test)
y_pred = np.argmax(y_pred_probs, axis=1)

# Matrice de confusion
cm = confusion_matrix(y_test, y_pred)
disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=range(10))

# Affichage
fig, ax = plt.subplots(figsize=(8,8))
disp.plot(ax=ax, cmap='Blues', colorbar=False)
plt.title("Confusion matrix on test set")
plt.show()

```

    [1m313/313[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m1s[0m 2ms/step
    


    
![png](output_15_1.png)
    



```python
# Calcul de la trace (somme des éléments diagonaux)
trace = np.trace(cm)
total = np.sum(cm)
accuracy_from_confusion = trace / total

print(f"Trace (total correct predictions) : {trace}")
print(f"Total number of predictions       : {total}")
print(f"Accuracy from confusion matrix    : {accuracy_from_confusion:.4%}")

# 👇 Commentaire explicatif :
# La trace représente le nombre total de prédictions correctes (valeurs diagonales de la matrice).
# Le taux de réussite calculé ici (trace / total) donne une mesure brute de l'accuracy.
# Il peut différer légèrement de model.evaluate(...) en raison :
# - d'arrondis internes,
# - du calcul batch-wise de la loss dans evaluate(),
# - ou du fait que certaines métriques utilisent une moyenne pondérée.
# Cela reste une méthode complémentaire très fiable pour valider les performances finales.

```

    Trace (total correct predictions) : 9856
    Total number of predictions       : 10000
    Accuracy from confusion matrix    : 98.5600%
    


```python
# ------------------------------------------------------------------------
# 🧠 Commentaire pédagogique :
# Ici, nous adoptons une perspective géométrique sur la matrice de confusion.
# Chaque cellule (i,j) est interprétée comme un point pondéré (erreur i → j),
# où la pondération est le nombre d’occurrences de cette erreur.
#
# La diagonale (i = j) représente la prédiction parfaite. Les erreurs sont
# donc vues comme une "dispersion" autour de cette diagonale idéale.
#
# Nous calculons :
# - la distance moyenne aux prédictions parfaites (MAE)
# - l'erreur quadratique moyenne (MSE), qui pèse plus les confusions graves
# - et la masse hors-diagonale (off-diagonal mass), représentant l’intensité
#   globale de la dispersion hors de la justesse.
#
# Ce glissement conceptuel transforme une matrice discrète en un nuage 
# d'erreurs pondérées, introduisant une lecture statistique continue de la 
# qualité de classification.
# ------------------------------------------------------------------------


# Initialisation
n_classes = cm.shape[0]
errors = []
weights = []

# On parcourt toute la matrice cm[i,j]
for i in range(n_classes):
    for j in range(n_classes):
        count = cm[i, j]
        if count == 0:
            continue
        dist = abs(i - j)                   # Distance à la diagonale (|i - j|)
        sq_dist = (i - j)**2                # Distance au carré
        errors.append((dist, sq_dist, count))
        weights.append(count)

# Extraction des colonnes
dists     = np.array([e[0] for e in errors])
sq_dists  = np.array([e[1] for e in errors])
w         = np.array([e[2] for e in errors])

# Calculs
mean_abs_error  = np.sum(dists * w) / np.sum(w)
mean_sq_error   = np.sum(sq_dists * w) / np.sum(w)
off_diag_mass   = (np.sum(w) - np.trace(cm)) / np.sum(w)

print(f"Mean Absolute Error (MAE)       : {mean_abs_error:.4f}")
print(f"Mean Squared Error (MSE)        : {mean_sq_error:.4f}")
print(f"Off-diagonal mass (error ratio) : {off_diag_mass:.4%}")

```

    Mean Absolute Error (MAE)       : 0.0583
    Mean Squared Error (MSE)        : 0.3131
    Off-diagonal mass (error ratio) : 1.4400%
    


```python
# ------------------------------------------------------------------------
# 🧠 Interprétation avancée des mesures :
#
# - Le MAE (Mean Absolute Error) représente ici une **distance moyenne à la diagonale**,
#   c’est-à-dire à la prédiction parfaite. Plus MAE est faible, plus les erreurs sont
#   "proches" de la vérité. Cela prend tout son sens dans un contexte où les classes
#   ont une **structure ordonnée ou continue**, comme des niveaux de ressemblance
#   (ex. : visages humains classés de 1 à 10, deux personnes affectées de chiffres proches (2 pour l'une, 3 pour l'autre) se ressemblent comme des jumeaux).
#
# - Le MSE (Mean Squared Error), en amplifiant les erreurs lointaines, permet de
#   quantifier la **gravité** des confusions : une erreur de 1 à 2 est bénigne,
#   mais une confusion de 1 à 9 est sévèrement pénalisée.
#
# - L'Off-diagonal mass exprime la **proportion d’erreurs**,
#   soit 1 - accuracy. Elle donne une **mesure brute du taux de dispersion**,
#   indépendamment de la gravité ou de la proximité des erreurs.
#
# Cette approche permet de dépasser la simple "précision moyenne" pour entrer
# dans une **lecture géométrique, sémantique et nuancée** des erreurs d’un modèle,
# particulièrement utile lorsque les classes portent une continuité (valeurs, visages,
# stades biologiques, etc.).
# ------------------------------------------------------------------------

```


```python
%load_ext tensorboard
%tensorboard --logdir logs/fit

```



<iframe id="tensorboard-frame-25e70978514414b5" width="100%" height="800" frameborder="0">
</iframe>
<script>
  (function() {
    const frame = document.getElementById("tensorboard-frame-25e70978514414b5");
    const url = new URL("/", window.location);
    const port = 6006;
    if (port) {
      url.port = port;
    }
    frame.src = url;
  })();
</script>




```python

```
