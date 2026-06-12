import matplotlib.pyplot as plt
import numpy as np

# Données (tu peux les modifier)
categories = ["Signes d’alerte", "Complications possibles", "Importance du traitement"]
oui = [12, 10, 14]
non = [7, 8, 4]

x = np.arange(len(categories))
width = 0.35

plt.figure()

# Barres
bars1 = plt.bar(x - width/2, oui, width, label="Oui")
bars2 = plt.bar(x + width/2, non, width, label="Non")

# Labels
plt.xticks(x, categories)
plt.ylabel("Effectif")
plt.title("Répartition selon l’éducation thérapeutique reçue")

# Légende
plt.legend()

# Ajouter les valeurs au-dessus des barres
for bar in bars1:
    height = bar.get_height()
    plt.text(bar.get_x() + bar.get_width()/2, height,
             f'{height}', ha='center', va='bottom')

for bar in bars2:
    height = bar.get_height()
    plt.text(bar.get_x() + bar.get_width()/2, height,
             f'{height}', ha='center', va='bottom')

plt.tight_layout()
plt.show()