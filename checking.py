import numpy as np
import matplotlib.pyplot as plt
import os
from matplotlib import cm
import matplotlib as mpl


def distances(x, y, z):
    """Calcola la matrice delle distanze tra punti in 3D (in metri)."""
    coords = np.vstack((x, y, z)).T
    diff = coords[:, np.newaxis, :] - coords[np.newaxis, :, :]
    dist = np.linalg.norm(diff, axis=-1)
    return dist


def read_data(nome_file):
    """Legge un file con colonne: label, x, y, z."""
    data = np.genfromtxt(nome_file, dtype=None, encoding='utf-8')
    labels = np.array([row[0] for row in data])
    x = np.array([row[1] for row in data], dtype=float)
    y = np.array([row[2] for row in data], dtype=float)
    z = np.array([row[3] for row in data], dtype=float)
    return labels, x, y, z


# ==== Costanti fisiche ====
bond_threshold = 1.6  # soglia in Å

# ==== Lettura dati ====
nome_file1 = 'prima1.txt'
nome_file2 = 'dopo1.txt'

l1, x1, y1, z1 = read_data(nome_file1)
l2, x2, y2, z2 = read_data(nome_file2)

coord1 = np.stack((x1, y1, z1), axis=1)
coord2 = np.stack((x2, y2, z2), axis=1)

# ==== Colori per gli atomi ====
atom_colors = {
    'H': '#ffffff',
    'C': '#444444',
    'N': '#3366cc',
    'O': '#cc0000',
}

# ==== Figura ====
fig = plt.figure(figsize=(10, 8))
ax = fig.add_subplot(111, projection='3d')

# --- Molecola 1 (PRIMA)
for label, (x, y, z) in zip(l1, coord1):
    color = atom_colors.get(label, 'gray')
    ax.scatter(x, y, z, color=color, s=200, edgecolors='k', linewidths=0.5)
    ax.text(x, y, z, label, fontsize=9, ha='center', va='center', weight='bold')

for i in range(len(coord1)):
    for j in range(i + 1, len(coord1)):
        dist = np.linalg.norm(coord1[i] - coord1[j])
        if dist < bond_threshold:
            ax.plot([coord1[i, 0], coord1[j, 0]],
                    [coord1[i, 1], coord1[j, 1]],
                    [coord1[i, 2], coord1[j, 2]],
                    color='gray', linewidth=1.2)

# --- Molecola 2 (DOPO)
for label, (x, y, z) in zip(l2, coord2):
    color = atom_colors.get(label, 'gray')
    ax.scatter(x, y, z, color=color, s=200, edgecolors='k', linewidths=0.5)
    ax.text(x, y, z, label, fontsize=9, ha='center', va='center', weight='bold')

for i in range(len(coord2)):
    for j in range(i + 1, len(coord2)):
        dist = np.linalg.norm(coord2[i] - coord2[j])
        if dist < bond_threshold:
            ax.plot([coord2[i, 0], coord2[j, 0]],
                    [coord2[i, 1], coord2[j, 1]],
                    [coord2[i, 2], coord2[j, 2]],
                    color='gray', linewidth=1.2)

# ==== Aggiungi etichette “PRIMA” e “DOPO” ====
center1 = np.mean(coord1, axis=0)
center2 = np.mean(coord2, axis=0)

# Sposta leggermente le etichette sopra le molecole

# ==== Layout e aspetto ====
ax.set_xlabel('X (Å)', labelpad=10)
ax.set_ylabel('Y (Å)', labelpad=10)
ax.set_zlabel('Z (Å)', labelpad=10)
ax.set_title("Transition Dipole Moments", fontsize=14, pad=20)
ax.grid(False)
ax.set_box_aspect([1, 1, 1])

plt.tight_layout()
plt.show()
