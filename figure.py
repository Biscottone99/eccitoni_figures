import numpy as np
import matplotlib.pyplot as plt
import os
from matplotlib import cm

def distances(x, y, z):
    """Calcola la matrice delle distanze tra punti in 3D (in metri)."""
    coords = np.vstack((x, y, z)).T  # shape (N, 3)
    diff = coords[:, np.newaxis, :] - coords[np.newaxis, :, :]
    dist = np.linalg.norm(diff, axis=-1)
    return dist  # Converti da angstrom a metri


def read_data_charges(nome_file):
    """Legge un file .chg con colonne: label, x, y, z, charge."""
    data = np.genfromtxt(
        nome_file,
        dtype=None,
        encoding='utf-8'
    )
    x = np.array([row[0] for row in data], dtype=float)
    y = np.array([row[1] for row in data], dtype=float)
    z = np.array([row[2] for row in data], dtype=float)
    charges = np.array([row[3] for row in data], dtype=float)

    # Conversione delle cariche da e a Coulomb
    charges = -charges

    return x, y, z, charges

def read_data(nome_file):
    """Legge un file .chg con colonne: label, x, y, z, charge."""
    data = np.genfromtxt(
        nome_file,
        dtype=None,
        encoding='utf-8'
    )
    labels = np.array([row[0] for row in data])
    x = np.array([row[1] for row in data], dtype=float)
    y = np.array([row[2] for row in data], dtype=float)
    z = np.array([row[3] for row in data], dtype=float)



    return labels, x, y, z
# ==== Costanti fisiche ====
e = 1.602176634e-19         # Carica elettrone (C)
epsilon = 8.8541878128e-12  # Permittività vuoto (C²/(N·m²))
m = 9.10938356e-31          # Massa elettrone (kg)
c = 299792458               # Velocità luce (m/s)
lambda_nm = 285.13          # Lunghezza d'onda (nm)
lunghezza_onda = lambda_nm * 1e-9
h_bar = 1.0545718e-34       # Costante di Planck ridotta (J·s)
cv_Debye_to_Cm = 3.33564e-30
cv_cm_to_au = 8.47835436255e-30
bond_threshold = 1.6  # soglia in Å

# ==== Lettura dati ====
nome_file1 = ('ope_in_ope-ope_new.txt')
nome_file2 = 'ope.chg'

molecule1, _ = os.path.splitext(nome_file1)
molecule2, _ = os.path.splitext(nome_file2)

l1, x1, y1, z1 = read_data(nome_file1)
x2, y2, z2, charges2 = read_data_charges(nome_file2)

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

# --- Molecola 1 (grigia con etichette)
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

# --- Molecola 2 (colorata per carica)
all_charges = charges2
norm = plt.Normalize(vmin=all_charges.min(), vmax=all_charges.max())
cmap = cm.get_cmap('seismic')

for (x, y, z), q in zip(coord2, charges2):
    color = cmap(norm(q))
    ax.scatter(x, y, z, color=color, s=200, edgecolors='k', linewidths=0.3)

for i in range(len(coord2)):
    for j in range(i + 1, len(coord2)):
        dist = np.linalg.norm(coord2[i] - coord2[j])
        if dist < bond_threshold:
            ax.plot([coord2[i, 0], coord2[j, 0]],
                    [coord2[i, 1], coord2[j, 1]],
                    [coord2[i, 2], coord2[j, 2]],
                    color='gray', linewidth=1.0)

# ==== Layout e aspetto ====
ax.set_xlabel('X (Å)', labelpad=10)
ax.set_ylabel('Y (Å)', labelpad=10)
ax.set_zlabel('Z (Å)', labelpad=10)
ax.set_title("Transition Dipole Moments", fontsize=14, pad=20)
ax.grid(False)
ax.set_box_aspect([1, 1, 1])

# Barra dei colori
sm = cm.ScalarMappable(norm=norm, cmap=cmap)
sm.set_array([])
cbar = plt.colorbar(sm, ax=ax, fraction=0.03, pad=0.08)
cbar.set_label('Carica (C)', rotation=270, labelpad=15)

plt.tight_layout()
plt.show()