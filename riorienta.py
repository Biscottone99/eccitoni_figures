import numpy as np
import numpy as np
import matplotlib.pyplot as plt
import numpy as np
import matplotlib.pyplot as plt

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
def plot_multiple_molecules(molecules, bond_threshold=1.6):
    """
    Disegna più molecole 3D, con colore dei legami definito da ciascuna molecola.

    Parametri:
        molecules: lista di liste di dizionari, ciascuna contenente:
            - 'atom_labels': lista di stringhe
            - 'coordinates': array Nx3
            - 'name': stringa (opzionale)
            - 'bond_color': stringa colore (opzionale)
        bond_threshold: float
            Distanza massima (in Å) per considerare un legame
    """

    # Colori standard per atomi
    atom_colors = {
        'H': '#ffffff',
        'C': '#444444',
        'N': '#3366cc',
        'O': '#cc0000',
        # aggiungi altri elementi se necessario
    }

    fig = plt.figure(figsize=(10, 8))
    ax = fig.add_subplot(111, projection='3d')

    for mol_list in molecules:
        for mol in mol_list:
            labels = [str(lbl).strip().capitalize() for lbl in mol['atom_labels']]
            coords = np.array(mol['coordinates'])
            bond_color = mol.get('bond_color', 'gray')
            name = mol.get('name', '')

            # --- Disegno atomi
            for label, (x, y, z) in zip(labels, coords):
                color = atom_colors.get(label, 'gray')
                ax.scatter(x, y, z, color=color, s=200, edgecolors='k', linewidths=0.5)
                ax.text(x, y, z, label, fontsize=9, ha='center', va='center', weight='bold')

            # --- Disegno legami
            for i in range(len(coords)):
                for j in range(i + 1, len(coords)):
                    dist = np.linalg.norm(coords[i] - coords[j])
                    if dist < bond_threshold:
                        ax.plot(
                            [coords[i, 0], coords[j, 0]],
                            [coords[i, 1], coords[j, 1]],
                            [coords[i, 2], coords[j, 2]],
                            color=bond_color, linewidth=1.8, label=name if i == 0 and j == 1 else ""
                        )

    ax.set_xlabel('X (Å)', labelpad=10)
    ax.set_ylabel('Y (Å)', labelpad=10)
    ax.set_zlabel('Z (Å)', labelpad=10)
    ax.set_title("3D Molecules", fontsize=14, pad=20)
    ax.set_box_aspect([1, 1, 1])
    ax.grid(False)

    # Mostra legenda solo se ci sono nomi unici
    handles, labels = ax.get_legend_handles_labels()
    if handles:
        unique = dict(zip(labels, handles))
        ax.legend(unique.values(), unique.keys())

    plt.tight_layout()
    plt.show()
def rotate_system(points, A1, A2, B1, B2):
    """
    Ruota un insieme di punti 3D in modo che l'asse A1-A2 venga allineato con l'asse B1-B2.

    Parametri:
        points : array Nx3
            Punti da ruotare.
        A1, A2 : array(3,)
            Punti che definiscono l'asse iniziale.
        B1, B2 : array(3,)
            Punti che definiscono l'asse target.

    Ritorna:
        points_rot : array Nx3
            Punti ruotati.
        R : array 3x3
            Matrice di rotazione applicata.
    """

    # Converte tutto in array numpy
    A1, A2, B1, B2 = map(np.asarray, (A1, A2, B1, B2))

    # Calcola i vettori direzionali normalizzati
    v1 = A2 - A1
    v2 = B2 - B1
    v1 /= np.linalg.norm(v1)
    v2 /= np.linalg.norm(v2)

    # Se i vettori sono già paralleli, niente da fare
    if np.allclose(v1, v2):
        return points.copy(), np.eye(3)

    # Asse e angolo di rotazione (formula di Rodrigues)
    axis = np.cross(v1, v2)
    axis_norm = np.linalg.norm(axis)
    axis /= axis_norm
    angle = np.arccos(np.clip(np.dot(v1, v2), -1.0, 1.0))

    a = np.cos(angle / 2)
    b, c, d = -axis * np.sin(angle / 2)
    R = np.array([
        [a*a + b*b - c*c - d*d, 2*(b*c - a*d),       2*(b*d + a*c)],
        [2*(b*c + a*d),       a*a + c*c - b*b - d*d, 2*(c*d - a*b)],
        [2*(b*d - a*c),       2*(c*d + a*b),       a*a + d*d - b*b - c*c]
    ])

    # Applica la rotazione ai punti rispetto ad A1
    points_shifted = points - A1
    points_rot = (R @ points_shifted.T).T + A1

    return points_rot, R
def translate_system(points, A, B):
    """
    Calcola e applica la traslazione che porta il punto A nel punto B.

    Parametri:
        points : array Nx3
            Punti da traslare.
        A : array(3,)
            Punto di partenza.
        B : array(3,)
            Punto di arrivo.

    Ritorna:
        points_translated : array Nx3
            Punti traslati.
        t : array(3,)
            Vettore di traslazione applicato (B - A).
    """

    A = np.asarray(A)
    B = np.asarray(B)
    t = B - A  # vettore di traslazione

    points_translated = points + t
    return points_translated, t
def rotate_xy_by_vectors(coords, p1a, p2a, p1b, p2b):
    """
    Ruota un insieme di punti 3D nel piano XY di un angolo pari a quello tra
    i vettori (p2a - p1a) e (p2b - p1b), proiettati sul piano XY.

    Parametri:
        coords : np.ndarray Nx3
            Coordinate dei punti da ruotare.
        p1a, p2a : array-like
            Prima coppia di punti che definisce il primo vettore.
        p1b, p2b : array-like
            Seconda coppia di punti che definisce il vettore di riferimento.

    Ritorna:
        coords_rot : np.ndarray Nx3
            Coordinate ruotate nel piano XY.
        angle_deg : float
            Angolo di rotazione in gradi (positivo = antiorario).
        Rz : np.ndarray 3x3
            Matrice di rotazione attorno a Z.
    """
    # Converto in array numpy
    p1a, p2a, p1b, p2b = map(np.asarray, (p1a, p2a, p1b, p2b))

    # Vettori originali e di riferimento, proiettati sul piano XY
    v1 = p2a - p1a
    v2 = p2b - p1b
    v1[2] = 0
    v2[2] = 0

    # Calcolo angolo fra i due vettori nel piano XY
    dot = np.dot(v1, v2)
    det = v1[0]*v2[1] - v1[1]*v2[0]
    angle = np.arctan2(det, dot)  # rad
    angle_deg = np.degrees(angle)

    # Matrice di rotazione nel piano XY
    Rz = np.array([
        [np.cos(angle), -np.sin(angle), 0],
        [np.sin(angle),  np.cos(angle), 0],
        [0, 0, 1]
    ])

    # Applica rotazione attorno a Z (intorno all’origine)
    coords_rot = (Rz @ coords.T).T

    return coords_rot
def rotate_around_axis(coords, p1, p2, angle_deg):
    """
    Ruota un insieme di punti 3D attorno all'asse definito dai punti p1 e p2.

    Parametri:
        coords : np.ndarray Nx3
            Coordinate dei punti da ruotare.
        p1, p2 : array-like di forma (3,)
            Punti che definiscono l'asse di rotazione.
        angle_deg : float
            Angolo di rotazione in gradi (positivo = antiorario seguendo la regola della mano destra).

    Ritorna:
        coords_rot : np.ndarray Nx3
            Coordinate ruotate.
        R : np.ndarray 3x3
            Matrice di rotazione.
    """
    coords = np.asarray(coords, dtype=float)
    p1 = np.asarray(p1, dtype=float)
    p2 = np.asarray(p2, dtype=float)

    # Direzione dell'asse (vettore normalizzato)
    axis = p2 - p1
    axis /= np.linalg.norm(axis)

    # Conversione in radianti
    theta = np.radians(angle_deg)

    # Formula di Rodrigues per la matrice di rotazione
    ux, uy, uz = axis
    cos_t = np.cos(theta)
    sin_t = np.sin(theta)

    R = np.array([
        [cos_t + ux**2 * (1 - cos_t),
         ux * uy * (1 - cos_t) - uz * sin_t,
         ux * uz * (1 - cos_t) + uy * sin_t],

        [uy * ux * (1 - cos_t) + uz * sin_t,
         cos_t + uy**2 * (1 - cos_t),
         uy * uz * (1 - cos_t) - ux * sin_t],

        [uz * ux * (1 - cos_t) - uy * sin_t,
         uz * uy * (1 - cos_t) + ux * sin_t,
         cos_t + uz**2 * (1 - cos_t)]
    ])

    # Trasla in modo che p1 sia l'origine, ruota, poi riporta indietro
    coords_shifted = coords - p1
    coords_rot = (R @ coords_shifted.T).T + p1

    return coords_rot
def rotate_in_plane(coords, p1, p2, p3):
    """
    Calcola l'angolo formato dai tre punti (p1, p2, p3)
    e ruota le coordinate 'coords' attorno alla normale del piano
    definito da (p1, p2, p3) di quell'angolo.

    Parametri:
        coords : np.ndarray Nx3
            Coordinate dei punti da ruotare.
        p1, p2, p3 : array-like (3,)
            Punti che definiscono l'angolo e il piano di rotazione.

    Ritorna:
        coords_rot : np.ndarray Nx3
            Coordinate ruotate.
        angle_deg : float
            Angolo calcolato (in gradi).
        R : np.ndarray 3x3
            Matrice di rotazione.
    """
    coords = np.asarray(coords, dtype=float)
    p1 = np.asarray(p1, dtype=float)
    p2 = np.asarray(p2, dtype=float)
    p3 = np.asarray(p3, dtype=float)

    # --- 1️⃣ Calcolo dell'angolo in p2
    v1 = p1 - p2
    v2 = p3 - p2
    v1 /= np.linalg.norm(v1)
    v2 /= np.linalg.norm(v2)

    dot = np.dot(v1, v2)
    dot = np.clip(dot, -1.0, 1.0)  # evita errori numerici
    angle = np.arccos(dot)
    angle_deg = np.degrees(angle)

    # --- 2️⃣ Definizione della normale al piano
    normal = np.cross(v1, v2)
    normal /= np.linalg.norm(normal)

    # --- 3️⃣ Matrice di rotazione (formula di Rodrigues)
    ux, uy, uz = normal
    cos_t = np.cos(angle)
    sin_t = np.sin(angle)

    R = np.array([
        [cos_t + ux**2 * (1 - cos_t),
         ux * uy * (1 - cos_t) - uz * sin_t,
         ux * uz * (1 - cos_t) + uy * sin_t],

        [uy * ux * (1 - cos_t) + uz * sin_t,
         cos_t + uy**2 * (1 - cos_t),
         uy * uz * (1 - cos_t) - ux * sin_t],

        [uz * ux * (1 - cos_t) - uy * sin_t,
         uz * uy * (1 - cos_t) + ux * sin_t,
         cos_t + uz**2 * (1 - cos_t)]
    ])

    # --- 4️⃣ Ruota attorno al punto p2 (centro dell’angolo)
    coords_shifted = coords - p2
    coords_rot = (R @ coords_shifted.T).T + p2

    return coords_rot

def main():
    # --- Corpo principale del programma ---
    label, x, y, z = read_data('dopo1.txt')
    label2, x2, y2, z2 = read_data('prima1.txt')
    coord = np.stack((x, y, z), axis=1)
    coord2 = np.stack((x2, y2, z2), axis=1)
    A = coord2[2]
    B = coord2[35]
    baricentro_iniziale = np.mean(coord2, axis=0)
    rodrigues, R = rotate_system(coord, coord[2],coord[5], A, B)
    baricentro_finale = np.mean(rodrigues, axis=0)
    traslazione, t = translate_system(rodrigues, baricentro_finale, baricentro_iniziale)
    bar_traslazione = np.mean(traslazione, axis=0)
    new = rotate_in_plane(traslazione, traslazione[35], bar_traslazione, B)
    # baricentro = np.mean(new, axis=0)
    # new2 = rotate_xy_by_vectors(new, baricentro, new[35], baricentro_iniziale, B)
    # bar_new2 = np.mean(new2, axis=0)
    # new3, t3 = translate_system(new2, bar_new2, baricentro_iniziale)
    b4 = np.mean(new, axis=0)
    new4 = rotate_around_axis(new, b4, new[35], -80)
    mol1 = [
        {'atom_labels': label, 'coordinates':coord2, 'name': 'iniziale', 'bond_color':'gray'}
    ]
    mol3 = [
        {'atom_labels': label, 'coordinates': traslazione, 'name': 'ruotata', 'bond_color': 'orange'}
    ]
    mol2 = [
        {'atom_labels': label, 'coordinates': new4, 'name': 'finale', 'bond_color': 'blue'}
    ]
    plot_multiple_molecules([mol1, mol2], bond_threshold=1.6)
    # Questo serve a eseguire main() solo se il file è lanciato direttamente
    output = 'ope_in_ope-ope_new.txt'
    with open(output, 'w') as f:
        f.write(f"{'Atom':<4} {'X':>12} {'Y':>12} {'Z':>12}\n")
        f.write("-" * 43 + "\n")
        for l, (x_new, y_new, z_new) in zip(label, new4):
            f.write(f"{l:<4} {x_new:12.6f} {y_new:12.6f} {z_new:12.6f}\n")
if __name__ == "__main__":
    main()

