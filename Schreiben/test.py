import numpy as np
import matplotlib.pyplot as plt

# Gegebene Punkte
points = np.array([
    [1.4167, -1.2667],
    [-1.4833, 1.2333],
    [1.3167, -0.1667],
    [-1.9833, -0.2667],
    [0.9167, -0.2667],
    [-0.1833, 0.7333]
])

# SVD der Punkte (Singulärwertzerlegung)
U, S, Vt = np.linalg.svd(points, full_matrices=False)
V = Vt.T  # Vektor aus der rechten singulären Matrix

# Richtungsvektor der ersten Hauptkomponente (bereits ein Einheitsvektor)
V_first = V[:, 0]

# Berechne die orthogonale Projektion auf die Linie (Einheitsvektor benötigt keine Normierung)
points_proj = np.array([np.dot(p, V_first) * V_first for p in points])

# Plot der ursprünglichen Punkte, projizierten Punkte und der Geraden
plt.figure(figsize=(6, 6))
plt.scatter(points[:, 0], points[:, 1], color='blue', label='Ursprüngliche Punkte')
plt.scatter(points_proj[:, 0], points_proj[:, 1], color='red', label='Projizierte Punkte')
plt.plot([0, V_first[0]], [0, V_first[1]], color='gray', label='Hauptkomponente (Gerade)', linewidth=2)

# Verbindungslinien zwischen den ursprünglichen Punkten und den projizierten Punkten
for i in range(len(points)):
    plt.plot([points[i, 0], points_proj[i, 0]], [points[i, 1], points_proj[i, 1]], 'gray', linestyle='dotted')

# Achsen, Titel und Legende
plt.axhline(0, color='black',linewidth=0.5)
plt.axvline(0, color='black',linewidth=0.5)
plt.gca().set_aspect('equal', adjustable='box')
plt.xlabel('X')
plt.ylabel('Y')
plt.legend()
plt.title('Orthogonale Projektion auf die Hauptkomponente')

plt.grid(True)
plt.show()
