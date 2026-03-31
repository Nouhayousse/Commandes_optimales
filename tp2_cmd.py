import numpy as np

import matplotlib.pyplot as plt



def solve_optimal_control(N, alpha, xi0_val, xi1_val,verbose=False):

    h = 1 / (N + 1)

    x = np.linspace(h, 1-h, N)

    

    # 1. Matrice du Laplacien (Différences Finies)

    A = (1/h**2) * (2*np.eye(N) - np.eye(N, k=1) - np.eye(N, k=-1))

    

    # 2. Définition des données pour la solution exacte

    y_exact = np.sin(np.pi * x)

    p_exact = x*(1-x)*(x-0.5)

    

    # CORRECTION : f = -y*'' - u* = pi²·sin(πx) + p*/alpha  pour TOUT x
    # (l'ancienne formule splittée x<0.5 / x>=0.5 était incorrecte)
    f = np.pi**2 * np.sin(np.pi * x) + p_exact / alpha

    z_d = (1-np.pi**2)*np.sin(np.pi*x)

    

    # Contraintes

    xi0 = xi0_val * np.ones(N)

    xi1 = xi1_val * np.ones(N)

    

    # 3. Boucle d'optimisation (Gradient Projeté)

    u = np.zeros(N)

    for k in range(200):

        y = np.linalg.solve(A, f + u)

        p = np.linalg.solve(A, y - z_d)

        

        u_new = -p / alpha

        u_new = np.maximum(xi0, np.minimum(xi1, u_new)) # Projection

        

        if np.linalg.norm(u_new - u) < 1e-9:

            break

        u = u_new
    if verbose:

        print(f"A : matrice {A.shape}, diagonale = {A[0,0]:.4f}, sous-diag = {A[0,1]:.4f}")
        print(f"f  : min={f.min():.4f}, max={f.max():.4f}")
        print(f"u* : min={u.min():.4f}, max={u.max():.4f}")
        print(f"p  : min={p.min():.4f}, max={p.max():.4f}")


    return x, y, p, u, y_exact, p_exact



# =====================================================

# 1. VALIDATION ET AFFICHAGE DES RÉSULTATS (p, y, u)

# =====================================================

x, y, p, u, y_ex, p_ex = solve_optimal_control(100, 0.1, -10, 10,verbose=True)
plt.figure(figsize=(15, 4))
plt.subplot(1, 3, 1)
plt.plot(x, y, 'b', label="y numérique")
plt.plot(x, y_ex, 'r--', label="y exact")
plt.title("État (y)")
plt.legend()


plt.subplot(1, 3, 2)
plt.plot(x, p, 'g', label="p numérique")
plt.plot(x, p_ex, 'r--', label="p exact")
plt.title("Adjoint (p)")
plt.legend()


plt.subplot(1, 3, 3)
plt.plot(x, u, label="u*")
plt.plot(x, y, label="y*")
plt.title("Contrôle vs État")
plt.legend()
plt.tight_layout()



# =====================================================

# 2. ÉTUDE DE SENSIBILITÉ (ALPHA ET SATURATION)

# =====================================================

# Influence de Alpha

alphas = [0.5, 0.1, 0.01]
plt.figure(figsize=(12, 5))
for a in alphas:
    res = solve_optimal_control(100, a, -10, 10)
    plt.subplot(1, 2, 1); plt.plot(res[0], res[1], label=f'alpha={a}')
    plt.subplot(1, 2, 2); plt.plot(res[0], res[3], label=f'alpha={a}')

plt.subplot(1, 2, 1); plt.title("Influence de alpha sur y"); plt.legend()
plt.subplot(1, 2, 2); plt.title("Influence de alpha sur u"); plt.legend()



# Saturation (Contraintes)
x, y_un, p_un, u_un, _, _ = solve_optimal_control(100, 0.1, -10, 10)
x, y_sat, p_sat, u_sat, _, _ = solve_optimal_control(100, 0.1, -0.2, 0.2)
plt.figure(figsize=(12, 5))
plt.subplot(1, 2, 1)
plt.plot(x, u_un, '--', label="Sans contrainte")
plt.plot(x, u_sat, 'r', label="Saturé [-0.2, 0.2]")
plt.title("Saturation du contrôle u")
plt.legend()
plt.subplot(1, 2, 2)
plt.plot(x, y_un, '--', label="y sans contrainte")
plt.plot(x, y_sat, 'g', label="y avec saturation")
plt.title("Impact sur l'état y")
plt.legend()



# =====================================================

# 3. CALCUL NUMÉRIQUE DES ERREURS (ORDRE 2)

# =====================================================

print("\n" + "="*45)

print(f"{'N':<5} | {'h':<10} | {'Erreur L-inf':<15} |{'Rapport':<10}")
print("-" * 45)
Ns = [20, 40, 80, 160]
prev_err = None
for N in Ns:
    h = 1/(N+1)
    x_err, y_err, _, _, y_ex_err, _ = solve_optimal_control(N, 0.1, -10, 10)
    err = np.linalg.norm(y_err - y_ex_err, ord=np.inf)
    rapport = "-" if prev_err is None else f"{prev_err/err:.2f}"
    print(f"{N:<5} | {h:<10.4f} | {err:<15.2e} | {rapport:<10}")
    prev_err = err
print("="*45)
plt.show()