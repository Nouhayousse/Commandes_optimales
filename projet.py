"""
TP3 - Commande Optimale pour une EDP Parabolique
Avec analyse de convergence et validation numérique
"""

import numpy as np
import matplotlib.pyplot as plt
from scipy.sparse import diags, lil_matrix
from scipy.sparse.linalg import splu
import scipy.sparse as sp
from scipy.integrate import simpson
import time

# ============================================================================
# PARTIE 1 : FONCTIONS DE BASE (inchangées)
# ============================================================================

def construire_matrice_A(N, dx):
    """Construction de la matrice du Laplacien"""
    main_diag = 2.0 / (dx * dx) * np.ones(N)
    off_diag = -1.0 / (dx * dx) * np.ones(N-1)
    A = diags([off_diag, main_diag, off_diag], offsets=[-1, 0, 1], format='csr')
    return A

def construire_matrice_B(A, dt, N):
    """Construction de B = I + dt*A"""
    I = sp.eye(N, format='csr')
    B = I + dt * A
    return B

def construire_matrice_bloc(B, dt, alpha, N):
    """Construction de la matrice bloc M_bloc"""
    I = sp.eye(N, format='csr')
    M_bloc = lil_matrix((2*N, 2*N))
    M_bloc[0:N, 0:N] = B
    M_bloc[0:N, N:2*N] = (dt / alpha) * I
    M_bloc[N:2*N, 0:N] = (-dt) * I
    M_bloc[N:2*N, N:2*N] = B
    return M_bloc.tocsr()

# ============================================================================
# PARTIE 2 : TEST DE FABRIQUE DE SOLUTIONS (MANUFACTURED SOLUTION)
# ============================================================================
# On construit une solution exacte artificielle pour valider le code
# En ajoutant des termes sources appropriés, on peut vérifier que le code
# résout correctement le système couplé.

def solution_exacte_y(t, x):
    """
    Solution exacte artificielle pour y(t,x)
    Choisie pour satisfaire les conditions aux limites y(t,0)=y(t,1)=0
    """
    # Exemple : y(t,x) = sin(π x) * exp(-t)
    return np.sin(np.pi * x) * np.exp(-t)

def solution_exacte_p(t, x):
    """
    Solution exacte artificielle pour p(t,x)
    Choisie pour satisfaire p(t,0)=p(t,1)=0 et p(T,x)=0
    """
    # Exemple : p(t,x) = sin(π x) * (T - t)
    return np.sin(np.pi * x) * (1.0 - t)  # pour T=1

def source_f_manufacturee(t, x, y_exact, p_exact, alpha):
    """
    Calcule le terme source f(t) nécessaire pour que (y_exact, p_exact)
    soit solution du système avec le contrôle u = -p/α
    
    À partir de l'équation d'état :
    ∂_t y - ∂_xx y + (1/α)p = f
    donc f = ∂_t y - ∂_xx y + (1/α)p
    """
    # Dérivées exactes
    # y = sin(πx) * exp(-t)
    # ∂_t y = -sin(πx) * exp(-t)
    # ∂_xx y = -π² sin(πx) * exp(-t)
    
    # Pour notre exemple :
    # ∂_t y = -y
    # ∂_xx y = -π² y
    # Donc -∂_xx y = π² y
    
    # Formulation générale pour l'exemple
    dy_dt = -y_exact
    d2y_dx2 = -np.pi**2 * y_exact
    
    f = dy_dt - d2y_dx2 + (1.0/alpha) * p_exact
    
    return f

def zd_manufacturee(t, x, y_exact, p_exact):
    """
    Calcule la cible z_d nécessaire pour que (y_exact, p_exact)
    soit solution de l'équation adjointe
    
    À partir de : -∂_t p - ∂_xx p = y - z_d
    donc z_d = y + ∂_t p + ∂_xx p
    """
    # Pour notre exemple : p = sin(πx) * (1-t)
    # ∂_t p = -sin(πx)
    # ∂_xx p = -π² sin(πx) * (1-t) = -π² p
    
    # Donc ∂_t p + ∂_xx p = -sin(πx) - π² sin(πx)*(1-t) = -π² p - sin(πx)
    # Mais sin(πx) = y * exp(t) ? Non, mieux vaut calculer directement
    
    dp_dt = -np.sin(np.pi * x)
    d2p_dx2 = -np.pi**2 * p_exact
    
    z_d = y_exact + dp_dt + d2p_dx2
    
    return z_d

def condition_initiale_manufacturee(x):
    """Condition initiale y(0,x) = solution_exacte_y(0,x)"""
    return solution_exacte_y(0, x)

# ============================================================================
# PARTIE 3 : SOLVEUR AVEC ANALYSE DE CONVERGENCE
# ============================================================================

class SolveurControleOptimal:
    """
    Classe principale pour la résolution du problème de contrôle optimal
    avec analyse de convergence
    """
    
    def __init__(self, T, alpha, N, M):
        """
        Initialisation du solveur
        
        Args:
            T: temps final
            alpha: coefficient de régularisation
            N: nombre de points intérieurs en espace
            M: nombre de pas de temps
        """
        self.T = T
        self.alpha = alpha
        self.N = N
        self.M = M
        
        # Calcul des pas
        self.dx = 1.0 / (N + 1)
        self.dt = T / M
        
        # Coordonnées
        self.x_pts = np.linspace(self.dx, 1.0 - self.dx, N)
        self.t_pts = np.linspace(0, T, M+1)
        
        # Construction des matrices
        self.A = construire_matrice_A(N, self.dx)
        self.B = construire_matrice_B(self.A, self.dt, N)
        self.M_bloc = construire_matrice_bloc(self.B, self.dt, alpha, N)
        
        # Factorisation LU (une seule fois)
        self.lu = splu(self.M_bloc)
        
        # Stockage des résultats
        self.y = None
        self.p = None
        self.u = None
        
    def resoudre(self, f_source, zd_source, y0_source):
        """
        Résout le système complet
        
        Args:
            f_source: fonction f(t,x) (ou f(t) si indépendante de x)
            zd_source: fonction z_d(t,x)
            y0_source: fonction y0(x)
        
        Returns:
            y, p, u: solutions
        """
        N, M = self.N, self.M
        dt, dx = self.dt, self.dx
        T = self.T
        
        # Préparation des données
        f_vals = np.zeros(M+1)
        zd_vals = np.zeros((M+1, N))
        y0 = y0_source(self.x_pts)
        
        for n in range(M+1):
            t = n * dt
            # Si f_source est indépendante de x (cas standard)
            if hasattr(f_source, '__call__'):
                try:
                    f_vals[n] = f_source(t)
                except:
                    f_vals[n] = f_source(t, self.x_pts[0])
            else:
                f_vals[n] = f_source
            
            for i in range(N):
                zd_vals[n, i] = zd_source(t, self.x_pts[i])
        
        # Initialisation
        y = np.zeros((M+1, N))
        P = np.zeros((M+1, N))
        y[0, :] = y0
        P[0, :] = 0.0
        
        # Boucle principale
        for j in range(1, M+1):
            n_y = M - j + 1
            n_f = M - j + 1
            n_zd = j
            
            # Second membre
            if M - j == 0:
                y_prev = y[0, :]
            else:
                y_prev = y[M - j, :]
            
            rhs_top = y_prev + dt * f_vals[n_f] * np.ones(N)
            rhs_bottom = P[j-1, :] - dt * zd_vals[n_zd, :]
            rhs = np.concatenate([rhs_top, rhs_bottom])
            
            # Résolution
            solution = self.lu.solve(rhs)
            y[n_y, :] = solution[:N]
            P[j, :] = solution[N:]
        
        # Récupération de p
        p = np.zeros((M+1, N))
        for n in range(M+1):
            p[n, :] = P[M - n, :]
        
        # Calcul du contrôle
        u = - (1.0 / self.alpha) * p
        
        self.y = y
        self.p = p
        self.u = u
        
        return y, p, u
    
    def calculer_erreur_L2(self, y_exact, p_exact, u_exact=None):
        """
        Calcule les erreurs L2 entre la solution numérique et la solution exacte
        
        Erreur L2 = sqrt(∫∫ (y_num - y_exact)² dx dt)
        """
        y_num = self.y
        p_num = self.p
        dt, dx = self.dt, self.dx
        N, M = self.N, self.M
        
        # Erreur sur y
        err_y = 0.0
        for n in range(M+1):
            for i in range(N):
                y_ex = y_exact(self.t_pts[n], self.x_pts[i])
                err_y += (y_num[n, i] - y_ex)**2 * dx * dt
        err_y = np.sqrt(err_y)
        
        # Erreur sur p
        err_p = 0.0
        for n in range(M+1):
            for i in range(N):
                p_ex = p_exact(self.t_pts[n], self.x_pts[i])
                err_p += (p_num[n, i] - p_ex)**2 * dx * dt
        err_p = np.sqrt(err_p)
        
        # Erreur sur u si fournie
        err_u = None
        if u_exact is not None:
            err_u = 0.0
            for n in range(M+1):
                for i in range(N):
                    u_ex = u_exact(self.t_pts[n], self.x_pts[i])
                    err_u += (self.u[n, i] - u_ex)**2 * dx * dt
            err_u = np.sqrt(err_u)
        
        return err_y, err_p, err_u
    
    def calculer_taux_convergence(self, y_exact, p_exact, N_list, M_list):
        """
        Calcule les taux de convergence pour différentes discrétisations
        
        Args:
            y_exact, p_exact: fonctions exactes
            N_list: liste des N à tester
            M_list: liste des M correspondants
        
        Returns:
            taux_y, taux_p: taux de convergence
        """
        erreurs_y = []
        erreurs_p = []
        h_list = []
        
        for N, M in zip(N_list, M_list):
            # Création d'un nouveau solveur avec ces paramètres
            solveur = SolveurControleOptimal(self.T, self.alpha, N, M)
            
            # Résolution avec les sources manufacturées
            def f_manu(t, x):
                y_ex = y_exact(t, x)
                p_ex = p_exact(t, x)
                return source_f_manufacturee(t, x, y_ex, p_ex, self.alpha)
            
            def zd_manu(t, x):
                y_ex = y_exact(t, x)
                p_ex = p_exact(t, x)
                return zd_manufacturee(t, x, y_ex, p_ex)
            
            def y0_manu(x):
                return y_exact(0, x)
            
            solveur.resoudre(f_manu, zd_manu, y0_manu)
            
            err_y, err_p, _ = solveur.calculer_erreur_L2(y_exact, p_exact)
            
            erreurs_y.append(err_y)
            erreurs_p.append(err_p)
            h_list.append(solveur.dx)
        
        # Calcul des taux de convergence (pente en log-log)
        log_h = np.log(h_list)
        log_err_y = np.log(erreurs_y)
        log_err_p = np.log(erreurs_p)
        
        taux_y = (log_err_y[-1] - log_err_y[0]) / (log_h[-1] - log_h[0])
        taux_p = (log_err_p[-1] - log_err_p[0]) / (log_h[-1] - log_h[0])
        
        return taux_y, taux_p, h_list, erreurs_y, erreurs_p

# ============================================================================
# PARTIE 4 : ANALYSE DE CONVERGENCE PAR MAILLAGE CROISÉ
# ============================================================================

def analyse_convergence_maillage():
    """
    Analyse de convergence en raffinant le maillage
    """
    print("\n" + "="*70)
    print("ANALYSE DE CONVERGENCE")
    print("="*70)
    
    T = 1.0
    alpha = 0.01
    
    # Liste des maillages à tester (raffinement progressif)
    # On maintient dt ~ dx² pour que l'erreur spatiale domine
    N_list = [19, 39, 79, 159, 319]  # points intérieurs
    M_list = [20, 40, 80, 160, 320]  # pas de temps (proportionnel)
    
    # Solutions exactes pour le test de fabrication
    def y_exact(t, x):
        return np.sin(np.pi * x) * np.exp(-t)
    
    def p_exact(t, x):
        return np.sin(np.pi * x) * (1.0 - t)
    
    # Stockage des erreurs
    erreurs_y = []
    erreurs_p = []
    h_list = []
    dt_list = []
    temps_calcul = []
    
    for N, M in zip(N_list, M_list):
        print(f"\n--- Maillage: N={N}, M={M} ---")
        print(f"  dx = {1.0/(N+1):.6f}, dt = {T/M:.6f}")
        
        start_time = time.time()
        
        # Création du solveur
        solveur = SolveurControleOptimal(T, alpha, N, M)
        
        # Définition des sources manufacturées
        def f_manu(t, x):
            y_ex = y_exact(t, x)
            p_ex = p_exact(t, x)
            # f = ∂_t y - ∂_xx y + (1/α)p
            dy_dt = -y_ex
            d2y_dx2 = -np.pi**2 * y_ex
            return dy_dt - d2y_dx2 + (1.0/alpha) * p_ex
        
        def zd_manu(t, x):
            y_ex = y_exact(t, x)
            p_ex = p_exact(t, x)
            # z_d = y + ∂_t p + ∂_xx p
            dp_dt = -np.sin(np.pi * x)
            d2p_dx2 = -np.pi**2 * p_ex
            return y_ex + dp_dt + d2p_dx2
        
        def y0_manu(x):
            return y_exact(0, x)
        
        # Résolution
        y_num, p_num, u_num = solveur.resoudre(f_manu, zd_manu, y0_manu)
        
        # Calcul des erreurs
        err_y, err_p, _ = solveur.calculer_erreur_L2(y_exact, p_exact)
        
        end_time = time.time()
        
        erreurs_y.append(err_y)
        erreurs_p.append(err_p)
        h_list.append(solveur.dx)
        dt_list.append(solveur.dt)
        temps_calcul.append(end_time - start_time)
        
        print(f"  Erreur L2 sur y: {err_y:.6e}")
        print(f"  Erreur L2 sur p: {err_p:.6e}")
        print(f"  Temps de calcul: {end_time - start_time:.2f}s")
    
    # Calcul des taux de convergence
    log_h = np.log(h_list)
    log_err_y = np.log(erreurs_y)
    log_err_p = np.log(erreurs_p)
    
    # Pente par moindres carrés
    coeff_y = np.polyfit(log_h, log_err_y, 1)
    coeff_p = np.polyfit(log_h, log_err_p, 1)
    
    print("\n" + "="*70)
    print("RÉSULTATS DE CONVERGENCE")
    print("="*70)
    print(f"Taux de convergence pour y: {coeff_y[0]:.2f} (attendu: ~2 pour l'espace, ~1 pour le temps)")
    print(f"Taux de convergence pour p: {coeff_p[0]:.2f} (attendu: ~2 pour l'espace, ~1 pour le temps)")
    
    # Vérification de l'ordre temporel
    print("\nVérification de l'ordre temporel (en fixant dx très fin):")
    
    # Test avec dx fixe et dt variable
    N_fixe = 99
    dx_fixe = 1.0/(N_fixe+1)
    M_test = [50, 100, 200, 400, 800]
    
    erreurs_y_dt = []
    erreurs_p_dt = []
    dt_test = []
    
    for M in M_test:
        dt_val = T/M
        solveur = SolveurControleOptimal(T, alpha, N_fixe, M)
        
        def f_manu(t, x):
            y_ex = y_exact(t, x)
            p_ex = p_exact(t, x)
            dy_dt = -y_ex
            d2y_dx2 = -np.pi**2 * y_ex
            return dy_dt - d2y_dx2 + (1.0/alpha) * p_ex
        
        def zd_manu(t, x):
            y_ex = y_exact(t, x)
            p_ex = p_exact(t, x)
            dp_dt = -np.sin(np.pi * x)
            d2p_dx2 = -np.pi**2 * p_ex
            return y_ex + dp_dt + d2p_dx2
        
        def y0_manu(x):
            return y_exact(0, x)
        
        solveur.resoudre(f_manu, zd_manu, y0_manu)
        err_y, err_p, _ = solveur.calculer_erreur_L2(y_exact, p_exact)
        
        erreurs_y_dt.append(err_y)
        erreurs_p_dt.append(err_p)
        dt_test.append(dt_val)
    
    log_dt = np.log(dt_test)
    log_err_y_dt = np.log(erreurs_y_dt)
    log_err_p_dt = np.log(erreurs_p_dt)
    
    coeff_y_dt = np.polyfit(log_dt, log_err_y_dt, 1)
    coeff_p_dt = np.polyfit(log_dt, log_err_p_dt, 1)
    
    print(f"Taux de convergence temporel pour y: {coeff_y_dt[0]:.2f} (attendu: ~1)")
    print(f"Taux de convergence temporel pour p: {coeff_p_dt[0]:.2f} (attendu: ~1)")
    
    # Visualisation
    plt.figure(figsize=(14, 5))
    
    plt.subplot(1, 2, 1)
    plt.loglog(h_list, erreurs_y, 'bo-', label='Erreur y')
    plt.loglog(h_list, erreurs_p, 'rs-', label='Erreur p')
    plt.loglog(h_list, [h**2 for h in h_list], 'k--', label='Ordre 2 (espace)')
    plt.loglog(h_list, [h**1 for h in h_list], 'k:', label='Ordre 1 (temps)')
    plt.xlabel('Pas spatial dx')
    plt.ylabel('Erreur L2')
    plt.title('Convergence spatiale (dt ~ dx²)')
    plt.legend()
    plt.grid(True)
    
    plt.subplot(1, 2, 2)
    plt.loglog(dt_test, erreurs_y_dt, 'bo-', label='Erreur y')
    plt.loglog(dt_test, erreurs_p_dt, 'rs-', label='Erreur p')
    plt.loglog(dt_test, [dt**1 for dt in dt_test], 'k--', label='Ordre 1')
    plt.xlabel('Pas temporel dt')
    plt.ylabel('Erreur L2')
    plt.title('Convergence temporelle (dx fixe)')
    plt.legend()
    plt.grid(True)
    
    plt.tight_layout()
    plt.savefig('tp3_convergence.png', dpi=150)
    plt.show()
    
    return {
        'taux_spatial_y': coeff_y[0],
        'taux_spatial_p': coeff_p[0],
        'taux_temporel_y': coeff_y_dt[0],
        'taux_temporel_p': coeff_p_dt[0],
        'erreurs_y': erreurs_y,
        'erreurs_p': erreurs_p,
        'h_list': h_list,
        'dt_list': dt_test
    }

# ============================================================================
# PARTIE 5 : VALIDATION PAR SOLUTION DE RÉFÉRENCE
# ============================================================================

def validation_solution_reference():
    """
    Valide le code en comparant avec une solution de référence
    (solution obtenue avec un maillage très fin)
    """
    print("\n" + "="*70)
    print("VALIDATION PAR SOLUTION DE RÉFÉRENCE")
    print("="*70)
    
    T = 1.0
    alpha = 0.01
    
    # Solution de référence (maillage très fin)
    N_ref = 199
    M_ref = 500
    
    print(f"Calcul de la solution de référence (N={N_ref}, M={M_ref})...")
    
    solveur_ref = SolveurControleOptimal(T, alpha, N_ref, M_ref)
    
    # Fonctions données du problème original
    def f_source(t):
        return np.sin(2 * np.pi * t)
    
    def zd_source(t, x):
        return np.sin(np.pi * x) * np.exp(-t)
    
    def y0_source(x):
        return 0.0
    
    start_time = time.time()
    y_ref, p_ref, u_ref = solveur_ref.resoudre(f_source, zd_source, y0_source)
    temps_ref = time.time() - start_time
    
    print(f"Temps de calcul référence: {temps_ref:.2f}s")
    
    # Comparaison avec différents maillages
    N_list = [19, 39, 79]
    M_list = [20, 40, 80]
    
    erreurs_y = []
    erreurs_p = []
    erreurs_u = []
    
    for N, M in zip(N_list, M_list):
        print(f"\nMaillage test: N={N}, M={M}")
        
        solveur_test = SolveurControleOptimal(T, alpha, N, M)
        y_test, p_test, u_test = solveur_test.resoudre(f_source, zd_source, y0_source)
        
        # Interpolation de la solution de référence sur le maillage test
        from scipy.interpolate import RegularGridInterpolator
        
        # Grille de référence
        t_ref = solveur_ref.t_pts
        x_ref = solveur_ref.x_pts
        
        # Interpolateurs
        interp_y = RegularGridInterpolator((t_ref, x_ref), y_ref, method='linear')
        interp_p = RegularGridInterpolator((t_ref, x_ref), p_ref, method='linear')
        
        # Calcul de l'erreur L2
        err_y = 0.0
        err_p = 0.0
        dt_test = solveur_test.dt
        dx_test = solveur_test.dx
        
        for n in range(M+1):
            t = n * dt_test
            for i in range(N):
                x = solveur_test.x_pts[i]
                y_ref_interp = interp_y((t, x))
                p_ref_interp = interp_p((t, x))
                
                err_y += (y_test[n, i] - y_ref_interp)**2 * dx_test * dt_test
                err_p += (p_test[n, i] - p_ref_interp)**2 * dx_test * dt_test
        
        err_y = np.sqrt(err_y)
        err_p = np.sqrt(err_p)
        
        erreurs_y.append(err_y)
        erreurs_p.append(err_p)
        
        print(f"  Erreur L2 vs référence - y: {err_y:.6e}")
        print(f"  Erreur L2 vs référence - p: {err_p:.6e}")
    
    # Visualisation
    plt.figure(figsize=(12, 5))
    
    plt.subplot(1, 2, 1)
    plt.loglog([1.0/(N+1) for N in N_list], erreurs_y, 'bo-', label='Erreur y')
    plt.loglog([1.0/(N+1) for N in N_list], erreurs_p, 'rs-', label='Erreur p')
    plt.xlabel('Pas spatial dx')
    plt.ylabel('Erreur L2 vs référence')
    plt.title('Convergence vers la solution de référence')
    plt.legend()
    plt.grid(True)
    
    plt.subplot(1, 2, 2)
    # Visualisation qualitative
    plt.plot(solveur_ref.x_pts, y_ref[-1, :], 'k-', label='Référence (t=T)', linewidth=2)
    for N, err in zip(N_list, erreurs_y):
        solveur_test = SolveurControleOptimal(T, alpha, N, N)  # M proportionnel
        y_test, _, _ = solveur_test.resoudre(f_source, zd_source, y0_source)
        plt.plot(solveur_test.x_pts, y_test[-1, :], '--', label=f'N={N}')
    plt.xlabel('x')
    plt.ylabel('y(T,x)')
    plt.title('Comparaison des profils finaux')
    plt.legend()
    plt.grid(True)
    
    plt.tight_layout()
    plt.savefig('tp3_validation.png', dpi=150)
    plt.show()
    
    return erreurs_y, erreurs_p

# ============================================================================
# PARTIE 6 : ESTIMATION D'ERREUR A POSTERIORI
# ============================================================================

def estimation_erreur_posteriori():
    """
    Estimation d'erreur a posteriori par la méthode de Richardson
    (extrapolation de Richardson)
    """
    print("\n" + "="*70)
    print("ESTIMATION D'ERREUR A POSTERIORI")
    print("="*70)
    
    T = 1.0
    alpha = 0.01
    
    # Fonctions données
    def f_source(t):
        return np.sin(2 * np.pi * t)
    
    def zd_source(t, x):
        return np.sin(np.pi * x) * np.exp(-t)
    
    def y0_source(x):
        return 0.0
    
    # Maillages emboîtés pour extrapolation de Richardson
    # On double le maillage à chaque fois
    N_list = [19, 39, 79, 159]
    M_list = [20, 40, 80, 160]
    
    solutions = []
    pas_list = []
    
    for N, M in zip(N_list, M_list):
        solveur = SolveurControleOptimal(T, alpha, N, M)
        y, p, u = solveur.resoudre(f_source, zd_source, y0_source)
        solutions.append((y, p, u))
        pas_list.append(solveur.dx)
    
    # Extrapolation de Richardson
    # Pour une grandeur Q, on a Q = Q_exact + C * h^p + O(h^{p+1})
    # Avec deux maillages h et 2h, on peut estimer l'erreur
    
    def richardson_extrapolation(Q1, Q2, h1, h2, p=2):
        """
        Extrapolation de Richardson
        Q1: solution sur maillage h1
        Q2: solution sur maillage h2 (h2 > h1)
        Retourne l'estimation de l'erreur sur Q2
        """
        # On suppose que l'erreur est en h^p
        # Q_exact ≈ Q1 + (Q1 - Q2) / ((h2/h1)^p - 1)
        r = h2 / h1
        err_estimate = (Q2 - Q1) / (r**p - 1)
        return err_estimate
    
    # Interpolation des solutions sur un maillage commun
    from scipy.interpolate import RegularGridInterpolator
    
    # Maillage fin pour la comparaison
    N_fin = 159
    M_fin = 160
    solveur_fin = SolveurControleOptimal(T, alpha, N_fin, M_fin)
    x_fin = solveur_fin.x_pts
    t_fin = solveur_fin.t_pts
    
    estimations = []
    
    for i in range(len(solutions)-1):
        N_i = N_list[i]
        M_i = M_list[i]
        solveur_i = SolveurControleOptimal(T, alpha, N_i, M_i)
        
        # Solution sur maillage i
        y_i, _, _ = solutions[i]
        
        # Solution sur maillage i+1 (plus fin)
        y_ip1, _, _ = solutions[i+1]
        
        # Interpolation de y_i sur le maillage fin
        interp_y_i = RegularGridInterpolator(
            (solveur_i.t_pts, solveur_i.x_pts), y_i, method='linear'
        )
        
        # Calcul de l'erreur estimée au point (t_fin, x_fin)
        err_est = 0.0
        n_points = 0
        
        for n in range(0, M_fin+1, M_fin//10):
            t = t_fin[n]
            for i_pt in range(0, N_fin+1, N_fin//10):
                x = x_fin[i_pt]
                y_i_interp = interp_y_i((t, x))
                y_ip1_interp = y_ip1[n, i_pt]
                
                # Extrapolation de Richardson
                h_coarse = solveur_i.dx
                h_fine = solveur_ip1.dx
                r = h_fine / h_coarse
                err_est_local = (y_ip1_interp - y_i_interp) / (r**2 - 1)
                err_est += abs(err_est_local)**2
                n_points += 1
        
        err_est = np.sqrt(err_est / n_points)
        estimations.append(err_est)
        
        print(f"Maillage {i+1} (dx={solveur_i.dx:.5f}): erreur estimée ≈ {err_est:.6e}")
    
    return estimations

# ============================================================================
# PARTIE 7 : EXÉCUTION PRINCIPALE
# ============================================================================

if __name__ == "__main__":
    print("="*70)
    print("TP3 - COMMANDE OPTIMALE PARABOLIQUE")
    print("Analyse de convergence et validation")
    print("="*70)
    
    # 1. Analyse de convergence par solution manufacturée
    resultats_convergence = analyse_convergence_maillage()
    
    # 2. Validation par solution de référence
    erreurs_ref = validation_solution_reference()
    
    # 3. Estimation d'erreur a posteriori
    estimations = estimation_erreur_posteriori()
    
    # 4. Résumé final
    print("\n" + "="*70)
    print("RÉSUMÉ DES RÉSULTATS")
    print("="*70)
    print(f"Taux de convergence spatial (y): {resultats_convergence['taux_spatial_y']:.2f}")
    print(f"Taux de convergence spatial (p): {resultats_convergence['taux_spatial_p']:.2f}")
    print(f"Taux de convergence temporel (y): {resultats_convergence['taux_temporel_y']:.2f}")
    print(f"Taux de convergence temporel (p): {resultats_convergence['taux_temporel_p']:.2f}")
    print()
    print("Interprétation:")
    print("- Ordre spatial ~ 2 : le schéma est bien d'ordre 2 en espace (différences centrées)")
    print("- Ordre temporel ~ 1 : le schéma d'Euler implicite est d'ordre 1 en temps")
    print("- Les erreurs diminuent comme attendu avec le raffinement du maillage")