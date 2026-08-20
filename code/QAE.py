import jax
jax.config.update("jax_platform_name", "cpu")
jax.config.update("jax_enable_x64", True)
import jax.numpy as jnp

import numpy as np
import pennylane as qml
import optax
import matplotlib.pyplot as plt

import os

# ==============================================================================
# ÉTAPE 0 : CONFIGURATION
# ==============================================================================

NOM_MOLECULE = "HeH+"

# Récupération des données bruitées du VQE
DOSSIER_CODE = os.path.dirname(os.path.abspath(__file__))
RACINE_MEMOIRE = os.path.dirname(DOSSIER_CODE)
DOSSIER_DATASETS = os.path.join(RACINE_MEMOIRE, "datasets")

FICHIER_1 = os.path.join(DOSSIER_DATASETS, f"dataset_{NOM_MOLECULE}_1.npy")
FICHIER_2 = os.path.join(DOSSIER_DATASETS, f"dataset_{NOM_MOLECULE}_2.npy")
FICHIER_THETA = os.path.join(DOSSIER_DATASETS, f"theta_optimal_{NOM_MOLECULE}.npy")

# Architecture du QAE
N_LATENT = 1  # Qubits gardés (bottleneck) après l'encodeur (QAE[N,1,N])
L_BLOCKS = 3  # Nombre de blocs de l'ansatz RY_CZ

# Hyperparamètres d'entraînement
N_EPOCHS = 100
LEARNING_RATE = 0.05
TAILLE_BATCH = 32 # Nombre d'échantillons bruités tirés à chaque époque
FRACTION_TEST = 0.2 # Part du dataset réservée à l'évaluation finale
SEED = 42

# Fonctions déclarées dans la fonction de configuration via la méthode "global"
circuit_encodeur = None
circuit_decodeur = None

DOSSIER_SORTIE = os.path.join(RACINE_MEMOIRE, "resultats_qae")
os.makedirs(DOSSIER_SORTIE, exist_ok=True)


# ==============================================================================
# ÉTAPE 1 : CHARGEMENT DES DONNÉES
# ==============================================================================

def charger_dataset_bruite(fichier1, fichier2):
    # Chargement des matrices de densité bruitées
    data1 = np.load(fichier1)
    data2 = np.load(fichier2)
    print(f"Dataset bruité chargé : {data1.shape[0]} échantillons de matrices {data1.shape[1]}x{data1.shape[2]}")
    
    n_qubits_detecte = int(round(np.log2(data1.shape[1])))
    print(f"Nombre de qubits déduit du dataset : {n_qubits_detecte}\n")
    return data1, data2, n_qubits_detecte


def calculer_etat_ideal(nom_molecule, dossier_datasets, fichier_theta):
    
    # Reconstruction du vecteur d'état pur pour le calcul de fidélité
    theta_optimal = jnp.array(np.load(fichier_theta))

    # Rechargement des mêmes informations que dans le VQE
    dataset = qml.data.load('qchem', molname=nom_molecule, folder_path=dossier_datasets)[0]
    num_qubits = len(dataset.hamiltonian.wires)
    electrons = dataset.molecule.n_electrons
    hf = qml.qchem.hf_state(electrons, num_qubits)

    # Utilisation d'un device "parfait"
    dev_ideal = qml.device("default.qubit", wires=num_qubits)

    # Le circuit est emballé dans un décorateur du device
    # (Déclaration de l'ansatz. Le décorateur force l'ansatz à se lancer sur le device choisi dans ce dernier.)
    @qml.qnode(dev_ideal, interface="jax")
    def circuit_ideal(theta):
        qml.BasisState(hf, wires=range(num_qubits))
        for i in range(num_qubits):
            qml.RY(theta[i], wires=i)
        for i in range(num_qubits - 1):
            qml.CNOT(wires=[i, i + 1])
        return qml.state()

    psi_ideal = circuit_ideal(theta_optimal)
    print(f"Etat pur idéal reconstruit pour {num_qubits} qubits à partir de theta_optimal.")
    return psi_ideal


# ==============================================================================
# ÉTAPE 2 : CIRCUITS QUANTIQUES
# ==============================================================================

def ansatz_ry_cz(params, wires, n_blocs):

    # Ansatz utilisé pour l'encodeur et le décodeur du QAE
    # params : tableau de forme (n_blocs + 1, len(wires))

    n_wires = len(wires)

    for l in range(n_blocs):
        for i, w in enumerate(wires):
            qml.RY(params[l, i], wires=w)
        if n_wires > 1:
            for i in range(n_wires):
                qml.CZ(wires=[wires[i], wires[(i + 1) % n_wires]])
    # Rotation finale : aligne la base de mesure avant la trace partielle
    for i, w in enumerate(wires):
        qml.RY(params[n_blocs, i], wires=w)

# Fonction de configuration qui crée l'encodeur et le décodeur
def configurer_circuits_qae(n_qubits, n_latent, l_blocks):
    
    global circuit_encodeur, circuit_decodeur

    n_trash = n_qubits - n_latent

    # Utilisation d'un device qui gère les états mixtes
    dev_encodeur = qml.device("default.mixed", wires=n_qubits)
    dev_decodeur = qml.device("default.mixed", wires=n_qubits)

    @qml.qnode(dev_encodeur, interface="jax")
    def _circuit_encodeur(phi_enc, rho_in):
        
        # Chargement de la matrice d'état mixte 
        qml.QubitDensityMatrix(rho_in, wires=range(n_qubits))
        
        # Exécution de l'ansatz
        ansatz_ry_cz(phi_enc, wires=range(n_qubits), n_blocs=l_blocks)

        # Opération équivalente à une trace partielle
        return qml.density_matrix(wires=range(n_latent))


    @qml.qnode(dev_decodeur, interface="jax")
    def _circuit_decodeur(phi_dec, rho_latent):
        
        # Réinitialisation des qubits qui ont été tracés à 0
        rho_zero_trash = jnp.zeros((2 ** n_trash, 2 ** n_trash), dtype=complex)
        rho_zero_trash = rho_zero_trash.at[0, 0].set(1.0 + 0.0j)

        # Produit de Kronecker pour reconstruire une matrice dans l'espace total (qui est le produit tensoriel de l'espace latent et de l'espace poubelle)
        rho_init = jnp.kron(rho_latent, rho_zero_trash)

        # Chargement de la matrice d'état mixte
        qml.QubitDensityMatrix(rho_init, wires=range(n_qubits))

        #Exécution de l'ansatz
        ansatz_ry_cz(phi_dec, wires=range(n_qubits), n_blocs=l_blocks)

        return qml.density_matrix(wires=range(n_qubits))

    circuit_encodeur = _circuit_encodeur
    circuit_decodeur = _circuit_decodeur

    print(f"\nCircuits QAE[{n_qubits},{n_latent},{n_qubits}] construits ({n_qubits} qubits, {n_latent} latent, {n_trash} trash, L={l_blocks}).")


# ==============================================================================
# ÉTAPE 3 : FONCTION DE COÛT ET DE FIDÉLITÉ
# ==============================================================================

# Recouvrement de Hilbert Schmidt
# (équivalent de la fidélité pour deux états mixtes)
def overlap_hilbert_schmidt(rho1, rho2):
    # Tr(rho1 @ rho2)
    return jnp.real(jnp.trace(rho1 @ rho2))


def cout_un_echantillon(phi_enc, phi_dec, rho1, rho2):
    # Perte = 1 - fidélité
    rho_latent = circuit_encodeur(phi_enc, rho1)
    rho_reconstruit = circuit_decodeur(phi_dec, rho_latent)
    return 1.0 - overlap_hilbert_schmidt(rho_reconstruit, rho2)

# Fonction de coût à minimiser
def cout_moyen_batch(params, batch1, batch2):
    # Jax optimise le calcul grâce à vmap (plus rapide que for ici)
    phi_enc, phi_dec = params
    pertes = jax.vmap(
        lambda r1, r2: cout_un_echantillon(phi_enc, phi_dec, r1, r2)
    )(batch1, batch2)
    return jnp.mean(pertes)


# ==============================================================================
# ÉTAPE 4 : BOUCLE D'ENTRAÎNEMENT
# ==============================================================================

def entrainer_qae(train1, train2, n_epochs, n_qubits, taille_batch, learning_rate, seed):
    
    # Gestion du hasard avec jax
    key = jax.random.PRNGKey(seed)
    key_enc, key_dec = jax.random.split(key, 2)

    # Initialisation des paramètres
    forme_enc = (L_BLOCKS + 1, n_qubits)
    forme_dec = (L_BLOCKS + 1, n_qubits)

    # Proche de zéro pour éviter les barren plateaus
    phi_enc = 0.1 * jax.random.normal(key_enc, shape=forme_enc)
    phi_dec = 0.1 * jax.random.normal(key_dec, shape=forme_dec)
    params = (phi_enc, phi_dec)

    # Adam est suffisament efficace et SPSA n'est pas implémenté par jax (mais possible de le recoder à la main)
    optimiseur = optax.adam(learning_rate)
    etat_opt = optimiseur.init(params)

    # Coût et gradient
    cout_et_grad_jit = jax.jit(jax.value_and_grad(cout_moyen_batch))

    # Sélection d'un batch d'entraînement
    n_disponibles = train1.shape[0]

    historique_perte = np.zeros(n_epochs)
    historique_fidelite = np.zeros(n_epochs)

    rng_np = np.random.default_rng(seed)

    print(f"\nDébut de l'entraînement...")

    for epoch in range(n_epochs):

        # Tirage d'un batch
        idx = rng_np.choice(n_disponibles, size=taille_batch, replace=False)
        rho_batch_1 = jnp.array(train1[idx])
        rho_batch_2 = jnp.array(train2[idx])

        # Optimisation
        perte, grads = cout_et_grad_jit(params, rho_batch_1, rho_batch_2)
        updates, etat_opt = optimiseur.update(grads, etat_opt)
        params = optax.apply_updates(params, updates)

        # Préparation des graphiques
        perte = float(perte)
        fidelite_moyenne = 1.0 - perte
        historique_perte[epoch] = perte
        historique_fidelite[epoch] = fidelite_moyenne

        if epoch % 10 == 0 or epoch == n_epochs - 1:
            print(f"  Époque {epoch:4d} | Perte = {perte:.6f} | "
                  f"Fidélité moyenne (batch) = {fidelite_moyenne:.6f}")
    
    return params, historique_perte, historique_fidelite


# ==============================================================================
# ÉTAPE 5 : ÉVALUATION — FIDÉLITÉ AVANT / APRÈS SUR LE JEU DE TEST
# ==============================================================================

def fidelite_etat_pur(psi, rho):
    # F(|psi>, rho) = <psi| rho |psi>
    return jnp.real(jnp.vdot(psi, rho @ psi))

# Calcul la fidélité à la fin en faisant passer les données test dans le QAE entrainé
def evaluer_qae(params, test1, psi_ideal):

    phi_enc, phi_dec = params
    n_test = test1.shape[0]

    fidelites_avant = np.zeros(n_test)
    fidelites_apres = np.zeros(n_test)

    for k in range(n_test):
        rho_k = jnp.array(test1[k])

        fidelites_avant[k] = float(fidelite_etat_pur(psi_ideal, rho_k))

        rho_latent = circuit_encodeur(phi_enc, rho_k)
        rho_debruite = circuit_decodeur(phi_dec, rho_latent)
        fidelites_apres[k] = float(fidelite_etat_pur(psi_ideal, rho_debruite))

    return fidelites_avant, fidelites_apres


# ==============================================================================
# ÉTAPE 6 : VISUALISATION
# ==============================================================================

def tracer_resultats(historique_perte, historique_fidelite,
                      fidelites_bruitees, fidelites_debruitees, dossier_sortie):
    fig, axes = plt.subplots(1, 3, figsize=(16, 5))

    # Courbe de perte
    axes[0].plot(historique_perte, color="tab:red")
    axes[0].set_xlabel("Epoque")
    axes[0].set_ylabel("Perte (1 - fidélité)")
    axes[0].set_title("Convergence de l'entraînement du QAE")
    axes[0].grid(alpha=0.3)

    # Fidélité moyenne sur le batch au fil de l'entraînement
    axes[1].plot(historique_fidelite, color="tab:blue")
    axes[1].set_xlabel("Epoque")
    axes[1].set_ylabel("Fidélité moyenne (batch)")
    axes[1].set_title("Fidélité pendant l'entraînement")
    axes[1].set_ylim(0, 1.02)
    axes[1].grid(alpha=0.3)

    # Comparaison des fidélités bruitée et débruitée sur le jeu de test
    donnees = [fidelites_bruitees, fidelites_debruitees]
    axes[2].boxplot(donnees, showmeans=True)
    axes[2].set_xticklabels(["Bruité", "Débruité (QAE)"])
    axes[2].set_ylabel("Fidélité avec l'état idéal")
    axes[2].set_title(f"Effet du débruitage (n={len(fidelites_bruitees)} échantillons test)")
    axes[2].set_ylim(0, 1.02)
    axes[2].grid(alpha=0.3)

    plt.tight_layout()
    chemin_fig = os.path.join(dossier_sortie, "resultats_qae.png")
    plt.savefig(chemin_fig, dpi=150)
    print(f"\nFigure sauvegardée : {chemin_fig}")
    plt.show()


# ==============================================================================
# ÉTAPE 7 : MAIN
# ==============================================================================

def main():
    print(f"============================ QAE pour débruitage de {NOM_MOLECULE} ============================")
    print(f"\n--- Étape 1 : Configuration du QAE ---\n")

    # Chargement des données
    data1, data2, N_QUBITS = charger_dataset_bruite(FICHIER_1, FICHIER_2)
    psi_ideal = calculer_etat_ideal(NOM_MOLECULE, DOSSIER_DATASETS, FICHIER_THETA)

    # Construction des circuits QAE (dépend du nombre de qubits réels)
    configurer_circuits_qae(n_qubits=N_QUBITS, n_latent=N_LATENT, l_blocks=L_BLOCKS)

    print(f"\n\n--- Étape 2 : Entraînement du QAE ---\n")

    # Split train/test
    rng = np.random.default_rng(SEED)
    n_total = data1.shape[0]
    indices = rng.permutation(n_total) # Lutte contre des biais d'ordre.
    n_test = max(1, int(FRACTION_TEST * n_total))
    idx_test, idx_train = indices[:n_test], indices[n_test:]

    train1, train2 = data1[idx_train], data2[idx_train]
    test1, test2 = data1[idx_test], data2[idx_test]
    print(f"Split : {len(idx_train)} paires train / {len(idx_test)} paires test")

    # Entraînement
    params_finaux, historique_perte, historique_fidelite = entrainer_qae(
        train1, train2,
        n_epochs=N_EPOCHS,
        n_qubits=N_QUBITS,
        taille_batch=TAILLE_BATCH,
        learning_rate=LEARNING_RATE,
        seed=SEED,
    )

    print("\n\n--- Étape 3 : Résultats finaux et affichage ---")
    # Évaluation finale sur le jeu de test
    fidelites_bruitees, fidelites_debruitees = evaluer_qae(params_finaux, test1, psi_ideal)

    print(f"\nFidélité moyenne AVANT débruitage  : {fidelites_bruitees.mean():.4f} ± {fidelites_bruitees.std():.4f}")
    print(f"Fidélité moyenne APRÈS débruitage  : {fidelites_debruitees.mean():.4f} ± {fidelites_debruitees.std():.4f}")
    print(f"Gain moyen de fidélité             : {(fidelites_debruitees - fidelites_bruitees).mean():+.4f}\n\n")

    print("=" * 80)
    print(f"Succès ! Le QAE entier s'est exécuté.")

    # Graphiques
    tracer_resultats(historique_perte, historique_fidelite, fidelites_bruitees, fidelites_debruitees, DOSSIER_SORTIE)
    print("=" * 80)


if __name__ == "__main__":
    main()
