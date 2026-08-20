import jax
jax.config.update("jax_platform_name", "cpu")
jax.config.update('jax_enable_x64', True)
import jax.numpy as jnp

import numpy as np
import pennylane as qml
import optax

from qiskit import QuantumCircuit
from qiskit_aer import AerSimulator
from qiskit_aer.noise import NoiseModel, depolarizing_error, thermal_relaxation_error

import os


# ======================================================================
# ÉTAPE 1 : VQE SANS BRUIT (RECHERCHE DES VRAIS PARAMÈTRES OPTIMAUX)
# ======================================================================

NOM_MOLECULE = "HeH+"
print(f"\n================================= VQE pour {NOM_MOLECULE} =================================")

print(f"\n--- Étape 1 : VQE Idéal pour la molécule {NOM_MOLECULE} ---\n")

# Configuration du dossier datasets
DOSSIER_CODE = os.path.dirname(os.path.abspath(__file__))
RACINE_MEMOIRE = os.path.dirname(DOSSIER_CODE)
DOSSIER_DATASETS = os.path.join(RACINE_MEMOIRE, "datasets")

# Import de toutes les informations de la molécule
dataset = qml.data.load('qchem', molname=NOM_MOLECULE, folder_path=DOSSIER_DATASETS)[0]
H = dataset.hamiltonian
NUM_QUBITS = len(H.wires)
NUM_ELECTRONS = dataset.molecule.n_electrons

NUM_SAMPLES = 50

# État de Hartree-Fock (état théorique à améliorer)
hf = qml.qchem.hf_state(NUM_ELECTRONS, NUM_QUBITS)

# Définition du matériel
dev = qml.device("lightning.qubit", wires=NUM_QUBITS)

# Circuit VQE avec Ansatz "Hardware Efficient"
@qml.qnode(dev, interface="jax")
def circuit(params, wires):
    qml.BasisState(hf, wires=wires) # Initialisation Hartree-Fock
    for i in range(NUM_QUBITS):
        qml.RY(params[i], wires=i) # Porte RY
    for i in range(NUM_QUBITS - 1):
        qml.CNOT(wires=[i, i+1]) # Porte CNOT
    return qml.expval(H)

# Fonction de coût à minimiser
@jax.jit
def cost(params):
    return circuit(params, wires=range(NUM_QUBITS))

# Initialisation de l'optimisation classique
MAX_ITERATIONS = 100
opt = optax.sgd(learning_rate=0.4)

key = jax.random.PRNGKey(42)
theta = jax.random.uniform(key, shape=(NUM_QUBITS), minval=-0.1, maxval=0.1) 
opt_state = opt.init(theta)

print("\nLancement de l'optimiseur classique...")
for n in range(MAX_ITERATIONS):
    gradient = jax.grad(cost)(theta) # Calcul du gradient
    updates, opt_state = opt.update(gradient, opt_state)
    theta = optax.apply_updates(theta, updates) # Optimisation de theta
    
    energy = cost(theta)
    if n % 10 == 0:
        print(f"  Step = {n:3d},  Energy = {energy:.6f} Ha")

print(f"\nÉnergie initiale : {cost(jnp.zeros(NUM_QUBITS)):.6f} Ha")
print(f"Énergie finale   : {cost(theta):.6f} Ha")

# Stockage des paramètres optimaux
BASE_PARAMS = np.array(theta)
os.makedirs(DOSSIER_DATASETS, exist_ok=True)
nom_fichier_theta = os.path.join(DOSSIER_DATASETS, f"theta_optimal_{NOM_MOLECULE}.npy")
np.save(nom_fichier_theta, BASE_PARAMS)
print(f"Vrais paramètres optimaux trouvés : {BASE_PARAMS}")


# ======================================================================
# ÉTAPE 2 : GESTION DU BRUIT
# ======================================================================

print(f"\n\n--- Étape 2 : Gestion du bruit ---")

# Temps de cohérence physiques (en nano seconde)
# Valeurs typiques d'un processeur IBM (Règle physique : T2 <= 2*T1)
T1 = 3_000   # Amortissement d'amplitude
T2 = 1_000    # Déphasage

print(f"\nAmortissement d'amplitude : {T1} nanosecondes")
print(f"Déphasage de : {T2} nanosecondes")

# Temps d'exécution des portes quantiques (en nanosecondes)
# C'est le temps durant lequel les qubits sont exposés aux effets
time_1qubit = 50   # Durée d'une porte RY
time_2qubit = 300  # Durée d'une porte CNOT

print(f"\nTemps d'execution d'une porte quantique à 1 qubit : {time_1qubit} nanosecondes")
print(f"Temps d'execution d'une porte quantique à 2 qubits : {time_2qubit} nanosecondes")

# Taux d'erreurs de dépolarisation sur les portes
p_1qubit = 0.005
p_2qubit = 0.05

print(f"\nTaux d'erreurs de dépolarisation sur les portes quantiques à 1 qubit : {p_1qubit*100}%")
print(f"Taux d'erreurs de dépolarisation sur les portes quantiques à 2 qubits : {p_2qubit*100}%")

def creer_simulateur_bruite(t1, t2):

    t1_f = float(t1)
    t2_f = float(t2)

    # Création des canaux de relaxation thermique (T1 + T2) ??
    err_th_1q = thermal_relaxation_error(t1_f, t2_f, time_1qubit)
    err_th_2q = thermal_relaxation_error(t1_f, t2_f, time_2qubit).tensor(thermal_relaxation_error(t1_f, t2_f, time_2qubit))

    # Erreurs de dépolarisation
    err_dep_1q = depolarizing_error(p_1qubit, 1)
    err_dep_2q = depolarizing_error(p_2qubit, 2)

    # Composition des erreurs (relaxation thermiques puis dépolarisation)
    full_1q = err_th_1q.compose(err_dep_1q)
    full_2q = err_th_2q.compose(err_dep_2q)

    # Assemblage du modèle de bruit
    noise_model = NoiseModel()
    noise_model.add_all_qubit_quantum_error(full_1q, ["ry"])
    noise_model.add_all_qubit_quantum_error(full_2q, ["cx"])

    return AerSimulator(noise_model=noise_model, method="density_matrix")


# ======================================================================
# ÉTAPE 3 : GÉNÉRATION DES PAIRES D'ÉTATS BRUITÉS
# ======================================================================
print(f"\n\n--- Étape 3 : Génération de {NUM_SAMPLES} paires d'états bruités ---")


def fabriquer_circuit(params):

    qc = QuantumCircuit(NUM_QUBITS)

    # Mapping des qubits inversé pour correspondre à PennyLane
    # Qiskit qubit = NUM_QUBITS - 1 - PennyLane wire
    q_map = lambda i: NUM_QUBITS - 1 - i

    # Traduction de l'état de Hartree-Fock de PennyLane à Qiskit
    for i in range(NUM_QUBITS):
        if hf[i] == 1:
            qc.x(q_map(i))

    # Ansatz "Hardware Efficient"
    for i in range(NUM_QUBITS):
        qc.ry(params[i], q_map(i))
    for i in range(NUM_QUBITS - 1):
        qc.cx(q_map(i), q_map(i + 1))
    qc.save_density_matrix()

    return qc


dataset1 = np.zeros((NUM_SAMPLES, 2**NUM_QUBITS, 2**NUM_QUBITS), dtype=complex)
dataset2 = np.zeros((NUM_SAMPLES, 2**NUM_QUBITS, 2**NUM_QUBITS), dtype=complex)

rng_np = np.random.default_rng(42)

print("\nGénération des paires d'états mixtes bruités...")

for k in range(NUM_SAMPLES):

    # Tirage de temps de cohérence fluctuants autour de la moyenne
    t1_k = rng_np.normal(loc=T1, scale=1000)
    t2_k = rng_np.normal(loc=T2, scale=330)

    # Création du simulateur unique pour cet échantillon
    simulator_k = creer_simulateur_bruite(t1_k, t2_k)

    # Deux exécutions indépendantes (Noise2Noise) sous ces conditions de bruit
    qc1 = fabriquer_circuit(BASE_PARAMS)
    qc2 = fabriquer_circuit(BASE_PARAMS)

    res1 = simulator_k.run(qc1).result()
    res2 = simulator_k.run(qc2).result()

    dataset1[k] = np.array(res1.data()["density_matrix"].data)
    dataset2[k] = np.array(res2.data()["density_matrix"].data)

print("Génération terminée.\n\n")

# Stockage du dataset dans un fichier npy pour le QAE
np.save(os.path.join(DOSSIER_DATASETS, f"dataset_{NOM_MOLECULE}_1.npy"), dataset1)
np.save(os.path.join(DOSSIER_DATASETS, f"dataset_{NOM_MOLECULE}_2.npy"), dataset2)

print("=" * 80)
print(f"Succès ! Le VQE entier s'est exécuté.")
print("\nSauvegarde terminée.")
print(f"\nFormat des données : {dataset1.shape}")
print("=" * 80)