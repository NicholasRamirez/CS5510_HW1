# Homework 1: Data Privacy
# Nicholas Ramirez-Ornelas
# Starter code

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import os

#Problem Setup

# Load dataset
DATA_FILE = "fake_healthcare_dataset_sample100.csv"

try:
    # Try loading locally
    if os.path.exists(DATA_FILE):
        print(f"Loaded local dataset: {DATA_FILE}")
        data = pd.read_csv(DATA_FILE)
    else:
        # Fallback: Colab manual upload if not found
        from google.colab import files
        print("Please upload fake_healthcare_dataset_sample100.csv")
        uploaded = files.upload()
        data = pd.read_csv(DATA_FILE)
except Exception as e:
    raise RuntimeError(f"Could not load dataset: {e}")

# names of public identifier columns
pub = ["age", "sex", "blood", "admission"]

# variable to reconstruct
target = "result"

def execute_subsetsums_exact(predicates):
    """Count the number of patients that satisfy each predicate.
    Resembles a public query interface on a sequestered dataset.
    Computed as in equation (1).

    :param predicates: a list of predicates on the public variables
    :returns a 1-d np.ndarray of exact answers the subset sum queries"""
    return data[target].values @ np.stack([pred(data) for pred in predicates], axis=1)


def make_random_predicate():
    """Returns a (pseudo)random predicate function by hashing public identifiers."""
    prime = 2003
    desc = np.random.randint(prime, size=len(pub))
    # this predicate maps data into a 1-d ndarray of booleans
    #   (where `@` is the dot product and `%` modulus)
    return lambda data: ((data[pub].values @ desc) % prime % 2).astype(bool)

# 1. Reconstruction Attack Part A
# TODO: Write the reconstruction function!
def reconstruction_attack(data_pub, predicates, answers):
    """Reconstructs a target column based on the `answers` to queries about `data`.

    :param data_pub: data of length n consisting of public identifiers
    :param predicates: a list of k predicate functions
    :param answers: a list of k answers to a query on data filtered by the k predicates
    :return 1-dimensional boolean ndarray"""

    # Build (n x k) matrix showing who matched query
    # shape (num_people, num_queries)
    predicate_matrix = np.column_stack([pred(data_pub).astype(float) for pred in predicates])

    query_answers = np.array(answers, dtype=float)
    estimated_results, *_ = np.linalg.lstsq(predicate_matrix.T, query_answers, rcond=None)
    # [0,1] Clip
    estimated_results = np.clip(estimated_results, 0.0, 1.0)
    reconstructed_results = (estimated_results >= 0.5)

    return reconstructed_results

def reconstruction_attack_test(num_trials: int = 10, seed: int = 42):
    # Part A tests

    random_number_generator = np.random.RandomState(seed)
    n = len(data)
    successes = []

    # 2n random predicates
    for trial in range(num_trials):
      np.random.seed(random_number_generator.randint(0, 10**9))
      predicates = [make_random_predicate() for _ in range(2 * n)]

      # Exact answers / Reconstruction / Real Results
      answers = execute_subsetsums_exact(predicates)
      reconstructed = reconstruction_attack(data[pub], predicates, answers)
      real_results = data[target].values.astype(bool)

      success = (reconstructed == real_results).mean()
      successes.append(success)

      print(f"Trial {trial + 1} success rate: {success}")

    print(f"Average success rate over {num_trials} trial: {np.mean(successes)}")

reconstruction_attack_test(num_trials=10, seed=42)

# 1. Reconstruction Attack Part B

def execute_subsetsums_round(R, predicates):
    # Round each result to the nearest multiple of R
    true_result = execute_subsetsums_exact(predicates).astype(float)
    return np.round(true_result / R) * R

def execute_subsetsums_noise(sigma, predicates):
    # Add independent Gaussian noise of mean zero and variance o^2 to each result
    true_result = execute_subsetsums_exact(predicates).astype(float)
    random_noise = np.random.normal(0, sigma, size=true_result.shape)
    return true_result + random_noise

def execute_subsetsums_sample(t, predicates):
    # Given a parameter t {1, ..., n}, randomly subsample a set T consisting of t out
    # of the n rows and calculate all of the answers using ony the rows in T (scaling up by factor of n/t)

    n = len(data) #  n rows
    T = np.random.choice(n, size=t, replace=False) # subsample

    true_answers = data[target].values.astype(float)
    match_matrix = np.stack([pred(data) for pred in predicates], axis=1).astype(float)

    answers = true_answers[T] @ match_matrix[T, :]
    scaled_answers = (n / t) * answers
    return scaled_answers

# 1. Reconstruction Attack Part C (i.)

# Root mean squared error
def rmse(y_pred: np.ndarray, y_true: np.ndarray) -> float:
    y_pred = np.asarray(y_pred, dtype=float)
    y_true = np.asarray(y_true, dtype=float)
    return float(np.sqrt(np.mean((y_pred - y_true) ** 2)))

# Sucecssful reconstruction result
def reconstruction_success(defended_answers, predicates) -> float:
    reconstructed = reconstruction_attack(data[pub], predicates, defended_answers)
    real_results = data[target].values.astype(bool)
    return float((reconstructed == real_results).mean())

# Produce 2n random predicates for testing
def make_2n_predicates(n: int, seed: int | None = None):
    if seed is not None:
        np.random.seed(seed)
    return [make_random_predicate() for _ in range(2 * n)]

# Run several trials using the same predicates for all defenses in every trial
# Mean + std is computed from RMSE and Reconstruction success for each type
def evaluate_Ci(R=5, sigma=2.0, t=None, num_trials=5, seed=42, lock_noise_to_trial_seed=False):
    n = len(data)
    if t is None:
        t = n // 2

    rng = np.random.RandomState(seed)
    results = {
        "exact":  {"rmse": [], "success": []},
        "round":  {"rmse": [], "success": []},
        "noise":  {"rmse": [], "success": []},
        "sample": {"rmse": [], "success": []},
    }

    for _ in range(num_trials):
        trial_seed = int(rng.randint(0, 10**9))
        predicates = make_2n_predicates(n, seed=trial_seed)
        exact_answers = execute_subsetsums_exact(predicates)

        # Exact
        results["exact"]["rmse"].append(0.0)
        results["exact"]["success"].append(reconstruction_success(exact_answers, predicates))

        # Round defense
        round_answers = execute_subsetsums_round(R, predicates)
        results["round"]["rmse"].append(rmse(round_answers, exact_answers))
        results["round"]["success"].append(reconstruction_success(round_answers, predicates))

        # Noise defense
        if lock_noise_to_trial_seed:
            np.random.seed(trial_seed)
        noise_answers = execute_subsetsums_noise(sigma, predicates)
        results["noise"]["rmse"].append(rmse(noise_answers, exact_answers))
        results["noise"]["success"].append(reconstruction_success(noise_answers, predicates))

        # Sample defense
        sample_answers = execute_subsetsums_sample(t, predicates)
        results["sample"]["rmse"].append(rmse(sample_answers, exact_answers))
        results["sample"]["success"].append(reconstruction_success(sample_answers, predicates))

    summary = {}
    for name, vals in results.items():
        rmse_arr = np.array(vals["rmse"], dtype=float)
        succ_arr = np.array(vals["success"], dtype=float)
        summary[name] = {
            "rmse_mean": float(rmse_arr.mean()),
            "rmse_std":  float(rmse_arr.std(ddof=0)),
            "success_mean": float(succ_arr.mean()),
            "success_std":  float(succ_arr.std(ddof=0)),
            "k": 2 * n,
            "params": {"R": R, "sigma": sigma, "t": t} if name != "exact" else {},
        }
    return summary


# Print
def print_Ci(summary):
    def fmt(v): return f"{v:.3f}" if isinstance(v, float) else v
    for name in ("exact", "round", "noise", "sample"):
        s = summary[name]
        params = s.get("params", {})
        param_str = (" " + ", ".join(f"{k}={v}" for k, v in params.items())) if params else ""
        print(f"[{name.upper()}{param_str}] k={s['k']}")
        print(f"  RMSE:   mean={fmt(s['rmse_mean'])}  std={fmt(s['rmse_std'])}")
        print(f"  Success: mean={fmt(s['success_mean'])}  std={fmt(s['success_std'])}")
        print()

summary = evaluate_Ci(R=5, sigma=2.0, t=len(data)//2, num_trials=5, seed=42)
print_Ci(summary)

# 1. Reconstruction Attack Part C (ii.)

# Sweep rounding parameter R through range of values
# For R, randomized trials ran
# Recording Average RMSE between rounded and exact and Average reconstruction success

def sweep_round(R_values=None, num_trials=10, seed=42):
    n = len(data)
    if R_values is None:
        R_values = range(1, n + 1)
    rng = np.random.RandomState(seed)
    out = []
    for R in R_values:
        rmses, succs = [], []
        for _ in range(num_trials):
            trial_seed = int(rng.randint(0, 10**9))
            predicates = make_2n_predicates(n, seed=trial_seed)
            exact_answers = execute_subsetsums_exact(predicates)
            defended_answers = execute_subsetsums_round(R, predicates)
            rmses.append(rmse(defended_answers, exact_answers))
            succs.append(reconstruction_success(defended_answers, predicates))
        out.append({
            "R": int(R),
            "rmse_mean": np.mean(rmses),
            "rmse_std": np.std(rmses),
            "success_mean": np.mean(succs),
            "success_std": np.std(succs),
            "k": 2 * n
        })
    return out

# Sweep noise standard deviation through range of values
# For sigma, evaluate how noise affects accuracy and reconstruction success

def sweep_noise(sigmas=None, num_trials=10, seed=42):
    n = len(data)
    if sigmas is None:
        sigmas = range(1, n + 1)
    rng = np.random.RandomState(seed)
    out = []
    for sigma in sigmas:
        rmses, succs = [], []
        for _ in range(num_trials):
            trial_seed = int(rng.randint(0, 10**9))
            predicates = make_2n_predicates(n, seed=trial_seed)
            exact_answers = execute_subsetsums_exact(predicates)
            defended_answers = execute_subsetsums_noise(float(sigma), predicates)
            rmses.append(rmse(defended_answers, exact_answers))
            succs.append(reconstruction_success(defended_answers, predicates))
        out.append({
            "sigma": float(sigma),
            "rmse_mean": np.mean(rmses),
            "rmse_std": np.std(rmses),
            "success_mean": np.mean(succs),
            "success_std": np.std(succs),
            "k": 2 * n
        })
    return out

# Sweep subsampling size t through range of values
# For t, evaluate how limit of size affects query accuracy and reconstruction success

def sweep_sample(t_values=None, num_trials=10, seed=42):
    n = len(data)
    if t_values is None:
        t_values = range(1, n + 1)
    rng = np.random.RandomState(seed)
    out = []
    for t in t_values:
        rmses, succs = [], []
        for _ in range(num_trials):
            trial_seed = int(rng.randint(0, 10**9))
            predicates = make_2n_predicates(n, seed=trial_seed)
            exact_answers = execute_subsetsums_exact(predicates)
            defended_answers = execute_subsetsums_sample(int(t), predicates)
            rmses.append(rmse(defended_answers, exact_answers))
            succs.append(reconstruction_success(defended_answers, predicates))
        out.append({
            "t": int(t),
            "rmse_mean": np.mean(rmses),
            "rmse_std": np.std(rmses),
            "success_mean": np.mean(succs),
            "success_std": np.std(succs),
            "k": 2 * n
        })
    return out

n = len(data)
NUM_TRIALS = 10
SEED = 42

# Sweep integer parameters 1..n for each defense
round_results  = sweep_round(R_values=range(1, n+1), num_trials=NUM_TRIALS, seed=SEED)
noise_results  = sweep_noise(sigmas=range(1, n+1),   num_trials=NUM_TRIALS, seed=SEED)
sample_results = sweep_sample(t_values=range(1, n+1), num_trials=NUM_TRIALS, seed=SEED)

df_round  = pd.DataFrame(round_results)
df_noise  = pd.DataFrame(noise_results)
df_sample = pd.DataFrame(sample_results)

# Averages of accuracy RMSE and Reconstruction success
plt.figure()
plt.plot(df_round["R"], df_round["rmse_mean"])
plt.xlabel("R")
plt.ylabel("Average RMSE")
plt.title("Rounding: Average Accuracy vs R")
plt.grid(True)
plt.savefig("rounding_accuracy_vs_R.png", dpi=300, bbox_inches='tight')
plt.show()

plt.figure()
plt.plot(df_round["R"], df_round["success_mean"])
plt.xlabel("R")
plt.ylabel("Average Reconstruction Success")
plt.title("Rounding: Success vs R")
plt.grid(True)
plt.savefig("rounding_success_vs_R.png", dpi=300, bbox_inches='tight')
plt.show()

plt.figure()
plt.plot(df_noise["sigma"], df_noise["rmse_mean"])
plt.xlabel("σ")
plt.ylabel("Average RMSE")
plt.title("Noise: Average Accuracy vs σ")
plt.grid(True)
plt.savefig("noise_accuracy_vs_sigma.png", dpi=300, bbox_inches='tight')
plt.show()

plt.figure()
plt.plot(df_noise["sigma"], df_noise["success_mean"])
plt.xlabel("σ")
plt.ylabel("Average Reconstruction Success")
plt.title("Noise: Success vs σ")
plt.grid(True)
plt.savefig("noise_success_vs_sigma.png", dpi=300, bbox_inches='tight')
plt.show()

plt.figure()
plt.plot(df_sample["t"], df_sample["rmse_mean"])
plt.xlabel("t (sample size)")
plt.ylabel("Average RMSE")
plt.title("Sampling: Average Accuracy vs t")
plt.grid(True)
plt.savefig("sampling_accuracy_vs_t.png", dpi=300, bbox_inches='tight')
plt.show()

plt.figure()
plt.plot(df_sample["t"], df_sample["success_mean"])
plt.xlabel("t (sample size)")
plt.ylabel("Average Reconstruction Success")
plt.title("Sampling: Success vs t")
plt.grid(True)
plt.savefig("sampling_success_vs_t.png", dpi=300, bbox_inches='tight')
plt.show()

# 1. Reconstruction Attack Part C (iii.)

# Rounding trade-off
plt.figure()
plt.plot(df_round["rmse_mean"], df_round["success_mean"], marker='o')
plt.xlabel("Accuracy (RMSE)")
plt.ylabel("Reconstruction Success")
plt.title("Trade-off: Rounding Defense")
plt.grid(True)
plt.savefig("tradeoff_rounding.png", dpi=300, bbox_inches='tight')
plt.show()

# Noise trade-off
plt.figure()
plt.plot(df_noise["rmse_mean"], df_noise["success_mean"], marker='o')
plt.xlabel("Accuracy (RMSE)")
plt.ylabel("Reconstruction Success")
plt.title("Trade-off: Noise Defense")
plt.grid(True)
plt.savefig("tradeoff_noise.png", dpi=300, bbox_inches='tight')
plt.show()

# Sampling trade-off
plt.figure()
plt.plot(df_sample["rmse_mean"], df_sample["success_mean"], marker='o')
plt.xlabel("Accuracy (RMSE)")
plt.ylabel("Reconstruction Success")
plt.title("Trade-off: Sampling Defense")
plt.grid(True)
plt.savefig("tradeoff_sampling.png", dpi=300, bbox_inches='tight')
plt.show()