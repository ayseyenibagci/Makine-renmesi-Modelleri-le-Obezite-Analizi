import urllib.request
import time
import math
import numpy as np
import pandas as pd


# =====================================================
# ESC VERİ SETLERİ VE BİLİNEN EN İYİ DEĞERLER
# =====================================================

ESC_INSTANCES = {
    "esc16a": 68,
    "esc16b": 292,
    "esc16c": 160,
    "esc16d": 16,
    "esc16e": 28,
    "esc16f": 0,
    "esc16g": 26,
    "esc16h": 996,
    "esc16i": 14,
    "esc16j": 8,
    "esc32a": 130,
    "esc32b": 168,
    "esc32c": 642,
    "esc32d": 200,
    "esc32e": 2,
    "esc32f": 64,
    "esc32g": 6,
    "esc32h": 438,
    "esc64a": 116,
    "esc128": 64
}

BASE_URL = "https://qaplib.mgi.polymtl.ca/data.d/"


# =====================================================
# QAPLIB VERİSİNİ İNDİR
# =====================================================

def download_qap_instance(name):

    url = BASE_URL + name + ".dat"

    with urllib.request.urlopen(url) as response:
        text = response.read().decode("utf-8")

    numbers = list(map(int, text.split()))

    n = numbers[0]

    total = n * n

    flow = np.array(
        numbers[1:1 + total]
    ).reshape(n, n)

    distance = np.array(
        numbers[1 + total:1 + 2 * total]
    ).reshape(n, n)

    return n, flow, distance


# =====================================================
# AMAÇ FONKSİYONU
# =====================================================

def objective(perm, flow, distance):

    n = len(perm)

    cost = 0

    for i in range(n):
        for j in range(n):

            cost += (
                flow[i, j] *
                distance[perm[i], perm[j]]
            )

    return cost


# =====================================================
# DETERMINISTIC PARAMETRELER
# =====================================================

def deterministic_parameters(n):

    return {
        "p_local": max(4, int(0.35 * n)),
        "tabu_tenure": max(5, int(1.5 * math.sqrt(n))),
        "elite_size": min(12, max(5, n // 2)),
        "max_iter": 120 * n,
        "no_improve_limit": 30 * n,
        "path_relink_freq": max(5, n // 2)
    }


# =====================================================
# DETERMINISTIC BAŞLANGIÇ ÇÖZÜMLERİ
# =====================================================

def construct_initial_solutions(flow, distance):

    n = flow.shape[0]

    facility_strength = (
        flow.sum(axis=0) +
        flow.sum(axis=1)
    )

    location_centrality = (
        distance.sum(axis=0) +
        distance.sum(axis=1)
    )

    high_flow = list(np.argsort(-facility_strength))
    low_flow = list(np.argsort(facility_strength))

    central_locations = list(np.argsort(location_centrality))
    peripheral_locations = list(np.argsort(-location_centrality))

    solutions = []

    # -------------------------------------------------

    perm1 = np.zeros(n, dtype=int)

    for f, loc in zip(high_flow, central_locations):
        perm1[f] = loc

    solutions.append(perm1)

    # -------------------------------------------------

    perm2 = np.zeros(n, dtype=int)

    for f, loc in zip(low_flow, peripheral_locations):
        perm2[f] = loc

    solutions.append(perm2)

    # -------------------------------------------------

    perm3 = np.zeros(n, dtype=int)

    for f, loc in zip(high_flow, peripheral_locations):
        perm3[f] = loc

    solutions.append(perm3)

    # -------------------------------------------------

    solutions.append(np.arange(n))

    # -------------------------------------------------

    solutions.append(np.arange(n - 1, -1, -1))

    # -------------------------------------------------

    unique = []
    seen = set()

    for p in solutions:

        key = tuple(p.tolist())

        if key not in seen:
            unique.append(p.copy())
            seen.add(key)

    return unique


# =====================================================
# SWAP DELTA
# =====================================================

def swap_delta(perm, flow, distance, r, s):

    if r == s:
        return 0

    n = len(perm)

    pr = perm[r]
    ps = perm[s]

    delta = 0

    for k in range(n):

        if k != r and k != s:

            pk = perm[k]

            delta += (
                flow[r, k] *
                (distance[ps, pk] - distance[pr, pk])
            )

            delta += (
                flow[s, k] *
                (distance[pr, pk] - distance[ps, pk])
            )

            delta += (
                flow[k, r] *
                (distance[pk, ps] - distance[pk, pr])
            )

            delta += (
                flow[k, s] *
                (distance[pk, pr] - distance[pk, ps])
            )

    return delta


# =====================================================
# p-LOCAL SEARCH
# =====================================================

def p_local_search(perm, flow, distance, p_local):

    n = len(perm)

    current = perm.copy()

    current_cost = objective(
        current,
        flow,
        distance
    )

    facility_strength = (
        flow.sum(axis=0) +
        flow.sum(axis=1)
    )

    candidate_facilities = list(
        np.argsort(-facility_strength)
    )[:p_local]

    improved = True

    while improved:

        improved = False

        best_delta = 0
        best_move = None

        for i in candidate_facilities:

            for j in range(n):

                if i == j:
                    continue

                delta = swap_delta(
                    current,
                    flow,
                    distance,
                    i,
                    j
                )

                if delta < best_delta:

                    best_delta = delta
                    best_move = (i, j)

        if best_move is not None:

            i, j = best_move

            current[i], current[j] = (
                current[j],
                current[i]
            )

            current_cost += best_delta

            improved = True

    return current, current_cost


# =====================================================
# TABU SEARCH
# =====================================================

def adaptive_tabu_search(
    start_perm,
    flow,
    distance,
    params
):

    n = len(start_perm)

    current = start_perm.copy()

    current_cost = objective(
        current,
        flow,
        distance
    )

    best = current.copy()
    best_cost = current_cost

    tabu = {}

    tenure = params["tabu_tenure"]

    max_iter = params["max_iter"]

    no_improve_limit = params["no_improve_limit"]

    no_improve = 0

    for iteration in range(max_iter):

        best_move = None

        best_move_delta = float("inf")

        for i in range(n - 1):

            for j in range(i + 1, n):

                delta = swap_delta(
                    current,
                    flow,
                    distance,
                    i,
                    j
                )

                candidate_cost = (
                    current_cost + delta
                )

                move = (
                    min(current[i], current[j]),
                    max(current[i], current[j])
                )

                is_tabu = (
                    tabu.get(move, -1) > iteration
                )

                aspiration = (
                    candidate_cost < best_cost
                )

                if (not is_tabu) or aspiration:

                    if delta < best_move_delta:

                        best_move_delta = delta

                        best_move = (
                            i,
                            j,
                            move,
                            candidate_cost
                        )

        if best_move is None:
            break

        i, j, move, candidate_cost = best_move

        current[i], current[j] = (
            current[j],
            current[i]
        )

        current_cost = candidate_cost

        tabu[move] = iteration + tenure

        if current_cost < best_cost:

            best = current.copy()
            best_cost = current_cost

            no_improve = 0

        else:
            no_improve += 1

        if no_improve >= no_improve_limit:
            break

    return best, best_cost


# =====================================================
# ANA HİBRİT METASEZGİSEL
# =====================================================

def deterministic_hybrid_qap(flow, distance):

    n = flow.shape[0]

    params = deterministic_parameters(n)

    initial_solutions = construct_initial_solutions(
        flow,
        distance
    )

    best_global_cost = float("inf")
    best_global_perm = None

    for solution in initial_solutions:

        local_solution, local_cost = p_local_search(
            solution,
            flow,
            distance,
            params["p_local"]
        )

        tabu_solution, tabu_cost = adaptive_tabu_search(
            local_solution,
            flow,
            distance,
            params
        )

        if tabu_cost < best_global_cost:

            best_global_cost = tabu_cost

            best_global_perm = tabu_solution.copy()

    return (
        best_global_perm,
        best_global_cost,
        params
    )


# =====================================================
# 10 DENEME
# =====================================================

def run_experiment():

    results = []

    for name, known_best in ESC_INSTANCES.items():

        print("\n" + "=" * 70)
        print(f"Çalışıyor: {name}")
        print("=" * 70)

        n, flow, distance = download_qap_instance(name)

        run_costs = []
        run_times = []

        best_perm_overall = None

        last_params = None

        for run in range(10):

            start = time.time()

            best_perm, best_cost, params = (
                deterministic_hybrid_qap(
                    flow,
                    distance
                )
            )

            end = time.time()

            cpu = end - start

            run_costs.append(best_cost)
            run_times.append(cpu)

            best_perm_overall = best_perm.copy()

            last_params = params

            print(
                f"Deneme {run + 1:02d} | "
                f"Sonuç: {best_cost} | "
                f"CPU: {cpu:.4f} sn"
            )

        best_found = min(run_costs)

        worst_found = max(run_costs)

        mean_cost = np.mean(run_costs)

        std_cost = (
            np.std(run_costs, ddof=1)
            if len(run_costs) > 1 else 0
        )

        mean_time = np.mean(run_times)

        gap = (
            (
                best_found - known_best
            ) / known_best
        ) * 100

        results.append({

            "Veri": name,
            "n": n,

            "Bilinen En İyi": known_best,

            "Bizim En İyi": best_found,

            "Bizim En Kötü": worst_found,

            "10 Deneme Ortalaması": mean_cost,

            "10 Deneme Std Sapması": std_cost,

            "Gap (%)": gap,

            "Ortalama CPU (sn)": mean_time,

            "p_local": last_params["p_local"],

            "tabu_tenure": last_params["tabu_tenure"],

            "elite_size": last_params["elite_size"],

            "max_iter": last_params["max_iter"],

            "En İyi Permütasyon":
                (best_perm_overall + 1).tolist()
        })

    df = pd.DataFrame(results)

    print("\n\n")
    print("=" * 70)
    print("GENEL SONUÇLAR")
    print("=" * 70)

    print(df.to_string(index=False))

    df.to_excel(
        "esc_hybrid_qap_results.xlsx",
        index=False
    )

    print("\nSonuçlar kaydedildi:")
    print("esc_hybrid_qap_results.xlsx")


# =====================================================
# MAIN
# =====================================================

if __name__ == "__main__":

    run_experiment()