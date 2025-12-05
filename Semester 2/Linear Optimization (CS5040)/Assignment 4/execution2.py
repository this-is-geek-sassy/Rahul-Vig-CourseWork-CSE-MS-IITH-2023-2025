import csv
from scipy.linalg import null_space
import numpy as np
import matplotlib.pyplot as plt
from sklearn.manifold import TSNE
np.random.seed(0)

# -------------------------
# Inline "helper" printing
# -------------------------
def _print_header(text, width=70):
    print("\n" + "=" * width)
    print(f"  {text}")
    print("=" * width)

def _print_subheader(text, width=70):
    print("\n" + "-" * width)
    print(f"  {text}")
    print("-" * width)

def _print_info(label, value, indent=2):
    print(" " * indent + f"{label}: {value}")


_print_header("LINEAR PROGRAMMING SOLVER")

# ---- Read data from CSV file (inlined) ----
filename = 'testcase_6.csv'
initial_point_given = False

with open(filename, newline='') as csvfile:
    reader = csv.reader(csvfile)
    data = list(reader)

if initial_point_given:
    z = np.array([float(val) for val in data[0][:-1]])
    c = np.array([float(val) for val in data[1][:-1]])
    b = np.array([float(row[-1]) for row in data[2:]])
    A = np.array([[float(val) for val in row[:-1]] for row in data[2:]])
else:
    c = np.array([float(val) for val in data[0][:-1]])
    b = np.array([float(row[-1]) for row in data[1:]])
    A = np.array([[float(val) for val in row[:-1]] for row in data[1:]])

m, n = len(b), len(c)

_print_subheader("Problem Specification")
_print_info("Cost vector (c)", c)
_print_info("Constraint vector (b)", b)
_print_info("Problem dimensions", f"{m} constraints × {n} variables")
print("\n  Matrix A:")
print("  " + str(A).replace("\n", "\n  "))

# ------------------------------
# find_initial_feasible_point
# ------------------------------
# Check if origin is feasible
if np.all(b >= 0):
    origin_feas = True
    z0 = np.zeros(c.shape)
    print("  ✓ Using origin (0) as initial feasible point")
    # We will use z0 as initial feasible point for main phase directly
    z_initial_for_original = z0
    modified_z_optimal = -1  # marker meaning no modified LP needed
    # Setup placeholders for downstream variables used later
    # (when origin feasible we will skip modified LP)
    z_optimal = None
    z_new_from_feas2vert = None
    feas2vert_z_all_cost = []
    feas2vert_z_all = []
    vert2vert_z_all_cost = []
    vert2vert_z_all = []
else:
    origin_feas = False
    # Build modified LP (inline of get_modified_LP)
    _A = A.copy()
    _b = b.copy()
    _c = c.copy()
    # initial_point uses min(_b)
    min_b = min(_b)
    initial_point = np.zeros((_A.shape[1] + 1, 1))
    initial_point[-1] = min_b
    modified_A = _A
    modified_b = _b
    modified_c = _c

    if initial_point.shape != _c.shape:
        rows, cols = _A.shape
        modified_A = np.append(np.append(_A, np.zeros((1, cols)), axis=0), np.ones((rows + 1, 1)), axis=1)
        modified_A[-1][-1] = -1
        modified_b = np.append(_b, [abs(min(_b))], axis=0)
        modified_c = np.zeros((cols + 1, 1))
        modified_c[-1] = 1

    modified_c = np.hstack(modified_c)
    initial_point = np.hstack(initial_point)

    _m, _n = len(modified_b), len(modified_c)

    _print_subheader("Modified LP Problem for Finding Initial Feasible Point")
    _print_info("Initial feasible point (z)", initial_point)
    _print_info("Cost vector (c)", modified_c)
    _print_info("Constraint vector (b)", modified_b)
    _print_info("Problem dimensions", f"{_m} constraints × {_n} variables")
    print("\n  Matrix A:")
    print("  " + str(modified_A).replace("\n", "\n  "))

    # Solve modified LP: Phase 1 (feasible->vertex) then Phase 2 (vertex->optimal)
    matrix_A = modified_A
    vector_b_original = modified_b.copy()
    vector_z = initial_point.copy()
    vector_c = modified_c.copy()
    dimension_n = _n

    epsilon = 0.1
    attempt = 0
    vector_b = vector_b_original.copy()

    z_optimal = None
    modified_z_optimal = None
    z_new_from_feas2vert = None
    feas2vert_z_all_cost = None
    feas2vert_z_all = None
    vert2vert_z_all_cost = None
    vert2vert_z_all = None

    # Outer loop to handle degeneracy attempts
    while True:
        if attempt > 0:
            _print_subheader(f"Handling Degeneracy - Attempt #{attempt}")

        # ---------- Phase 1: feasible_to_vertex_assign4 (inlined) ----------
        track_cost = []
        track_z = []
        track_cost.append(np.dot(vector_c, vector_z))
        track_z.append(vector_z.copy())

        z_old = vector_z.copy()
        iteration = 0
        print_interval = 1

        # Find tight rows initially
        epi_find = 1e-8
        product = np.dot(matrix_A, z_old)
        mask = np.abs(product - vector_b) < epi_find
        tight_rows = matrix_A[mask]
        untight_rows = matrix_A[~mask]

        if len(tight_rows) == 0:
            rank = 0
        else:
            rank = np.linalg.matrix_rank(tight_rows)

        if rank == dimension_n:
            # Already at vertex
            z_new = z_old.copy()
            z_new_from_feas2vert = z_new.copy()
            feas2vert_z_all_cost = track_cost.copy()
            feas2vert_z_all = track_z.copy()
        else:
            _print_subheader("Phase 1: Moving from Feasible Point to Vertex")
            print(f"  Initial rank: {rank} (target: {dimension_n})")

            # iterate until rank == dimension_n
            while rank != dimension_n:
                iteration += 1
                if iteration % print_interval == 0:
                    print(f"  Iteration {iteration:5d} | Rank: {rank}/{dimension_n}")
                    if iteration > 300:
                        print_interval = 1000
                    elif iteration > 10000:
                        print_interval = 10000

                if len(tight_rows) == 0:
                    # untight_rows.shape[-1] is number of columns
                    u = np.random.rand(untight_rows.shape[-1])
                else:
                    # compute nullspace of tight_rows
                    null_space_matrix = null_space(tight_rows)
                    # if nullspace empty (shouldn't usually happen), fallback to random
                    if null_space_matrix.size == 0:
                        u = np.random.rand(matrix_A.shape[1])
                    else:
                        u = null_space_matrix[:, 0]

                # find positive alphas for moving along u without violating constraints
                while True:
                    # recalc mask and untight rows for current z_old
                    # (mask already correct for z_old at this point)
                    alphas = [(_b_i - np.dot(a2_i, z_old)) / np.dot(a2_i, u) for _b_i, a2_i in zip(vector_b[~mask], untight_rows)]
                    alphas = [alpha for alpha in alphas if alpha > 0]

                    positive_alphas = []
                    epi_small = 1e-10
                    for i in alphas:
                        if not (i == np.inf or abs(i) < epi_small):
                            positive_alphas.append(i)

                    if len(positive_alphas) == 0:
                        u = -1 * u
                    else:
                        break

                alpha = min(positive_alphas)
                z_new = z_old + alpha * u

                # update tight/untight for z_new
                product = np.dot(matrix_A, z_new)
                mask = np.abs(product - vector_b) < epi_small
                tight_rows = matrix_A[mask]
                untight_rows = matrix_A[~mask]

                z_old = z_new.copy()

                if len(tight_rows) == 0:
                    rank = 0
                else:
                    rank = np.linalg.matrix_rank(tight_rows)

                track_cost.append(np.dot(vector_c, z_new))
                track_z.append(z_new.copy())

            z_new_from_feas2vert = z_new.copy()
            feas2vert_z_all_cost = track_cost.copy()
            feas2vert_z_all = track_z.copy()

        print("\n  ✓ Successfully reached vertex from feasible point")

        # ---------- Phase 2: vertex_to_vertex_assign4 (inlined) ----------
        _print_subheader("Phase 2: Searching for Optimal Vertex")

        # Prepare for vertex-to-vertex
        track_cost_v = []
        track_vertex_v = []
        z_old_v = z_new_from_feas2vert.copy()
        z_new_v = z_old_v.copy()
        track_cost_v.append(np.dot(vector_c, z_old_v))
        track_vertex_v.append(z_old_v.copy())
        iteration_v = 0
        print_interval_v = 1

        unbounded_flag = False
        degenerate_flag = False
        early_exit_flag = False

        while True:
            iteration_v += 1
            if iteration_v % print_interval_v == 0:
                print(f"  Iteration {iteration_v:5d} | Current cost: {track_cost_v[-1]:.6f}")

            # Find tight rows for current z_old_v
            epi_v = 1e-8
            product_v = np.dot(matrix_A, z_old_v)
            mask_v = np.abs(product_v - vector_b) < epi_v
            tight_rows_v = matrix_A[mask_v]
            untight_rows_v = matrix_A[~mask_v]

            # If degenerate (more tight rows than columns) -> return None
            if tight_rows_v.shape[0] > tight_rows_v.shape[1]:
                degenerate_flag = True
                break

            # compute directions = -inv(tight_rows_v).T
            try:
                A_inv = np.linalg.inv(tight_rows_v)
                neg_A_inv = -1 * A_inv
                directions = neg_A_inv.T
            except np.linalg.LinAlgError:
                print("  [ERROR] Matrix is singular. Cannot compute the inverse.")
                directions = None

            if directions is None:
                degenerate_flag = True
                break

            positive_directions = []
            for direction in directions:
                if np.dot(direction, vector_c) > 0:
                    positive_directions.append(direction)

            # If no improving directions or too many iterations, exit with current best
            if (not positive_directions) or iteration_v > 10:
                early_exit_flag = True
                break

            # choose first positive direction
            u_v = positive_directions[0]

            # compute alphas along u_v
            alphas_v = [(b_i - np.dot(a2_i, z_old_v)) / np.dot(a2_i, u_v) for b_i, a2_i in zip(vector_b[~mask_v], untight_rows_v)]
            alphas_v = [alpha for alpha in alphas_v if alpha > 0]

            positive_alphas_v = []
            epi_small_v = 1e-10
            for i in alphas_v:
                if not (i == np.inf or abs(i) < epi_small_v):
                    positive_alphas_v.append(i)

            if len(positive_alphas_v) == 0:
                print("  [WARNING] Problem is unbounded - no optimal solution exists!")
                unbounded_flag = True
                break

            alpha_v = min(positive_alphas_v)
            z_new_v = z_old_v + alpha_v * u_v
            z_old_v = z_new_v.copy()

            track_cost_v.append(np.dot(vector_c, z_new_v))
            track_vertex_v.append(z_new_v.copy())

        # Finished Phase 2 loop
        if degenerate_flag:
            # degenerate -> try again with perturbation (outer loop)
            attempt += 1
            # Perturb vector_b slightly for next attempt
            continue

        # If unbounded_flag: mark z_optimal accordingly
        if unbounded_flag:
            z_optimal = None
            modified_z_optimal = None
            vert2vert_z_all_cost = track_cost_v.copy()
            vert2vert_z_all = track_vertex_v.copy()
            print("\n  ⚠ Modified LP is unbounded!")
        else:
            z_optimal = z_new_v.copy()
            modified_z_optimal = -1
            vert2vert_z_all_cost = track_cost_v.copy()
            vert2vert_z_all = track_vertex_v.copy()
            print("\n  ✓ Optimal vertex found!")

        # break outer degeneracy handling loop
        break

    # After solving modified LP, if modified LP unbounded then modified_z_optimal == None
    if modified_z_optimal is None:
        print_header("RESULT: Problem is Unbounded")
    else:
        z_initial_for_original = z_optimal[:_n] if z_optimal is not None else None
        if z_initial_for_original is not None:
            _print_subheader("Initial Feasible Point for Original Problem")
            print("  " + str(z_initial_for_original))

# ------------------------------
# Now process the ORIGINAL problem
# ------------------------------
# If origin feasible, z_initial_for_original already set, else use computed one
if 'z_initial_for_original' not in locals() or z_initial_for_original is None:
    # If modified LP was unbounded, we'll still try to proceed but will likely find unbounded original if that's the case.
    z_initial_for_original = np.zeros(c.shape) if origin_feas else (z_optimal[:len(c)] if (z_optimal is not None) else np.zeros(c.shape))

# If modified LP indicated unbounded (modified_z_optimal is None), set marker
if 'modified_z_optimal' in locals() and modified_z_optimal is None:
    print_header("FINAL RESULT: Problem is Unbounded")
    # We'll still try to display what we have below; skip solving original
else:
    # Setup for solving original LP
    matrix_A_orig = A.copy()
    vector_b_original = b.copy()
    vector_z = z_initial_for_original.copy()
    vector_c = c.copy()
    dimension_n = n

    epsilon = 0.1
    attempt = 0
    vector_b = vector_b_original.copy()

    # Outer loop to handle degeneracy attempts on original
    while True:
        if attempt > 0:
            _print_subheader(f"Handling Degeneracy - Attempt #{attempt}")
            reduction_factor = 0.5
            epsilon = epsilon * reduction_factor
            vector_b = np.array([vector_b_original[i] + epsilon**(i+1) for i in range(len(vector_b_original))])

        # ---------- Phase 1 for original problem: feasible_to_vertex_assign4 (inlined) ----------
        track_cost = []
        track_z = []
        track_cost.append(np.dot(vector_c, vector_z))
        track_z.append(vector_z.copy())

        z_old = vector_z.copy()
        iteration = 0
        print_interval = 1

        # Find tight rows initially
        epi_find = 1e-8
        product = np.dot(matrix_A_orig, z_old)
        mask = np.abs(product - vector_b) < epi_find
        tight_rows = matrix_A_orig[mask]
        untight_rows = matrix_A_orig[~mask]

        if len(tight_rows) == 0:
            rank = 0
        else:
            rank = np.linalg.matrix_rank(tight_rows)

        if rank == dimension_n:
            # Already at vertex
            z_new = z_old.copy()
            z_new_final = z_new.copy()
            feas2vert_z_all_cost = track_cost.copy()
            feas2vert_z_all = track_z.copy()
        else:
            _print_subheader("Phase 1: Moving from Feasible Point to Vertex")
            print(f"  Initial rank: {rank} (target: {dimension_n})")

            # iterate until rank == dimension_n
            while rank != dimension_n:
                iteration += 1
                if iteration % print_interval == 0:
                    print(f"  Iteration {iteration:5d} | Rank: {rank}/{dimension_n}")
                    if iteration > 300:
                        print_interval = 1000
                    elif iteration > 10000:
                        print_interval = 10000

                if len(tight_rows) == 0:
                    u = np.random.rand(untight_rows.shape[-1])
                else:
                    null_space_matrix = null_space(tight_rows)
                    if null_space_matrix.size == 0:
                        u = np.random.rand(matrix_A_orig.shape[1])
                    else:
                        u = null_space_matrix[:, 0]

                while True:
                    alphas = [(_b_i - np.dot(a2_i, z_old)) / np.dot(a2_i, u) for _b_i, a2_i in zip(vector_b[~mask], untight_rows)]
                    alphas = [alpha for alpha in alphas if alpha > 0]

                    positive_alphas = []
                    epi_small = 1e-10
                    for i in alphas:
                        if not (i == np.inf or abs(i) < epi_small):
                            positive_alphas.append(i)

                    if len(positive_alphas) == 0:
                        u = -1 * u
                    else:
                        break

                alpha = min(positive_alphas)
                z_new = z_old + alpha * u

                # update tight/untight for z_new
                product = np.dot(matrix_A_orig, z_new)
                mask = np.abs(product - vector_b) < epi_small
                tight_rows = matrix_A_orig[mask]
                untight_rows = matrix_A_orig[~mask]

                z_old = z_new.copy()

                if len(tight_rows) == 0:
                    rank = 0
                else:
                    rank = np.linalg.matrix_rank(tight_rows)

                track_cost.append(np.dot(vector_c, z_new))
                track_z.append(z_new.copy())

            z_new_final = z_new.copy()
            feas2vert_z_all_cost = track_cost.copy()
            feas2vert_z_all = track_z.copy()

        print("\n  ✓ Successfully reached vertex from feasible point")

        # ---------- Phase 2 for original problem: vertex_to_vertex_assign4 (inlined) ----------
        _print_subheader("Phase 2: Searching for Optimal Vertex")

        track_cost_v = []
        track_vertex_v = []
        z_old_v = z_new_final.copy()
        z_new_v = z_old_v.copy()
        track_cost_v.append(np.dot(vector_c, z_old_v))
        track_vertex_v.append(z_old_v.copy())
        iteration_v = 0
        print_interval_v = 1

        unbounded_flag = False
        degenerate_flag = False
        early_exit_flag = False

        while True:
            iteration_v += 1
            if iteration_v % print_interval_v == 0:
                print(f"  Iteration {iteration_v:5d} | Current cost: {track_cost_v[-1]:.6f}")

            # Find tight rows for current z_old_v
            epi_v = 1e-8
            product_v = np.dot(matrix_A_orig, z_old_v)
            mask_v = np.abs(product_v - vector_b) < epi_v
            tight_rows_v = matrix_A_orig[mask_v]
            untight_rows_v = matrix_A_orig[~mask_v]

            # If degenerate (more tight rows than columns) -> signal attempt
            if tight_rows_v.shape[0] > tight_rows_v.shape[1]:
                degenerate_flag = True
                break

            # compute directions = -inv(tight_rows_v).T
            try:
                A_inv = np.linalg.inv(tight_rows_v)
                neg_A_inv = -1 * A_inv
                directions = neg_A_inv.T
            except np.linalg.LinAlgError:
                print("  [ERROR] Matrix is singular. Cannot compute the inverse.")
                directions = None

            if directions is None:
                degenerate_flag = True
                break

            positive_directions = []
            for direction in directions:
                if np.dot(direction, vector_c) > 0:
                    positive_directions.append(direction)

            # If no improving directions or too many iterations, exit with current best
            if (not positive_directions) or iteration_v > 1000:
                early_exit_flag = True
                break

            u_v = positive_directions[0]

            alphas_v = [(b_i - np.dot(a2_i, z_old_v)) / np.dot(a2_i, u_v) for b_i, a2_i in zip(vector_b[~mask_v], untight_rows_v)]
            alphas_v = [alpha for alpha in alphas_v if alpha > 0]

            positive_alphas_v = []
            epi_small_v = 1e-10
            for i in alphas_v:
                if not (i == np.inf or abs(i) < epi_small_v):
                    positive_alphas_v.append(i)

            if len(positive_alphas_v) == 0:
                print("  [WARNING] Problem is unbounded - no optimal solution exists!")
                unbounded_flag = True
                break

            alpha_v = min(positive_alphas_v)
            z_new_v = z_old_v + alpha_v * u_v
            z_old_v = z_new_v.copy()

            track_cost_v.append(np.dot(vector_c, z_new_v))
            track_vertex_v.append(z_new_v.copy())

        # Evaluate phase 2 results for original
        if degenerate_flag:
            attempt += 1
            # Continue outer loop to handle degeneracy by perturbing b
            continue

        if unbounded_flag:
            z_optimal_original = None
            vert2vert_z_all_cost = track_cost_v.copy()
            vert2vert_z_all = track_vertex_v.copy()
            print("\n  ⚠ Problem is unbounded!")
        else:
            z_optimal_original = z_new_v.copy()
            vert2vert_z_all_cost = track_cost_v.copy()
            vert2vert_z_all = track_vertex_v.copy()
            print("\n  ✓ Optimal vertex reached!")

        # Done solving original (break outer loop)
        break

# ------------------------------
# Display / Summarize results
# ------------------------------
_print_header("SOLUTION SUMMARY")

_print_subheader("Phase 1: Feasible Point → Vertex")
# Some of these variables may be undefined if earlier steps were skipped; guard accordingly.
try:
    _print_info("Visited points", len(feas2vert_z_all))
    _print_info("Points trajectory", feas2vert_z_all)
    _print_info("Cost trajectory", feas2vert_z_all_cost)
except Exception:
    print("  [INFO] Phase 1 data not available")

_print_subheader("Initial Vertex Found")
try:
    print("  " + str(z_new_final))
except Exception:
    print("  [INFO] Initial vertex not available")

_print_subheader("Phase 2: Vertex → Optimal Vertex")
try:
    _print_info("Visited vertices", len(vert2vert_z_all))
    _print_info("Vertex trajectory", vert2vert_z_all)
    _print_info("Cost trajectory", vert2vert_z_all_cost)
except Exception:
    print("  [INFO] Phase 2 data not available")

try:
    if z_optimal_original is None:
        _print_subheader("Final Result: UNBOUNDED")
    else:
        _print_subheader("Optimal Vertex")
        print("  " + str(z_optimal_original))
except NameError:
    try:
        if z_optimal is None:
            _print_subheader("Final Result: UNBOUNDED")
        else:
            _print_subheader("Optimal Vertex (from modified LP)")
            print("  " + str(z_optimal))
    except Exception:
        print("  [INFO] No optimal vertex computed")

# ------------------------------
# Visualizations (attempt; guard missing data)
# ------------------------------
_print_header("GENERATING VISUALIZATIONS")

# Phase 1 cost plot
try:
    iterations = range(1, len(feas2vert_z_all_cost) + 1)
    plt.figure(figsize=(10, 6))
    plt.plot(iterations, feas2vert_z_all_cost, marker='o', linewidth=2, markersize=6)
    plt.xlabel('Iterations', fontsize=12)
    plt.ylabel('Cost', fontsize=12)
    plt.title('Phase 1: Cost Variation (Feasible Point → Vertex)', fontsize=14, fontweight='bold')
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.show()
except Exception:
    print("  [INFO] Cannot plot Phase 1 cost - data missing")

# Phase 1 t-SNE trajectory
try:
    vectors = np.stack(feas2vert_z_all)
    if len(vectors) > 1:
        tsne = TSNE(n_components=2, random_state=42, perplexity=len(vectors) - 1)
        vectors_2d = tsne.fit_transform(vectors)

        plt.figure(figsize=(10, 8))
        for i in range(len(vectors_2d) - 1):
            plt.plot([vectors_2d[i][0], vectors_2d[i + 1][0]], 
                    [vectors_2d[i][1], vectors_2d[i + 1][1]], 
                    'b-', alpha=0.5, linewidth=2)
            plt.scatter(vectors_2d[i][0], vectors_2d[i][1], color='red', s=100, zorder=5)
            plt.text(vectors_2d[i][0], vectors_2d[i][1], str(i + 1),
                    ha='right', va='bottom', fontsize=10, fontweight='bold')

        plt.scatter(vectors_2d[-1][0], vectors_2d[-1][1], color='green', s=150, 
                   marker='*', label='Initial Vertex', zorder=5)
        plt.text(vectors_2d[-1][0], vectors_2d[-1][1], "Initial Vertex",
                ha='right', va='bottom', fontsize=10, fontweight='bold')

        plt.xlabel('t-SNE Component 1', fontsize=12)
        plt.ylabel('t-SNE Component 2', fontsize=12)
        plt.title('Phase 1: Point Trajectory in t-SNE Space', fontsize=14, fontweight='bold')
        plt.legend(fontsize=10)
        plt.grid(True, alpha=0.3)
        plt.tight_layout()
        plt.show()
    else:
        print("  [INFO] Cannot generate t-SNE plot - only one vector available")
except Exception:
    print("  [INFO] Cannot generate Phase 1 t-SNE - data missing")

# Phase 2 cost plot
try:
    if 'z_optimal_original' in locals() and z_optimal_original is None:
        print("  [INFO] Problem is unbounded - showing available data")

    iterations = range(1, len(vert2vert_z_all_cost) + 1)
    plt.figure(figsize=(10, 6))
    plt.plot(iterations, vert2vert_z_all_cost, marker='o', linewidth=2, markersize=6)
    plt.xlabel('Iterations', fontsize=12)
    plt.ylabel('Cost', fontsize=12)
    plt.title('Phase 2: Cost Variation (Vertex → Optimal)', fontsize=14, fontweight='bold')
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.show()
except Exception:
    print("  [INFO] Cannot plot Phase 2 cost - data missing")

# Phase 2 t-SNE trajectory
try:
    final_point_label = "Final Vertex" if ('z_optimal_original' in locals() and z_optimal_original is None) else "Optimal Vertex"
    vectors = np.stack(vert2vert_z_all)
    if len(vectors) > 1:
        tsne = TSNE(n_components=2, random_state=42, perplexity=len(vectors) - 1)
        vectors_2d = tsne.fit_transform(vectors)

        plt.figure(figsize=(10, 8))
        for i in range(len(vectors_2d) - 1):
            plt.plot([vectors_2d[i][0], vectors_2d[i + 1][0]],
                    [vectors_2d[i][1], vectors_2d[i + 1][1]],
                    'g-', alpha=0.5, linewidth=2)
            plt.scatter(vectors_2d[i][0], vectors_2d[i][1], color='blue', s=100, zorder=5)
            plt.text(vectors_2d[i][0], vectors_2d[i][1], str(i + 1),
                    ha='right', va='bottom', fontsize=10, fontweight='bold')

        plt.scatter(vectors_2d[-1][0], vectors_2d[-1][1], color='gold', s=150,
                   marker='*', label=final_point_label, zorder=5)
        plt.text(vectors_2d[-1][0], vectors_2d[-1][1], final_point_label,
                ha='right', va='bottom', fontsize=10, fontweight='bold')

        plt.xlabel('t-SNE Component 1', fontsize=12)
        plt.ylabel('t-SNE Component 2', fontsize=12)
        plt.title('Phase 2: Vertex Trajectory in t-SNE Space', fontsize=14, fontweight='bold')
        plt.legend(fontsize=10)
        plt.grid(True, alpha=0.3)
        plt.tight_layout()
        plt.show()
    else:
        print("  [INFO] Cannot generate t-SNE plot - only one vector available")
except Exception:
    print("  [INFO] Cannot generate Phase 2 t-SNE - data missing")

_print_header("EXECUTION COMPLETE")
