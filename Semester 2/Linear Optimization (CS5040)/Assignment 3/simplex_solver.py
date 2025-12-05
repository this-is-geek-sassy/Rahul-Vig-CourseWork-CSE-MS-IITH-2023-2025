import csv
import numpy as np
from scipy.linalg import null_space

np.random.seed(0)


def find_tight_rows(A, z, b, epsilon=1e-8):
    """Identifies tight and non-tight constraint rows."""
    product = np.dot(A, z)
    tight_mask = np.abs(product - b) < epsilon
    return tight_mask, A[tight_mask], A[~tight_mask]


def make_non_degenerate(b_original, epsilon, reduction_factor=0.5):
    """Perturbs b vector to handle degeneracy."""
    epsilon = epsilon * reduction_factor
    new_b = np.array([b_original[i] + epsilon**(i+1) for i in range(len(b_original))])
    return new_b, epsilon


def feasible_to_vertex(A, b, z, c, n):
    """Moves from feasible point to a vertex."""
    track_cost = [np.dot(c, z)]
    track_z = [z]
    
    z_old = z
    z_new = z
    iteration = 0
    print_interval = 1

    mask, tight_rows, untight_rows = find_tight_rows(A, z, b)
    rank = 0 if len(tight_rows) == 0 else np.linalg.matrix_rank(tight_rows)

    if rank == n:
        return z_old, track_cost, track_z

    print("Feasible point is not a vertex. Searching for a vertex...")
    print("=" * 50)

    while rank != n:
        iteration += 1

        if iteration % print_interval == 0:
            print(f"Iteration: {iteration} - Rank: {rank}")
            if iteration > 300:
                print_interval = 1000
            elif iteration > 10000:
                print_interval = 10000

        if len(tight_rows) == 0:
            u = np.random.rand(untight_rows.shape[-1])
        else:
            null_space_matrix = null_space(tight_rows)
            u = null_space_matrix[:, 0]

        while True:
            alphas = [(b[i] - np.dot(untight_rows[j], z_old)) / np.dot(untight_rows[j], u) 
                      for j, i in enumerate(np.where(~mask)[0])]
            all_alphas = [alpha for alpha in alphas if alpha > 0]
            if len(all_alphas) == 0:
                u = -1 * u
            else:
                break

        alpha = min(all_alphas)
        z_new = z_old + alpha * u

        mask, tight_rows, untight_rows = find_tight_rows(A, z_new, b)
        z_old = z_new

        rank = 0 if len(tight_rows) == 0 else np.linalg.matrix_rank(tight_rows)

        track_cost.append(np.dot(c, z_new))
        track_z.append(z_new)

    if tight_rows.shape[0] > tight_rows.shape[1]:
        return (None,)
    
    return z_new, track_cost, track_z


def vertex_to_vertex(A, b, z, c):
    """Moves from vertex to optimal vertex using simplex algorithm."""
    track_cost = [np.dot(c, z)]
    track_vertex = [z]
    
    z_old = z
    z_new = z_old
    iteration = 0

    while True:
        iteration += 1
        print(f"Iteration: {iteration}")

        mask, tight_rows, untight_rows = find_tight_rows(A, z_old, b)

        if tight_rows.shape[0] > tight_rows.shape[1]:
            return (None,)

        try:
            A_inv = np.linalg.inv(tight_rows)
            directions = (-1 * A_inv).T
        except np.linalg.LinAlgError:
            print("Matrix is singular. Cannot compute the inverse.")
            return None, track_cost, track_vertex

        positive_directions = [d for d in directions if np.dot(d, c) > 0]

        if not positive_directions:
            return z_new, track_cost, track_vertex

        u = positive_directions[0]

        alphas = [(b[i] - np.dot(untight_rows[j], z_old)) / np.dot(untight_rows[j], u)
                  for j, i in enumerate(np.where(~mask)[0])]
        positive_alphas = [alpha for alpha in alphas if alpha > 0]
        
        if len(positive_alphas) == 0:
            print("The problem is unbounded. Can't find an optimal solution!")
            return None, track_cost, track_vertex

        alpha = min(positive_alphas)
        z_new = z_old + alpha * u
        z_old = z_new

        track_cost.append(np.dot(c, z_new))
        track_vertex.append(z_new)


def solve_simplex(filename):
    """Main solver function."""
    with open(filename, newline='') as csvfile:
        reader = csv.reader(csvfile)
        data = list(reader)

    z = np.array([float(val) for val in data[0][:-1]])
    c = np.array([float(val) for val in data[1][:-1]])
    b = np.array([float(row[-1]) for row in data[2:]])
    A = np.array([[float(val) for val in row[:-1]] for row in data[2:]])

    m, n = len(b), len(c)

    print("Initial Feasible Point (z):", z)
    print("Cost Vector (c):", c)
    print("Constraint Vector (b):", b)
    print("Matrix A:")
    print(A)
    print(f"Rows: {m} | Columns: {n}")

    vector_b_original = b
    epsilon = 0.1
    attempt = 0
    vector_b = vector_b_original

    while True:
        if attempt > 0:
            print("\n" + "=" * 50)
            print(f"Degeneracy detected. Attempting to handle it. Attempt - {attempt}")
            print("=" * 50 + "\n")
            vector_b, epsilon = make_non_degenerate(vector_b_original, epsilon)

        outputs1 = feasible_to_vertex(A, vector_b, z, c, n)
        if len(outputs1) == 1:
            attempt += 1
            continue

        print("\n" + "=" * 50)
        print("Reached the initial vertex from feasible point!")
        print("=" * 50 + "\n")
        z_new, feas2vert_cost, feas2vert_z = outputs1

        print("Searching for optimal vertex...")
        print("=" * 50)
        outputs2 = vertex_to_vertex(A, vector_b, z_new, c)
        if len(outputs2) == 1:
            attempt += 1
            continue

        z_optimal, vert2vert_cost, vert2vert_z = outputs2
        if z_optimal is None or np.all(z_optimal == None):
            print("\n" + "=" * 50)
            print("The problem is unbounded!")
            print("=" * 50 + "\n")
        else:
            print("\n" + "=" * 50)
            print("Reached the optimal vertex!")
            print("=" * 50 + "\n")
        break

    print("\n" + "=" * 26)
    print("Feasible point to vertex")
    print("=" * 26)
    print(f"Points visited: {len(feas2vert_z)}")
    print(f"Final point: {feas2vert_z[-1]}")
    print(f"Final cost: {feas2vert_cost[-1]}")

    print("\n" + "=" * 26)
    print("Vertex to Optimal vertex")
    print("=" * 26)
    print(f"Vertices visited: {len(vert2vert_z)}")
    if z_optimal is None:
        print("The problem is unbounded!")
    else:
        print(f"Optimal vertex: {z_optimal}")
        print(f"Optimal cost: {vert2vert_cost[-1]}")

    return z_optimal, feas2vert_z, feas2vert_cost, vert2vert_z, vert2vert_cost


if __name__ == "__main__":
    result = solve_simplex('..\\Test Case alone\\testcase_10.csv')
