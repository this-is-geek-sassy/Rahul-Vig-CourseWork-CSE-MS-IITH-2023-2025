import csv
import numpy as np
import matplotlib.pyplot as plt
import sympy as sp
from sklearn.manifold import TSNE
np.random.seed(0)


# reading the input (inlined CSV parsing from previous read_data)
filename = 'testcase_1.csv'
initial_point_given = True
with open(filename, newline='') as csvfile:
    reader = csv.reader(csvfile)
    data = list(reader)

# Extract z (initial feasible point) and c (cost vector)
z = np.array([float(val) for val in data[0][:-1]])  # Excluding the last element (empty string)
c = np.array([float(val) for val in data[1][:-1]])  # Excluding the last element (empty string)

# Extract b (constraint vector)
b = np.array([float(row[-1]) for row in data[2:]])  # Last column excluding first two rows

# Extract A (matrix A)
A = np.array([[float(val) for val in row[:-1]] for row in data[2:]])  # Excluding the last element (constraint column)

m, n = len(b), len(c)

# Display extracted data
print("Initial Feasible Point (z):", z)
print("Cost Vector (c):", c)
print("Constraint Vector (b):", b)
print("Matrix A:")
print(A)
print(f"Rows: {m} | Columns: {n}")


#handling degenracy
matrix_A = A
vector_b_original = b
vector_z = z
vector_c = c
dimension_n = n

epsilon = 0.1
attempt = 0
vector_b = vector_b_original
attempt = 0
while True:
    if attempt > 0:
        print("\n==================================================")
        print(f"Degeneracy detected. Attempting to handle it. Attempt - {attempt}")
        print("==================================================\n")
        epsilon = epsilon * 0.5
        vector_b = np.array([vector_b_original[i] + epsilon**(i+1) for i in range(len(vector_b_original))])

    # Track variables
    _track_cost = []
    _track_z = []
    _track_cost.append(np.dot(vector_c, vector_z))
    _track_z.append(vector_z)

    # Initialize variables for iteration
    _z_old = vector_z
    _z_new = _z_old
    _iteration = 0
    _print_interval = 1

    _product = np.dot(matrix_A, vector_z)
    _mask = np.abs(_product - vector_b) < 1e-8
    _tight_rows = matrix_A[_mask]
    _untight_rows = matrix_A[~_mask]

    if len(_tight_rows) == 0:
        _rank = 0
    else:
        _rank = np.linalg.matrix_rank(_tight_rows)

    if _rank == dimension_n:
        outputs1 = (_z_old, _track_cost, _track_z)
    else:
        print("Feasible point is not a vertex. Searching for a vertex...")
        print("==================================================")

        while _rank != dimension_n:
            _iteration += 1

            # Display iteration information
            if _iteration % _print_interval == 0:
                print(f"Iteration: {_iteration} - Rank: {_rank}")
                if _iteration > 300:
                    _print_interval = 1000
                elif _iteration > 10000:
                    _print_interval = 10000

            # Determine direction to move in
            if len(_tight_rows) == 0:
                _u = np.random.rand(_untight_rows.shape[-1])
            else:
                # Compute nullspace using SymPy and convert to numpy
                _ns_basis = sp.Matrix(_tight_rows).nullspace()
                if len(_ns_basis) == 0:
                    _u = np.random.rand(_untight_rows.shape[-1])
                else:
                    _ns_cols = [np.asarray(v, dtype=float).reshape(-1, 1) for v in _ns_basis]
                    _null_space_matrix = np.hstack(_ns_cols)
                    _u = _null_space_matrix[:, 0]

            # Calculate step magnitude
            while True:
                _alphas = [(_b_i - np.dot(a2_i, _z_old)) / np.dot(a2_i, _u) for _b_i, a2_i in zip(vector_b[~_mask], _untight_rows)]
                _all_alphas = [alpha for alpha in _alphas if alpha > 0]
                if len(_all_alphas) == 0:
                    _u = -1 * _u
                else:
                    break

            _alpha = min(_all_alphas)

            # Move to the new vertex
            _z_new = _z_old + _alpha * _u

            # Update tight and untight rows based on the new vertex
            _product = np.dot(matrix_A, _z_new)
            _mask = np.abs(_product - vector_b) < 1e-8
            _tight_rows = matrix_A[_mask]
            _untight_rows = matrix_A[~_mask]

            _z_old = _z_new

            # Recalculate rank based on updated tight rows
            if len(_tight_rows) == 0:
                _rank = 0
            else:
                _rank = np.linalg.matrix_rank(_tight_rows)

            # Store the newly found vertex
            _track_cost.append(np.dot(vector_c, _z_new))
            _track_z.append(_z_new)

        if not (_tight_rows.shape[0] > _tight_rows.shape[1]):
            outputs1 = (_z_new, _track_cost, _track_z)
        else:
            outputs1 = (None,)
    if len(outputs1) == 1:
        attempt+=1
        continue

    print("\n==================================================")
    print("Reached the initial vertex from feasible point!")
    print("==================================================\n")
    z_new, feas2vert_z_all_cost, feas2vert_z_all = outputs1

    print("Searching for optimal vertex...")
    print("==================================================")
    vert2vert_z_all_cost = []
    vert2vert_z_all = []
    _v_z_old = z_new
    _v_z_new = _v_z_old
    vert2vert_z_all_cost.append(np.dot(vector_c, _v_z_old))
    vert2vert_z_all.append(_v_z_old)
    _v_iteration = 0
    _v_print_interval = 1
    while True:
        _v_iteration += 1
        if _v_iteration % _v_print_interval == 0:
            print(f"Iteration: {_v_iteration}")
        _v_product = np.dot(matrix_A, _v_z_old)
        _v_mask = np.abs(_v_product - vector_b) < 1e-8
        _v_tight_rows = matrix_A[_v_mask]
        _v_untight_rows = matrix_A[~_v_mask]
        if _v_tight_rows.shape[0] > _v_tight_rows.shape[1]:
            outputs2 = (None,)
            break
        try:
            _v_A_inv = np.linalg.inv(_v_tight_rows)
            _v_directions_matrix = -1 * _v_A_inv
        except np.linalg.LinAlgError:
            print("Matrix is singular. Cannot compute the inverse.")
            outputs2 = (None,)
            break
        _v_directions = _v_directions_matrix.T
        _v_positive_dirs = [d for d in _v_directions if np.dot(d, vector_c) > 0]
        if not _v_positive_dirs:
            outputs2 = (_v_z_new, vert2vert_z_all_cost, vert2vert_z_all)
            break
        _v_u = _v_positive_dirs[0]
        _v_alphas = [(b_i - np.dot(a2_i, _v_z_old)) / np.dot(a2_i, _v_u) for b_i, a2_i in zip(vector_b[~_v_mask], _v_untight_rows)]
        _v_pos_alphas = [a for a in _v_alphas if a > 0]
        if len(_v_pos_alphas) == 0:
            print("The problem is unbounded. Can't find a optimal solution!")
            outputs2 = (None, vert2vert_z_all_cost, vert2vert_z_all)
            break
        _v_alpha = min(_v_pos_alphas)
        _v_z_new = _v_z_old + _v_alpha * _v_u
        _v_z_old = _v_z_new
        outputs2 = (_v_z_new, vert2vert_z_all_cost, vert2vert_z_all)
        break

    if len(outputs2) == 1:
        attempt += 1
        continue

    z_optimal, vert2vert_z_all_cost, vert2vert_z_all = outputs2
    if np.all(z_optimal == None):
        print("\n==================================================")
        print("The problem is unbounded!")
        print("==================================================\n")
    else:
        print("\n==================================================")
        print("Reached the optimal vertex!")
        print("==================================================\n")
    break


print("==========================")
print("Feasible point to vertex")
print("==========================")
print(f"Point: {feas2vert_z_all}")
print(f"Cost: {feas2vert_z_all_cost}")

print("\nInitial vertex")
print("==========================")

print(z_new)


print("\n==========================")
print("Vertex to Optimal vertex")
print("==========================")

print(f"Point: {vert2vert_z_all}")
print(f"Cost: {vert2vert_z_all_cost}")

if np.all(z_optimal == None):
    print("\nThe problem is unbounded!")
else:
    print("\nOptimal vertex")
    print("==========================")
    print(z_optimal)


# Plots:-
# Plotting the costs against iterations
try:
    iterations = range(1, len(feas2vert_z_all_cost) + 1)
    plt.figure()
    plt.plot(iterations, feas2vert_z_all_cost, marker='o')
    plt.xlabel('Iterations')
    plt.ylabel('Cost')
    plt.title('Cost Variation over Iterations')
    plt.grid(True)
    plt.tight_layout()
    plt.show()
except Exception as e:
    # Fallback: save figure to file when interactive display not available
    fname = 'feasible_cost_plot.png'
    try:
        plt.savefig(fname)
        print(f"Could not display plot interactively ({e}). Saved to {fname}")
    except Exception as e2:
        print(f"Failed to save plot: {e2}")

vectors = np.stack(feas2vert_z_all)

if len(vectors) > 1:
    try:
        # Perform t-SNE to reduce the vectors to a 2-dimensional space
        tsne = TSNE(n_components=2, random_state=42, perplexity=max(5, len(vectors) - 1))
        vectors_2d = tsne.fit_transform(vectors)

        # Plot the vectors in the 2D t-SNE space and connect them across iterations
        plt.figure(figsize=(8, 6))
        for i in range(len(vectors_2d) - 1):
            plt.plot(
                [vectors_2d[i][0], vectors_2d[i + 1][0]],
                [vectors_2d[i][1], vectors_2d[i + 1][1]],
                'b-',
                alpha=0.5
            )
            plt.scatter(vectors_2d[i][0], vectors_2d[i][1], color='red')
            plt.text(
                vectors_2d[i][0],
                vectors_2d[i][1],
                str(i + 1),
                horizontalalignment='right',
                verticalalignment='bottom',
                fontsize=12,
                color='black'
            )

        # Plot the last vector separately to avoid connecting it to the next iteration
        plt.scatter(vectors_2d[-1][0], vectors_2d[-1][1], color='red', label='Visited points')
        plt.text(
            vectors_2d[-1][0],
            vectors_2d[-1][1],
            "Initial Vertex",
            horizontalalignment='right',
            verticalalignment='bottom',
            fontsize=12,
            color='black'
        )

        plt.xlabel('t-SNE Component 1')
        plt.ylabel('t-SNE Component 2')
        plt.title('t-SNE Visualization of "z" Across Iterations')
        plt.legend()
        plt.grid(True)
        plt.tight_layout()
        plt.show()
    except Exception as e:
        print(f"t-SNE or plotting failed: {e}. Falling back to raw coordinates or saving figure.")
        try:
            if vectors.shape[1] >= 2:
                plt.figure()
                plt.scatter(vectors[:, 0], vectors[:, 1], color='red')
                plt.title('Raw 2D projection of vectors')
                plt.tight_layout()
                plt.show()
            else:
                fname = 'feasible_vectors.npy'
                np.save(fname, vectors)
                print(f"Saved vectors to {fname}")
        except Exception as e2:
            print(f"Failed fallback plotting/saving: {e2}")
else:
    print("Cant plot TSNE as there is only one vector!")



# Visualizing the vertex to optimal vertex path
if np.all(z_optimal == None):
    print("The problem is unbounded!")

# Plotting the costs against iterations
iterations = range(1, len(vert2vert_z_all_cost) + 1)
plt.plot(iterations, vert2vert_z_all_cost, marker='o')

# Adding labels and title
plt.xlabel('Iterations')
plt.ylabel('Cost')
plt.title('Cost Variation over Iterations')

# Show the plot
plt.grid(True)
plt.tight_layout()
plt.show()


if np.all(z_optimal == None):
    print("The problem is unbounded!")
    final_point = "Vertex"
else:
    final_point = "Optimal Vertex"

vectors = np.stack(vert2vert_z_all)

if len(vectors) > 1:
    # Perform t-SNE to reduce the vectors to a 2-dimensional space
    tsne = TSNE(n_components=2, random_state=42, perplexity=len(vectors) - 1)
    vectors_2d = tsne.fit_transform(vectors)

    # Plot the vectors in the 2D t-SNE space and connect them across iterations
    plt.figure(figsize=(8, 6))
    for i in range(len(vectors_2d) - 1):
        plt.plot(
            [vectors_2d[i][0], vectors_2d[i + 1][0]],
            [vectors_2d[i][1], vectors_2d[i + 1][1]],
            'b-',
            alpha=0.5
        )
        plt.scatter(vectors_2d[i][0], vectors_2d[i][1], color='red')
        plt.text(
            vectors_2d[i][0],
            vectors_2d[i][1],
            str(i + 1),
            horizontalalignment='right',
            verticalalignment='bottom',
            fontsize=12,
            color='black'
        )

    # Plot the last vector separately to avoid connecting it to the next iteration
    plt.scatter(vectors_2d[-1][0], vectors_2d[-1][1], color='red', label='Visited Vertex')
    plt.text(
        vectors_2d[-1][0],
        vectors_2d[-1][1],
        final_point,
        horizontalalignment='right',
        verticalalignment='bottom',
        fontsize=12,
        color='black'
    )

    # Set labels and title
    plt.xlabel('t-SNE Component 1')
    plt.ylabel('t-SNE Component 2')
    plt.title('t-SNE Visualization of Vertices Across Iterations')
    plt.legend()
    plt.grid(True)
    plt.tight_layout()

    # Show the plot
    plt.show()
else:
    print("Cant plot TSNE as there is only one vector!")

