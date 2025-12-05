"""
Simplex Algorithm Implementation

Implements the simplex algorithm to maximize the objective function.
Assumptions: Polytope is non-degenerate, bounded, and rank of A is n.
"""

# importing all neccesary librarys for algorithm
import csv
import numpy as np
from scipy.linalg import null_space
from rich.console import Console
from rich.table import Table
from rich.panel import Panel
from rich import box

try:
    from IPython.core.getipython import get_ipython  # type: ignore
    if get_ipython() is not None and 'IPKernelApp' in get_ipython().config:  # type: ignore
        IN_NOTEBOOK = True
        import pandas as pd  # type: ignore
        from IPython.display import display, HTML, Markdown  # type: ignore
        import matplotlib.pyplot as plt  # type: ignore
    else:
        IN_NOTEBOOK = False
except:
    IN_NOTEBOOK = False

console = Console()

# reading the csv file which contain all input data for simplex
with open('testcase_1.csv', newline='') as csvfile:
    reader = csv.reader(csvfile)
    data = list(reader)

# extracting initial feasable point from first row (except last column)
z = np.array([float(val) for val in data[0][:-1]])
# extracting cost vector from secound row (except last column)
c = np.array([float(val) for val in data[1][:-1]])
# extracting constraint vector b from last column of all rows starting from 3rd
b = np.array([float(row[-1]) for row in data[2:]])
# extracting constraint matix A from remaining data
A = np.array([[float(val) for val in row[:-1]] for row in data[2:]])

# m is number of constraint, n is number of variable
m, n = len(b), len(c)

console.print("\n[bold cyan]═══════════════════════════════════════════════════════════[/bold cyan]")
console.print("[bold yellow]           SIMPLEX ALGORITHM - INPUT DATA[/bold yellow]")
console.print("[bold cyan]═══════════════════════════════════════════════════════════[/bold cyan]\n")

input_table = Table(show_header=True, header_style="bold magenta", box=box.ROUNDED)
input_table.add_column("Parameter", style="cyan", width=30)
input_table.add_column("Value", style="green")

input_table.add_row("Initial Feasible Point (z)", str(z))
input_table.add_row("Cost Vector (c)", str(c))
input_table.add_row("Constraint Vector (b)", str(b))
input_table.add_row("Dimensions (m × n)", f"{m} × {n}")

console.print(input_table)

if IN_NOTEBOOK:
    display(Markdown("### Matrix A:"))  # type: ignore
    df_A = pd.DataFrame(A, columns=[f'x{i+1}' for i in range(n)],  # type: ignore
                        index=[f'Constraint {i+1}' for i in range(m)])
    display(df_A)  # type: ignore
else:
    console.print("\n[bold cyan]Matrix A:[/bold cyan]")
    console.print(Panel(str(A), border_style="cyan", box=box.ROUNDED))

console.print()

# ========== PHASE 1: CONVERTING FEASABLE POINT TO VERTEX ==========
# in this phase we will move from initial feasable point to a vertex
console.print("[bold cyan]═══════════════════════════════════════════════════════════[/bold cyan]")
console.print("[bold yellow]        PHASE 1: Feasible Point to Vertex[/bold yellow]")
console.print("[bold cyan]═══════════════════════════════════════════════════════════[/bold cyan]\n")

track_cost = [np.dot(c, z)]
track_z = [z]

# checking which constraints are tight at current point z
# constraint is tight if A*z = b (with small tolerence for numerical error)
product = np.dot(A, z)
mask = np.abs(product - b) < 1e-8
tight_rows = A[mask]
untight_rows = A[~mask]
rank = 0 if len(tight_rows) == 0 else np.linalg.matrix_rank(tight_rows)

# if rank of tight constraints equal to n, then we are at vertex!
if rank == n:
    console.print(Panel("[bold green]Already at a vertex![/bold green]", border_style="green", box=box.DOUBLE))
    z_vertex = z
    cost_to_vertex = track_cost
    path_to_vertex = track_z
else:
    console.print("[bold yellow]Not at a vertex. Searching for a vertex...[/bold yellow]\n")
    z_old = z
    z_new = z
    iteration = 0
    
    # keep moving untill rank become n (which mean we reach vertex)
    while rank != n:
        iteration += 1
        if iteration % 1 == 0:
            console.print(f"[cyan]Iteration {iteration}[/cyan] | [magenta]Rank: {rank}[/magenta]")
        
        # finding direction to move
        if len(tight_rows) == 0:
            u = np.random.rand(untight_rows.shape[-1])
        else:
            # move in direction perpendiculer to tight constraints using null space
            null_space_matrix = null_space(tight_rows)
            u = null_space_matrix[:, 0]
        
        # calculating how far we can move before hitting new constraint
        alphas = [(b_i - np.dot(a2_i, z_old)) / np.dot(a2_i, u) 
                  for b_i, a2_i in zip(b[~mask], untight_rows)]
        alpha = min([a for a in alphas if a > 0])
        
        z_new = z_old + alpha * u
        product = np.dot(A, z_new)
        mask = np.abs(product - b) < 1e-8
        tight_rows = A[mask]
        untight_rows = A[~mask]
        z_old = z_new
        
        rank = 0 if len(tight_rows) == 0 else np.linalg.matrix_rank(tight_rows)
        track_cost.append(np.dot(c, z_new))
        track_z.append(z_new)
    
    console.print(f"\n[bold green]Vertex found! | Rank: {rank}[/bold green]")
    z_vertex = z_new
    cost_to_vertex = track_cost
    path_to_vertex = track_z

console.print("\n[bold]Initial Vertex:[/bold]", style="yellow")
console.print(Panel(str(z_vertex), border_style="yellow", box=box.ROUNDED))
console.print(f"[bold]Costs during Phase 1:[/bold] [green]{cost_to_vertex}[/green]\n")

if IN_NOTEBOOK:
    display(Markdown("### Phase 1: Cost Progression"))  # type: ignore
    fig, ax = plt.subplots(figsize=(10, 4))  # type: ignore
    ax.plot(range(1, len(cost_to_vertex) + 1), cost_to_vertex, 
            marker='o', linewidth=2, markersize=8, color='#2196F3')
    ax.set_xlabel('Iteration', fontsize=12, fontweight='bold')
    ax.set_ylabel('Objective Value', fontsize=12, fontweight='bold')
    ax.set_title('Phase 1: Feasible Point to Vertex', fontsize=14, fontweight='bold')
    ax.grid(True, alpha=0.3)
    for i, cost in enumerate(cost_to_vertex):
        ax.annotate(f'{cost:.2f}', (i+1, cost), textcoords="offset points", 
                    xytext=(0,10), ha='center', fontsize=9)
    plt.tight_layout()  # type: ignore
    plt.show()  # type: ignore


# ========== PHASE 2: MOVING FROM VERTEX TO OPTIMAL VERTEX ==========
console.print("[bold cyan]═══════════════════════════════════════════════════════════[/bold cyan]")
console.print("[bold yellow]       PHASE 2: Vertex to Optimal Vertex[/bold yellow]")
console.print("[bold cyan]═══════════════════════════════════════════════════════════[/bold cyan]\n")

track_cost = [np.dot(c, z_vertex)]
track_vertex = [z_vertex]
z_old = z_vertex
iteration = 0

# keep going untill we reach optimal vertex
while True:
    iteration += 1
    console.print(f"[cyan]Iteration {iteration}[/cyan]")
    
    product = np.dot(A, z_old)
    mask = np.abs(product - b) < 1e-8
    tight_rows = A[mask]
    untight_rows = A[~mask]
    
    # calculating all posible directions we can move from this vertex
    try:
        directions = (-np.linalg.inv(tight_rows)).T
    except np.linalg.LinAlgError:
        console.print(Panel("[bold red]Matrix is singular. Cannot compute the inverse.[/bold red]", border_style="red", box=box.DOUBLE))
        z_optimal = z_old
        cost_to_optimal = track_cost
        path_to_optimal = track_vertex
        break
    
    # filtering only those directions which increase the objective function
    positive_directions = [d for d in directions if np.dot(d, c) > 0]
    
    # if no direction increase objective, we are at optimal vertex!
    if not positive_directions:
        console.print(Panel("[bold green]Reached the optimal vertex![/bold green]", border_style="green", box=box.DOUBLE))
        z_optimal = z_old
        cost_to_optimal = track_cost
        path_to_optimal = track_vertex
        break
    
    u = positive_directions[0]
    
    # calculating how far we can go before hitting another constraint
    alphas = [(b_i - np.dot(a2_i, z_old)) / np.dot(a2_i, u) 
              for b_i, a2_i in zip(b[~mask], untight_rows)]
    positive_alphas = [a for a in alphas if a > 0]
    console.print(f"  [dim]Positive alphas count: {len(positive_alphas)}[/dim]")
    alpha = min(positive_alphas)
    
    z_new = z_old + alpha * u
    z_old = z_new
    
    track_cost.append(np.dot(c, z_new))
    track_vertex.append(z_new)

console.print("\n[bold]Optimal Vertex:[/bold]", style="green")
console.print(Panel(str(z_optimal), border_style="green", box=box.ROUNDED))
console.print(f"[bold]Costs during Phase 2:[/bold] [green]{cost_to_optimal}[/green]\n")

if IN_NOTEBOOK:
    display(Markdown("### Phase 2: Cost Progression"))  # type: ignore
    fig, ax = plt.subplots(figsize=(10, 4))  # type: ignore
    ax.plot(range(1, len(cost_to_optimal) + 1), cost_to_optimal, 
            marker='s', linewidth=2, markersize=8, color='#4CAF50')
    ax.set_xlabel('Iteration', fontsize=12, fontweight='bold')
    ax.set_ylabel('Objective Value', fontsize=12, fontweight='bold')
    ax.set_title('Phase 2: Vertex to Optimal Vertex', fontsize=14, fontweight='bold')
    ax.grid(True, alpha=0.3)
    for i, cost in enumerate(cost_to_optimal):
        ax.annotate(f'{cost:.2f}', (i+1, cost), textcoords="offset points", 
                    xytext=(0,10), ha='center', fontsize=9)
    plt.tight_layout()  # type: ignore
    plt.show()  # type: ignore


# ========== FINAL SUMMARY ==========
console.print("[bold cyan]═══════════════════════════════════════════════════════════[/bold cyan]")
console.print("[bold yellow]                    SUMMARY[/bold yellow]")
console.print("[bold cyan]═══════════════════════════════════════════════════════════[/bold cyan]\n")

summary_table = Table(show_header=True, header_style="bold magenta", box=box.DOUBLE)
summary_table.add_column("Metric", style="cyan", width=35)
summary_table.add_column("Value", style="green", width=20)

# we subtract 1 because last vertex of phase 1 is same as first vertex of phase 2
total_vertices = len(path_to_vertex) + len(path_to_optimal) - 1
summary_table.add_row("All vertices visited", str(total_vertices))
summary_table.add_row("Optimal objective value", f"{cost_to_optimal[-1]:.6f}")
summary_table.add_row("Optimal solution", str(z_optimal))

console.print(summary_table)
console.print("\n[bold green]Simplex algorithm completed successfully![/bold green]\n")

if IN_NOTEBOOK:
    display(Markdown("### Combined Cost Progression (Both Phases)"))  # type: ignore
    all_costs = cost_to_vertex + cost_to_optimal[1:]
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 5))  # type: ignore

    ax1.plot(range(1, len(cost_to_vertex) + 1), cost_to_vertex, 
             marker='o', linewidth=2, markersize=8, color='#2196F3', label='Phase 1')
    phase2_x = range(len(cost_to_vertex), len(cost_to_vertex) + len(cost_to_optimal))
    ax1.plot(phase2_x, cost_to_optimal, 
             marker='s', linewidth=2, markersize=8, color='#4CAF50', label='Phase 2')
    ax1.axvline(x=len(cost_to_vertex), color='red', linestyle='--', alpha=0.7, label='Phase Transition')
    ax1.set_xlabel('Iteration', fontsize=12, fontweight='bold')
    ax1.set_ylabel('Objective Value', fontsize=12, fontweight='bold')
    ax1.set_title('Full Algorithm Progression', fontsize=14, fontweight='bold')
    ax1.legend()
    ax1.grid(True, alpha=0.3)

    ax2.axis('tight')
    ax2.axis('off')
    summary_data = {
        'Metric': ['Phase 1 Iterations', 'Phase 2 Iterations', 'Total Vertices Visited', 
                   'Initial Cost', 'Final Cost', 'Cost Improvement'],
        'Value': [len(cost_to_vertex) - 1, len(cost_to_optimal) - 1, total_vertices,
                  f'{cost_to_vertex[0]:.6f}', f'{cost_to_optimal[-1]:.6f}',
                  f'{cost_to_optimal[-1] - cost_to_vertex[0]:.6f}']
    }
    df_summary = pd.DataFrame(summary_data)  # type: ignore
    table = ax2.table(cellText=df_summary.values, colLabels=df_summary.columns,
                      cellLoc='center', loc='center', 
                      colWidths=[0.6, 0.4])
    table.auto_set_font_size(False)
    table.set_fontsize(10)
    table.scale(1, 2)
    for i in range(len(df_summary.columns)):
        table[(0, i)].set_facecolor('#4CAF50')
        table[(0, i)].set_text_props(weight='bold', color='white')
    ax2.set_title('Summary Statistics', fontsize=14, fontweight='bold', pad=20)

    plt.tight_layout()  # type: ignore
    plt.show()  # type: ignore

    # type: ignore
    display(    # type: ignore
        HTML(    # type: ignore
            f"""
    <div style="background: linear-gradient(135deg, #667eea 0%, #764ba2 100%); 
                padding: 20px; border-radius: 10px; color: white; margin: 20px 0;">
        <h2 style="margin: 0 0 15px 0; text-align: center;">Optimization Results</h2>
        <div style="background: rgba(255,255,255,0.1); padding: 15px; border-radius: 5px;">
            <p style="margin: 5px 0; font-size: 16px;"><strong>Optimal Solution:</strong> {z_optimal}</p>
            <p style="margin: 5px 0; font-size: 16px;"><strong>Optimal Value:</strong> {cost_to_optimal[-1]:.6f}</p>
            <p style="margin: 5px 0; font-size: 16px;"><strong>Total Iterations:</strong> {total_vertices}</p>
        </div>
    </div>
    """))  # type: ignore
