# Max-Flow Min-Cost Solver

## Algorithm Approach

### Two-Phase Hybrid Method:

1. **Phase 1: Edmonds-Karp (Max-Flow)**

   - Uses BFS to find augmenting paths in residual graph
   - Computes maximum flow value F from source to sink

2. **Phase 2: Network Simplex (Min-Cost)**
   - Maintains spanning tree basis with |V|-1 arcs
   - Computes node potentials (dual variables) and reduced costs
   - Selects entering arc with most negative reduced cost
   - Forms cycle, computes theta (bottleneck), and pivots
   - Uses **Bland's rule** for degeneracy handling (anti-cycling)

### Key Features:

- Handles degenerate pivots without cycling
- Guarantees optimal solution for min-cost max-flow problems
- Source = 1, Sink = 2 (per problem requirement)

### Files:

- `m2.py` - Python script implementation
- `m2.ipynb` - Jupyter notebook with detailed sections
- `testcase_no_degeneracy.csv` - Test case without degeneracy
- `testcase_with_degeneracy.csv` - Test case with degeneracy
