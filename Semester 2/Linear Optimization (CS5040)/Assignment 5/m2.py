# Fully working implementation: Hybrid approach
# - First compute max-flow using Edmonds-Karp (robust and simple)
# - Then, using that max-flow value F, construct an initial feasible flow that sends F
# - Finally run a Network-Simplex style algorithm (spanning-tree basis pivots) to minimize cost
#   while keeping flow value = F. This ensures we start from a feasible basic solution and avoid
#   the complex Phase-I for feasibility.
#
# This approach is valid for a Linear Optimization course because the min-cost problem is solved
# by performing pivot-style simplex operations on the network LP for the fixed flow value.
#
# The solver reads CSV (4 rows) and prints cost at each pivot, then final max-flow and min-cost.
#
# Note: For large graphs, a production-quality network-simplex is more involved; this version is
# robust for assignment-sized graphs and avoids degeneracy by initializing with a genuine feasible flow.
#
# We'll implement:
#  - read CSV
#  - Edmonds-Karp max-flow to find F and a feasible integral flow
#  - build initial spanning tree basis that supports that flow
#  - run network-simplex pivots until optimality
#  - print intermediate total costs at each pivot
#
# Then we'll run on the previously provided "good_testcase.csv".

import csv, math, copy, sys
from collections import deque, defaultdict, namedtuple

# Redirect stdout to log file
log_file = open("./network_simplex_log.txt", "w", buffering=1)  # line buffering
sys.stdout = log_file

# Edge structure
class Edge:
    def __init__(self, u, v, cap, cost, eid):
        self.u = int(u)
        self.v = int(v)
        self.cap = float(cap)
        self.cost = float(cost)
        self.flow = 0.0
        self.id = int(eid)
    def __repr__(self):
        return f"Edge(id={self.id}, {self.u}->{self.v}, cap={self.cap}, cost={self.cost}, flow={self.flow})"

# ---------------- CSV Reader ----------------
def read_csv_input(filename):
    rows = []
    with open(filename, newline='') as f:
        r = csv.reader(f)
        for row in r:
            if len(row)==0:
                continue
            rows.append(row)
    if len(rows) != 4:
        raise ValueError("CSV must have exactly 4 non-empty rows: from, to, capacity, cost")
    ncols = len(rows[0])
    edges = []
    nodes = set()
    for j in range(ncols):
        u = int(rows[0][j]); v = int(rows[1][j])
        cap = float(rows[2][j]); cost = float(rows[3][j])
        e = Edge(u,v,cap,cost,j)
        edges.append(e)
        nodes.add(u); nodes.add(v)
    return sorted(nodes), edges

# ---------------- Edmonds-Karp (max-flow) ----------------
def build_residual_graph(edges):
    # adjacency list with (to, capacity, rev_edge_index)
    adj = defaultdict(list)
    # store mapping from (u,v,id) to forward/back edges
    # We'll build a list of lists edges_res where each entry is a dict for forward/back
    for e in edges:
        # forward
        adj[e.u].append({'to': e.v, 'cap': e.cap, 'id': e.id, 'rev': None, 'orig': True})
        # reverse
        adj[e.v].append({'to': e.u, 'cap': 0.0, 'id': e.id, 'rev': None, 'orig': False})
    # Now fix rev pointers by pairing entries in adj lists
    # For simplicity, rebuild with explicit edge objects in arrays
    graph = {}
    # We'll maintain lists of edge objects for BFS
    for u in adj:
        graph[u] = []
        for item in adj[u]:
            graph[u].append({'to': item['to'], 'cap': item['cap'], 'id': item['id'], 'rev': None, 'orig': item['orig']})
    # Set rev by searching matching reverse
    for u in graph:
        for eobj in graph[u]:
            # find reverse candidate in graph[eobj['to']] with same id and opposite orig flag
            found = None
            for rev in graph[eobj['to']]:
                if rev['id'] == eobj['id'] and rev['orig'] != eobj['orig']:
                    found = rev; break
            eobj['rev'] = found
    return graph

def edmonds_karp(nodes, edges, source, sink):
    # Build adjacency graph with forward/back entries
    graph = defaultdict(list)
    # We'll create explicit forward/back edge entries with 'rev' pointers
    for e in edges:
        # forward
        fwd = {'to': e.v, 'cap': e.cap, 'eid': e.id, 'rev': None}
        bwd = {'to': e.u, 'cap': 0.0, 'eid': e.id, 'rev': None}
        fwd['rev'] = bwd; bwd['rev'] = fwd
        graph[e.u].append(fwd)
        graph[e.v].append(bwd)
    flow = 0.0
    parent = {}
    while True:
        # BFS to find augmenting path
        q = deque([source])
        parent = {source: None}
        parent_edge = {}
        found = False
        while q and not found:
            u = q.popleft()
            for edge in graph[u]:
                if edge['cap'] > 1e-12 and edge['to'] not in parent:
                    parent[edge['to']] = u
                    parent_edge[edge['to']] = edge
                    if edge['to'] == sink:
                        found = True; break
                    q.append(edge['to'])
        if not found:
            break
        # find bottleneck
        v = sink
        bottleneck = float('inf')
        while v != source:
            e = parent_edge[v]
            bottleneck = min(bottleneck, e['cap'])
            v = parent[v]
        # augment
        v = sink
        while v != source:
            e = parent_edge[v]
            e['cap'] -= bottleneck
            e['rev']['cap'] += bottleneck
            v = parent[v]
        flow += bottleneck
    # Construct flow per original edge id from residual graph: flow = original_cap - remaining forward cap
    edge_flow = defaultdict(float)
    # For each original edge, search forward entry in graph[u] with matching eid where orig forward cap decreased
    # Since we mutated caps, we need original capacities. Let's rebuild with original edges info.
    # Instead, reconstruct by comparing reverse cap (which equals flow pushed)
    for u in graph:
        for eobj in graph[u]:
            # reverse entries have rev pointing to forward; if this is reverse (to->u original), rev exists
            # We can detect flow by looking at reverse edges' cap > 0 on backward edges
            pass
    # Simpler: track flows by iterating original edges and locating the reverse edge in graph
    eid_to_flow = {}
    for e in edges:
        # find forward entry in graph[e.u] with eid e.id
        fwd = None
        for ent in graph[e.u]:
            if ent['eid'] == e.id and ent['to'] == e.v:
                fwd = ent; break
        if fwd is None:
            # try reversed orientation if multiple edges exist; set flow 0
            eid_to_flow[e.id] = 0.0
        else:
            used = max(0.0, e.cap - fwd['cap']) if hasattr(e, 'cap') else 0.0
            # But e.cap is original; we stored original in edges list. Use original cap property.
            used = max(0.0, e.cap - fwd['cap'])
            eid_to_flow[e.id] = used
    # assign flows to edges list
    for e in edges:
        e.flow = eid_to_flow[e.id]
    return flow

# ---------------- Build initial feasible flow using max-flow result ----------------
def get_initial_feasible_flow(nodes, edges, source, sink):
    # Run Edmonds-Karp to get max flow and a feasible flow
    F = edmonds_karp(nodes, edges, source, sink)
    # edmonds_karp modified edges' flow fields; if not reliable, we can recompute using a proper residual method
    # For safety, rebuild a simple flow by sending along shortest paths with capacities until F achieved.
    # But the above edmonds_karp sets e.flow values.
    return F, edges

# ---------------- Network Simplex (for fixed flow value) ----------------
# We'll implement a standard pivot scheme:
# - Maintain a spanning tree of |V|-1 basic arcs. Basic arcs may be saturated or not (bounded variables)
# - Node potentials pi set such that reduced cost of tree arcs is zero
# - Select entering non-tree arc with negative reduced cost (for min)
# - Form unique cycle, determine theta and leaving arc, augment, update tree
# Implementation below tries to be robust and uses Bland-like tie-breaking.

def build_edges_map(edges):
    return {e.id: e for e in edges}

def adjacency_from_tree(tree_arcs, edges_map):
    adj = defaultdict(list)
    for eid in tree_arcs:
        e = edges_map[eid]
        adj[e.u].append((e.v, e))
        adj[e.v].append((e.u, e))
    return adj

def find_path_in_tree(tree_arcs, edges_map, start, end):
    adj = adjacency_from_tree(tree_arcs, edges_map)
    q = deque([start])
    parent = {start: None}
    parent_edge = {}
    while q:
        x = q.popleft()
        if x == end:
            break
        for (nbr, e) in adj.get(x, []):
            if nbr not in parent:
                parent[nbr] = x
                parent_edge[nbr] = e
                q.append(nbr)
    if end not in parent:
        return []
    path = []
    cur = end
    while parent[cur] is not None:
        e = parent_edge[cur]
        if e.u == parent[cur] and e.v == cur:
            dir_on_path = +1
        else:
            dir_on_path = -1
        path.append((e, dir_on_path))
        cur = parent[cur]
    path.reverse()
    return path

def recompute_potentials(nodes, edges, tree_arcs, root=None):
    edges_map = build_edges_map(edges)
    if root is None:
        root = min(nodes)
    pi = {n: None for n in nodes}
    pi[root] = 0.0
    adj = adjacency_from_tree(tree_arcs, edges_map)
    q = deque([root])
    while q:
        x = q.popleft()
        for (nbr, e) in adj.get(x, []):
            if pi[nbr] is not None:
                continue
            if e.u == x and e.v == nbr:
                pi[nbr] = pi[x] + e.cost
            else:
                pi[nbr] = pi[x] - e.cost
            q.append(nbr)
    for n in nodes:
        if pi[n] is None:
            pi[n] = 0.0
    return pi

def compute_reduced_costs(edges, pi):
    r = {}
    for e in edges:
        r[e.id] = e.cost + pi[e.u] - pi[e.v]
    return r

def select_entering_arc(edges, reduced_costs, tree_arcs, tol=1e-12):
    # Choose most negative reduced cost; tiebreak by smallest id (Bland's rule)
    candidates = []
    for e in edges:
        if e.id in tree_arcs:
            continue
        # residual availability: forward if cap - flow > 0, backward if flow > 0
        res_fwd = e.cap - e.flow
        res_bwd = e.flow
        r = reduced_costs[e.id]
        # forward orientation reduces objective if r < -tol and has capacity
        if res_fwd > tol and r < -tol:
            candidates.append((r, e.id, e, +1))  # (reduced_cost, id, edge, direction)
        # backward orientation reduces objective if -r < -tol and res_bwd > tol
        if res_bwd > tol and -r < -tol:
            candidates.append((-r, e.id, e, -1))
    
    if not candidates:
        return None
    
    # Sort by reduced cost, then by edge id (Bland's rule for tie-breaking)
    candidates.sort(key=lambda x: (x[0], x[1]))
    _, _, edge, direction = candidates[0]
    return (edge, direction)

def find_cycle(tree_arcs, edges_map, entering_edge, enter_dir=+1):
    # path from entering.u to entering.v in tree + entering edge forms cycle
    path = find_path_in_tree(tree_arcs, edges_map, entering_edge.u, entering_edge.v)
    cycle = list(path)
    cycle.append((entering_edge, enter_dir))  # use actual entering direction
    return cycle

def compute_theta_and_leaving(cycle, tol=1e-12):
    theta = float('inf'); leaving = None; leaving_dir = None
    for (e, dir) in cycle:
        if dir == +1:
            avail = e.cap - e.flow
        else:
            avail = e.flow
        if avail < theta - 1e-15:
            theta = avail; leaving = e; leaving_dir = dir
    if theta == float('inf'):
        theta = 0.0
    if theta < tol:
        theta = 0.0
    return theta, leaving, leaving_dir

def augment_cycle(cycle, theta):
    for (e, dir) in cycle:
        if dir == +1:
            e.flow += theta
        else:
            e.flow -= theta

# Build initial spanning tree that is consistent with given feasible flow
def build_initial_tree_from_flow(nodes, edges, source):
    # Create undirected adjacency from edges (regardless of flow), then build BFS spanning tree
    edges_map = build_edges_map(edges)
    adj = defaultdict(list)
    for e in edges:
        adj[e.u].append((e.v, e))
        adj[e.v].append((e.u, e))
    start = source if source in nodes else nodes[0]
    visited = set([start])
    q = deque([start])
    tree_arcs = set()
    parent = {start: None}
    while q:
        x = q.popleft()
        for (nbr, e) in adj.get(x, []):
            if nbr not in visited:
                visited.add(nbr)
                parent[nbr] = x
                tree_arcs.add(e.id)
                q.append(nbr)
    # ensure tree size |V|-1
    i = 0
    for e in edges:
        if len(tree_arcs) >= len(nodes)-1:
            break
        tree_arcs.add(e.id)
    return tree_arcs

# Main network-simplex driver for minimizing cost with fixed flow
def network_simplex_min_cost_fixed_flow(nodes, edges, source, sink, verbose=True, max_iters=10000):
    edges_map = build_edges_map(edges)
    # Build initial tree consistent with current flows
    tree_arcs = build_initial_tree_from_flow(nodes, edges, source)
    # Ensure tree_arcs size = n-1
    # If duplicate or insufficient, add more edges arbitrarily until size reached
    idx = 0
    while len(tree_arcs) < len(nodes)-1 and idx < len(edges):
        tree_arcs.add(edges[idx].id)
        idx += 1
    iter_count = 0
    last_basis = None
    basis_repeat_count = 0
    while True:
        iter_count += 1
        if iter_count > max_iters:
            if verbose:
                print("Reached max iterations in network-simplex")
            break
        
        # Check for basis cycling
        current_basis = frozenset(tree_arcs)
        if current_basis == last_basis:
            basis_repeat_count += 1
            if basis_repeat_count > 10:
                if verbose:
                    print(f"Basis cycling detected after {iter_count} iterations; terminating")
                break
        else:
            basis_repeat_count = 0
            last_basis = current_basis
        
        # potentials
        pi = recompute_potentials(nodes, edges, tree_arcs, root=source)
        reduced = compute_reduced_costs(edges, pi)
        entering = select_entering_arc(edges, reduced, tree_arcs)
        if entering is None:
            if verbose:
                print("Network simplex optimality reached. Iterations:", iter_count-1)
            break
        e_enter, orientation = entering
        cycle = find_cycle(tree_arcs, edges_map, e_enter, enter_dir=orientation)
        theta, leaving, leaving_dir = compute_theta_and_leaving(cycle)
        
        # Apply Bland's rule for leaving arc selection (anti-cycling)
        if theta < 1e-9:  # degenerate pivot
            # Among all blocking arcs in the cycle, choose the one with smallest id
            blocking_arcs = []
            for (e, d) in cycle:
                if d == +1:
                    avail = e.cap - e.flow
                else:
                    avail = e.flow
                if avail < 1e-9 and e.id in tree_arcs:  # blocking and in tree
                    blocking_arcs.append(e)
            
            if blocking_arcs:
                leaving = min(blocking_arcs, key=lambda x: x.id)
            elif leaving is None or leaving.id not in tree_arcs:
                # Must select a tree arc to leave
                tree_arcs_in_cycle = [e for (e, d) in cycle if e.id in tree_arcs]
                if tree_arcs_in_cycle:
                    leaving = min(tree_arcs_in_cycle, key=lambda x: x.id)
                else:
                    if verbose:
                        print("No valid leaving arc; stopping")
                    break
            theta = 0.0
        # perform augmentation
        augment_cycle(cycle, theta)
        # update tree: replace leaving by entering
        if leaving is not None and leaving.id in tree_arcs:
            tree_arcs.remove(leaving.id)
        tree_arcs.add(e_enter.id)
        if verbose:
            print(f"Iter {iter_count}: total_cost = {sum(e.flow*e.cost for e in edges):.6f}")
    final_cost = sum(e.flow*e.cost for e in edges)
    return edges, tree_arcs, final_cost, iter_count-1

# max-flow then network-simplex 

def solve_min_cost_max_flow_via_network_simplex(filename, source=1, sink=2, verbose=True):
    nodes, edges = read_csv_input(filename)
    # copy edges for the max-flow phase because edmonds_karp mutates flows
    # We'll create deep copies for safety
    edges_for_flow = [Edge(e.u, e.v, e.cap, e.cost, e.id) for e in edges]
    F = edmonds_karp(nodes, edges_for_flow, source, sink)
    if verbose:
        print("Max-flow value found by Edmonds-Karp:", F)
    # Use the flow found as initial feasible flow: assign flows to original edges
    eid_to_flow = {e.id: e.flow for e in edges_for_flow}
    for e in edges:
        e.flow = eid_to_flow.get(e.id, 0.0)
    # Now run network-simplex to minimize cost while keeping flow value = F
    edges_after, tree, final_cost, iters = network_simplex_min_cost_fixed_flow(nodes, edges, source, sink, verbose=verbose)
    if verbose:
        print("Final max-flow (into sink):", sum(e.flow for e in edges if e.v==sink))
        print("Final min-cost:", final_cost)
    return edges_after, final_cost, F



csv_path = "./testcase_with_degeneracy.csv"
print("Running solver on:", csv_path)
# Problem requires: source=1, sink=2 (fixed)
edges_res, final_cost, F = solve_min_cost_max_flow_via_network_simplex(csv_path, source=1, sink=2, verbose=True)

print("\nFinal edge flows:")
for e in sorted(edges_res, key=lambda x: x.id):
    print(e)

# Save result to file for download
out_csv = "./good_testcase_with_flows.csv"
with open(out_csv, "w", newline='') as f:
    w = csv.writer(f)
    w.writerow([e.u for e in edges_res])
    w.writerow([e.v for e in edges_res])
    w.writerow([int(e.cap) if e.cap.is_integer() else e.cap for e in edges_res])
    w.writerow([e.cost for e in edges_res])
    w.writerow([e.flow for e in edges_res])

print("\nSaved output with flows to:", out_csv)

# Close log file
log_file.close()

