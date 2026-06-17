import osmnx as ox
import networkx as nx

G = ox.graph_from_point((12.9352, 77.6245), dist=2000, network_type="drive")

def find_best_scenario(G, incident_node, min_length=600, n_candidates=3):
    """Find source/dest pairs with meaningful distance so the detour is visible."""
    
    # BFS outward from incident to get nodes at various depths
    def bfs_nodes(G, start, depth, direction="pred"):
        visited = {start}
        frontier = {start}
        for _ in range(depth):
            next_frontier = set()
            for n in frontier:
                neighbors = G.predecessors(n) if direction == "pred" else G.successors(n)
                for nb in neighbors:
                    if nb not in visited:
                        next_frontier.add(nb)
                        visited.add(nb)
            frontier = next_frontier
        return frontier

    # Sources = nodes 4-8 hops BEFORE incident
    sources = bfs_nodes(G, incident_node, depth=6, direction="pred")
    # Dests   = nodes 4-8 hops AFTER incident  
    dests   = bfs_nodes(G, incident_node, depth=6, direction="succ")
    
    print(f"  Searching {len(sources)} sources × {len(dests)} dests (min {min_length}m)...")
    
    valid = []
    for src in sources:
        for dst in dests:
            if src == dst or src == incident_node or dst == incident_node:
                continue
            try:
                path = nx.shortest_path(G, src, dst, weight="length")
                if incident_node not in path:
                    continue
                length = nx.shortest_path_length(G, src, dst, weight="length")
                if length < min_length:
                    continue
                
                # Verify diversion is actually different
                G_div = G.copy()
                G_div.remove_node(incident_node)
                try:
                    div_path = nx.shortest_path(G_div, src, dst, weight="length")
                    div_len  = nx.shortest_path_length(G_div, src, dst, weight="length")
                    extra    = div_len - length
                    if extra > 0:  # must actually add distance
                        valid.append((src, dst, length, div_len, extra, path))
                except nx.NetworkXNoPath:
                    pass
            except nx.NetworkXNoPath:
                continue
    
    valid.sort(key=lambda x: -x[4])  # sort by most detour first
    
    print(f"  Found {len(valid)} valid pairs with real detour:")
    for src, dst, orig, div, extra, _ in valid[:n_candidates]:
        print(f"    source={src}  dest={dst}  orig={orig:.0f}m  diverted={div:.0f}m  extra=+{extra:.0f}m")
    
    return valid[:n_candidates]


incidents = {
    "Forum Mall Signal":     3145158631,
    "Sony World Junction":   305092008,
}

for name, node in incidents.items():
    print(f"\n>>> {name}  (node {node})")
    find_best_scenario(G, node, min_length=600)

# For Scenario 3, swap to a node that actually bottlenecks traffic
# 3145158635 is on the path to Forum Mall AND Ejipura corridor
print(f"\n>>> Replacing '3rd Block' with bottleneck node search")
find_best_scenario(G, 3145158635, min_length=600)
find_best_scenario(G, 1272830640, min_length=600)