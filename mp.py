def dfs(graph, root):
    visited = []
    tot_dist = {x: 99999 for x in graph.keys()}
    stack = [(root, 0)]
    while stack:
        node, dist = stack.pop()
        tot_dist[node] = min(tot_dist[node], dist)
        if node not in visited:
            visited.append(node)
            stack.extend([(x, dist + 1)for x in graph[node] if x not in visited])
    return visited, tot_dist


graph = {"a": ["c"], "b": ["c", "e"], "c": ["a", "b", "d", "e"], "d": ["c"], "e": ["c", "b"], "f": []}

v, t = dfs(graph, "a")
print(v)
print(t)
