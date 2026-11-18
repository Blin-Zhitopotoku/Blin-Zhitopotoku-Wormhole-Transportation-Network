"""Randomized routing and local improvement; uses only the standard library."""

import random


def _edge_index(indptr, targets, source, target):
    lo, hi = indptr[source], indptr[source+1]
    end = hi
    while lo < hi:
        mid = (lo+hi)//2
        if targets[mid] < target:
            lo = mid+1
        else:
            hi = mid
    return lo if lo < end and targets[lo] == target else -1

def _improve_route(path, indptr, targets, means, variances, delay, lower, upper, blocked):
    """Improve one/two-edge segments while preserving arrival and disjointness."""
    path = path.copy()
    length = len(path)
    used = blocked.copy()
    arrival = float(delay)
    for j in range(length):
        used[path[j]] = True
        if j+1 < length:
            arrival += means[_edge_index(indptr, targets, path[j], path[j+1])]
    for sweep in range(3):
        changed = False
        j = 0
        while j+2 < length:
            a, b, c = path[j], path[j+1], path[j+2]
            ab = _edge_index(indptr, targets, a, b)
            bc = _edge_index(indptr, targets, b, c)
            old_mean = float(means[ab]) + means[bc]
            old_cost = float(variances[ab]) + variances[bc]
            best_cost, new_mean, middle = old_cost, old_mean, b
            direct = _edge_index(indptr, targets, a, c)
            if direct >= 0 and lower <= arrival-old_mean+means[direct] <= upper:
                if variances[direct] < best_cost:
                    best_cost, new_mean, middle = variances[direct], means[direct], -1
            for ax in range(indptr[a], indptr[a+1]):
                x = targets[ax]
                if used[x] or variances[ax] >= best_cost:
                    continue
                xc = _edge_index(indptr, targets, x, c)
                if xc < 0:
                    continue
                candidate_cost = float(variances[ax]) + variances[xc]
                candidate_mean = float(means[ax]) + means[xc]
                if (candidate_cost < best_cost
                        and lower <= arrival-old_mean+candidate_mean <= upper):
                    best_cost, new_mean, middle = candidate_cost, candidate_mean, x
            if middle != b:
                used[b] = False
                arrival += new_mean-old_mean
                if middle < 0:
                    for k in range(j+1, length-1):
                        path[k] = path[k+1]
                    length -= 1
                else:
                    path[j+1] = middle
                    used[middle] = True
                changed = True
            j += 1
        if not changed:
            break
    return path[:length]

def _construct(indptr, targets, means, variances, origins, delay, destination,
               jump_limit, lower, upper, blocked, successor, suffix_mean,
               suffix_hops, attempts, seed):
    """Randomized benefit/cost walks; finish with a feasible shortest suffix."""
    rng = random.Random(seed)
    best_cost = float('inf')
    best_path = [0] * (jump_limit + 1)
    best_length = 0
    for attempt in range(attempts):
        visited = blocked.copy()
        path = [0] * (jump_limit + 1)
        node = origins[attempt % len(origins)]
        if visited[node]:
            continue
        path[0] = node
        length = 1
        clock = float(delay)
        cost = 0.0
        # Vary the minimum temporal progress required per remaining jump.
        urgency = 0.9 + 0.3 * rng.random()
        noise = 0.0 if attempt < len(origins) else 0.35
        while length <= jump_limit and cost < best_cost:
            visited[node] = True
            arrival = clock + suffix_mean[node]
            if lower <= arrival <= upper and length-1+suffix_hops[node] <= jump_limit:
                tail = node
                extra_cost = 0.0
                valid = True
                for _ in range(suffix_hops[node]):
                    nxt = successor[tail]
                    if nxt < 0 or visited[nxt]:
                        valid = False
                        break
                    for edge in range(indptr[tail], indptr[tail+1]):
                        if targets[edge] == nxt:
                            extra_cost += variances[edge]
                            break
                    tail = nxt
                if valid and cost + extra_cost < best_cost:
                    best_cost = cost + extra_cost
                    best_length = length
                    best_path[:length] = path[:length]
                    tail = node
                    for _ in range(suffix_hops[node]):
                        tail = successor[tail]
                        best_path[best_length] = tail
                        best_length += 1
                    break
            if node == destination:
                break
            sign = 1.0 if arrival < lower else -1.0
            remaining = max(lower-arrival, arrival-upper, 0.0)
            available = jump_limit - (length-1) - 2
            if available <= 0:
                break
            required = urgency * remaining / available
            best_edge = -1
            best_score = float('inf')
            for edge in range(indptr[node], indptr[node+1]):
                nxt = targets[edge]
                if visited[nxt] or nxt == destination or suffix_hops[nxt] > jump_limit:
                    continue
                progress = sign * means[edge]
                if progress <= max(1e-8, required):
                    continue
                score = variances[edge] / progress
                score *= 1.0 + noise * rng.random()
                if score < best_score:
                    best_score = score
                    best_edge = edge
            if best_edge < 0:
                break
            cost += variances[best_edge]
            clock += means[best_edge]
            node = targets[best_edge]
            path[length] = node
            length += 1
    return best_path[:best_length], best_cost
