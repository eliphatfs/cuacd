"""MCTS-based cutting plane search.

Translates CoACD/src/mcts.cpp to Python.
"""

import math
import numpy as np

from ._geometry import Plane, compute_bbox
from ._mesh import Mesh
from ._clip import clip
from ._cost import compute_rv, compute_total_rv

INF = float("inf")
MCTS_RANDOM_CUT = 1


def compute_axes_aligned_planes(bbox, mcts_nodes: int, shuffle: bool = False, rng=None):
    """Generate axis-aligned candidate cutting planes within bbox."""
    planes = []
    eps = 1e-6
    for axis in range(3):
        lo = bbox[2 * axis]
        hi = bbox[2 * axis + 1]
        interval = max(0.01, abs(hi - lo) / (mcts_nodes + 1))
        margin = max(0.015, interval)
        normal = [0.0, 0.0, 0.0]
        normal[axis] = 1.0
        i = lo + margin
        while i <= hi - margin + eps:
            planes.append(Plane(normal[0], normal[1], normal[2], -i))
            i += interval

    if shuffle and rng is not None:
        rng.shuffle(planes)
    return planes


def _best_rv_plane(mesh: Mesh, planes, rv_k: float):
    """Find plane with minimum max-Rv cost among candidates."""
    best_cost = INF
    best_plane = None
    for pl in planes:
        ok, pos, neg = clip(mesh, pl)
        if not ok or pos is None or neg is None:
            continue
        if pos.n_vertices == 0 or neg.n_vertices == 0:
            continue
        pos_ch = pos.convex_hull()
        neg_ch = neg.convex_hull()
        h = compute_total_rv(pos, pos_ch, neg, neg_ch, rv_k)
        if h < best_cost:
            best_cost = h
            best_plane = pl
    return best_plane, best_cost


class Part:
    __slots__ = ("mesh", "available_moves", "next_choice")

    def __init__(self, mesh: Mesh, mcts_nodes: int, rng):
        self.mesh = mesh
        self.available_moves = compute_axes_aligned_planes(mesh.bbox, mcts_nodes, shuffle=True, rng=rng)
        self.next_choice = 0

    def get_one_move(self):
        if self.next_choice >= len(self.available_moves):
            return None
        p = self.available_moves[self.next_choice]
        self.next_choice += 1
        return p


class State:
    def __init__(self, parts, costs, initial_mesh: Mesh, params, rng):
        self.parts = parts
        self.costs = costs
        self.initial_mesh = initial_mesh
        self.params = params
        self.rng = rng
        self.current_round = 0
        self.current_cost = 0.0
        self.current_value = None  # (Plane, int)
        self.worst_part_idx = 0

        self.ori_mesh_area = initial_mesh.area()
        self.ori_mesh_volume = initial_mesh.volume()
        ch = initial_mesh.convex_hull()
        self.ori_meshCH_volume = ch.volume()

    def is_terminal(self):
        if self.current_round >= self.params["mcts_max_depth"]:
            return True
        if self.worst_part_idx >= len(self.parts):
            return True
        if len(self.parts[self.worst_part_idx].available_moves) == 0:
            return True
        return False

    def compute_reward(self):
        """Find worst part and return max cost."""
        h_max = 0.0
        worst = 0
        for i, h in enumerate(self.costs):
            if h > h_max:
                h_max = h
                worst = i
        self.worst_part_idx = worst
        return h_max

    def get_next_state_with_random_choice(self):
        """Expand by cutting the worst part with next available move."""
        plane = self.parts[self.worst_part_idx].get_one_move()
        if plane is None:
            return self._failed_state()

        ok, pos, neg = clip(self.parts[self.worst_part_idx].mesh, plane)
        if not ok:
            return self._failed_state()

        new_parts = []
        new_costs = []
        for i in range(len(self.parts)):
            if i != self.worst_part_idx:
                new_parts.append(self.parts[i])
                new_costs.append(self.costs[i])

        pos_ch = pos.convex_hull()
        neg_ch = neg.convex_hull()
        cost_pos = compute_rv(pos, pos_ch, self.params["rv_k"])
        cost_neg = compute_rv(neg, neg_ch, self.params["rv_k"])

        new_parts.append(Part(pos, self.params["mcts_nodes"], self.rng))
        new_parts.append(Part(neg, self.params["mcts_nodes"], self.rng))
        new_costs.append(cost_pos)
        new_costs.append(cost_neg)

        ns = State(new_parts, new_costs, self.initial_mesh, self.params, self.rng)
        ns.current_value = (plane, self.worst_part_idx)
        reward = ns.compute_reward()
        ns.current_cost = self.current_cost + reward
        ns.current_round = self.current_round + 1
        return ns

    def _failed_state(self):
        ns = State(self.parts[:], self.costs[:], self.initial_mesh, self.params, self.rng)
        ns.current_cost = INF
        ns.current_round = self.params["mcts_max_depth"]
        return ns


class Node:
    __slots__ = ("parent", "children", "visit_times", "quality_value", "state", "params")

    def __init__(self, params):
        self.parent = None
        self.children = []
        self.visit_times = 0
        self.quality_value = INF
        self.state = None
        self.params = params

    def is_all_expand(self):
        s = self.state
        max_expand = len(s.parts[s.worst_part_idx].available_moves)
        return len(self.children) >= max_expand


def _expand(node):
    new_state = node.state.get_next_state_with_random_choice()
    child = Node(node.params)
    child.state = new_state
    child.parent = node
    node.children.append(child)
    return child


def _best_child(node, is_exploration, initial_cost=0.1):
    best_score = INF
    best_node = None
    for child in node.children:
        if is_exploration:
            C = initial_cost / math.sqrt(2.0)
        else:
            C = 0.0
        left = child.quality_value
        if child.visit_times == 0:
            score = -INF  # Ensure unvisited nodes are selected
        else:
            right = 2.0 * math.log(node.visit_times) / child.visit_times
            score = left - C * math.sqrt(right)
        if score < best_score:
            best_score = score
            best_node = child
    return best_node


def _tree_policy(node, initial_cost):
    while not node.state.is_terminal():
        if node.is_all_expand():
            node = _best_child(node, True, initial_cost)
            if node is None:
                break
        else:
            return _expand(node)
    return node


def _default_policy(node, params, rng):
    """Rollout: repeatedly cut worst part with Rv-only cost."""
    current_parts = [p for p in node.state.parts]
    current_costs = list(node.state.costs)
    current_round = node.state.current_round
    worst_idx = node.state.worst_part_idx
    current_cost = node.state.current_cost
    current_path = []

    while current_round < params["mcts_max_depth"]:
        if worst_idx >= len(current_parts):
            break
        worst_mesh = current_parts[worst_idx].mesh
        planes = compute_axes_aligned_planes(worst_mesh.bbox, MCTS_RANDOM_CUT)
        if not planes:
            break
        best_plane, best_cost = _best_rv_plane(worst_mesh, planes, params["rv_k"])
        if best_plane is None:
            break

        ok, pos, neg = clip(worst_mesh, best_plane)
        if not ok:
            break
        current_path.append(best_plane)

        new_parts = []
        new_costs = []
        for i in range(len(current_parts)):
            if i != worst_idx:
                new_parts.append(current_parts[i])
                new_costs.append(current_costs[i])

        pos_ch = pos.convex_hull()
        neg_ch = neg.convex_hull()
        cost_pos = compute_rv(pos, pos_ch, params["rv_k"])
        cost_neg = compute_rv(neg, neg_ch, params["rv_k"])

        new_parts.append(Part(pos, params["mcts_nodes"], rng))
        new_parts.append(Part(neg, params["mcts_nodes"], rng))
        new_costs.append(cost_pos)
        new_costs.append(cost_neg)

        current_parts = new_parts
        current_costs = new_costs

        # Find worst
        h_max = 0.0
        worst_idx = 0
        for i, h in enumerate(current_costs):
            if h > h_max:
                h_max = h
                worst_idx = i
        current_cost += h_max
        current_round += 1

    return current_cost / params["mcts_max_depth"], current_path


def _backup(node, reward, current_path, best_path_holder):
    """Backpropagate reward up the tree."""
    # Build reversed path
    tmp_path = list(reversed(current_path))

    n = node
    while n is not None:
        if n.state.current_round == 0 and n.quality_value > reward:
            best_path_holder[0] = list(tmp_path)
        if n.state.current_value is not None:
            tmp_path.append(n.state.current_value[0])
        n.visit_times += 1
        n.quality_value = min(n.quality_value, reward)
        n = n.parent


def monte_carlo_tree_search(mesh: Mesh, params: dict, rng):
    """Run MCTS to find best cutting plane.

    Returns (best_plane, best_path, best_quality) or (None, [], INF) if no valid cut.
    """
    root_part = Part(mesh, params["mcts_nodes"], rng)
    root_state = State([root_part], [INF], mesh, params, rng)
    root = Node(params)
    root.state = root_state

    ch = mesh.convex_hull()
    initial_cost = compute_rv(mesh, ch, params["rv_k"]) / params["mcts_max_depth"]
    best_path_holder = [[]]  # mutable container

    for _ in range(params["mcts_iteration"]):
        expand_node = _tree_policy(root, initial_cost)
        if expand_node is None:
            break
        reward, current_path = _default_policy(expand_node, params, rng)
        _backup(expand_node, reward, current_path, best_path_holder)

    best_child_node = _best_child(root, False)
    if best_child_node is None or best_child_node.state.current_value is None:
        return None, [], INF

    plane = best_child_node.state.current_value[0]
    return plane, best_path_holder[0], best_child_node.quality_value


# ---------------------------------------------------------------------------
# Ternary refinement
# ---------------------------------------------------------------------------

def _clip_by_path(mesh: Mesh, first_plane: Plane, best_path: list, rv_k: float):
    """Clip mesh by first_plane, then recurse on worst part along path.

    Returns (success, cost).
    """
    ok, pos, neg = clip(mesh, first_plane)
    if not ok:
        return False, INF

    pos_ch = pos.convex_hull()
    neg_ch = neg.convex_hull()
    pos_cost = compute_rv(pos, pos_ch, rv_k)
    neg_cost = compute_rv(neg, neg_ch, rv_k)

    scores = [pos_cost, neg_cost]
    parts = [pos, neg]
    worst_idx = 0 if pos_cost > neg_cost else 1
    final_cost = max(pos_cost, neg_cost)

    N = len(best_path)
    for i in range(1, N):
        plane_i = best_path[N - 1 - i]
        ok, p, n = clip(parts[worst_idx], plane_i)
        if not ok:
            return False, INF
        p_ch = p.convex_hull()
        n_ch = n.convex_hull()
        pc = compute_rv(p, p_ch, rv_k)
        nc = compute_rv(n, n_ch, rv_k)

        new_scores = []
        new_parts = []
        for j in range(len(parts)):
            if j != worst_idx:
                new_scores.append(scores[j])
                new_parts.append(parts[j])
        new_scores.extend([pc, nc])
        new_parts.extend([p, n])
        scores = new_scores
        parts = new_parts

        max_cost = scores[0]
        worst_idx = 0
        for j in range(1, len(scores)):
            if scores[j] > final_cost:
                worst_idx = j
                max_cost = scores[j]
        final_cost += max_cost

    if N > 0:
        final_cost /= N
    return True, final_cost


def ternary_refine(mesh: Mesh, best_plane: Plane, best_path: list,
                   best_cost: float, params: dict):
    """Ternary search refinement of plane position along its axis."""
    bbox = mesh.bbox
    min_itv = 0.01
    thres = 10
    epsilon = 0.0001
    rv_k = params["rv_k"]
    mcts_nodes = params["mcts_nodes"]

    result_plane = best_plane
    best_within_three = INF

    for axis in range(3):
        normal = [0.0, 0.0, 0.0]
        normal[axis] = 1.0
        coeff = [best_plane.a, best_plane.b, best_plane.c][axis]

        # Only refine the axis that matches the best plane
        if abs(coeff - 1.0) >= 1e-4:
            continue

        lo = bbox[2 * axis]
        hi = bbox[2 * axis + 1]
        interval = max(0.01, abs(hi - lo) / (mcts_nodes + 1))

        left = max(lo + min_itv, -best_plane.d - interval)
        right = min(hi - min_itv, -best_plane.d + interval)

        if left > right:
            return result_plane

        it = 0
        res = 0.0
        while left + epsilon < right and it < thres:
            it += 1
            margin = (right - left) / 3.0
            m1 = left + margin
            m2 = m1 + margin
            p1 = Plane(normal[0], normal[1], normal[2], -m1)
            p2 = Plane(normal[0], normal[1], normal[2], -m2)

            _, e1 = _clip_by_path(mesh, p1, best_path, rv_k)
            _, e2 = _clip_by_path(mesh, p2, best_path, rv_k)

            if e1 < e2:
                right = m2
                res = m1
            else:
                left = m1
                res = m2

        tp = Plane(normal[0], normal[1], normal[2], -res)
        _, hmin = _clip_by_path(mesh, tp, best_path, rv_k)

        if hmin < best_cost:
            result_plane = tp

    return result_plane
