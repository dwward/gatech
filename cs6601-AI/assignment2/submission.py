# coding=utf-8
"""
This file is your main submission that will be graded against. Only copy-paste
code on the relevant classes included here. Do not add any classes or functions
to this file that are not part of the classes that we want.

References:
Priority Queue Implementation Notes
(https://docs.python.org/3/library/heapq.html#priority-queue-implementation-notes)
"""

import heapq
import os
import pickle
import math
import itertools

from explorable_graph import ExplorableGraph


class PriorityQueue(object):
    """
    A queue structure where each element is served in order of priority.

    Elements in the queue are popped based on the priority with higher priority
    elements being served before lower priority elements.  If two elements have
    the same priority, they will be served in the order they were added to the
    queue.

    Traditionally priority queues are implemented with heaps, but there are any
    number of implementation options.

    (Hint: take a look at the module heapq)

    Attributes:
        queue (list): Nodes added to the priority queue.
    """

    def __init__(self):
        """Initialize a new Priority Queue."""
        self.counter = itertools.count()
        self.queue = []

    def pop(self):
        """
        Pop top priority node from queue.

        Returns:
            The node with the highest priority.
        """
        popped = heapq.heappop(self.queue)
        return popped[0], popped[2]

    def remove(self, node_id):
        """
        Remove a node from the queue.

        This is a hint, you might require this in ucs,
        however, if you choose not to use it, you are free to
        define your own method and not use it.

        Args:
            node_id (int): Index of node in queue.
        """
        del self.queue[node_id]
        heapq.heapify(self.queue)

    def __iter__(self):
        """Queue iterator."""

        return iter(sorted(self.queue))

    def __str__(self):
        """Priority Queue to string."""

        return 'PQ:%s' % self.queue

    def append(self, node):
        """
        Append a node to the queue.

        Args:
            node: Comparable Object to be added to the priority queue.
        """
        priority = node[0]
        task = node[1]
        count = next(self.counter)
        entry = (priority, count, task)
        heapq.heappush(self.queue, entry)

    def __contains__(self, key):
        """
        Containment Check operator for 'in'

        Args:
            key: The key to check for in the queue.

        Returns:
            True if key is found in queue, False otherwise.
        """

        return key in [n[-1] for n in self.queue]

    def __eq__(self, other):
        """
        Compare this Priority Queue with another Priority Queue.

        Args:
            other (PriorityQueue): Priority Queue to compare against.

        Returns:
            True if the two priority queues are equivalent.
        """

        return self.queue == other.queue

    def size(self):
        """
        Get the current size of the queue.
        Returns:
            Integer of number of items in queue.
        """

        return len(self.queue)

    def clear(self):
        """Reset queue to empty (no nodes)."""

        self.queue = []

    def top(self):
        """
        Get the top item in the queue.

        Returns:
            The first item stored in teh queue.
        """

        return self.queue[0]


# ------------------------------------ end priority queue


def breadth_first_search(graph, start, goal):
    """
    Warm-up exercise: Implement breadth-first-search.  See README.md for exercise description.
    Args:
        graph (ExplorableGraph): Undirected graph to search.
        start (str): Key for the start node.
        goal (str): Key for the end node.
    Returns:
        The best path as a list from the start and goal nodes (including both).
    """
    """
        - Adds nodes in alphabetical order
        - Terminating conditions "different" linked
    """

    if start == goal:
        return []

    # Priority queue (priority, counter={provided}, [path])
    frontier = PriorityQueue()
    frontier.append((0, [start]))
    explored = []

    while True:
        if frontier.size() == 0:
            raise Exception("Frontier can't be empty")
        path = frontier.pop()[1]  # Get next path in queue
        s = path[-1]  # Gets node to expand from
        explored.append(s)

        if s == goal:
            return path

        neighbors = sorted(graph[s])
        for a in neighbors:
            # f[-1][-1], last tuple in frontier, last item in tuple
            if a not in explored and a not in [f[-1][-1] for f in frontier]:
                tmp_path = list(path)
                tmp_path.append(a)
                if a == goal:
                    return tmp_path
                frontier.append((0, tmp_path))


def path_weight(graph, path):
    cost = 0
    try:
        for i in range(0, len(path) - 1):
            cost = cost + graph.get_edge_weight(path[i], path[i + 1])
    except Exception:
        return 0
    return cost


def cheaper_path_in_frontier(frontier, item):
    if item in frontier:
        return True
    item_cost = item[0]
    item_path = item[1]
    edge_label = item_path[-1]
    enum_front = list(enumerate(frontier.queue))
    enum_front = [(x[0], x[1][-1][-1]) for x in enum_front]  # index, frontier label
    for i in enum_front:
        curr_path_cost = frontier.queue[i[0]][0]
        if i[1] == edge_label and curr_path_cost < item_cost:
            return True
    return False


def prune_mu_alternates(frontier, item, h=0):
    item_cost = item[0]
    item_path = item[1]
    edge_label = item_path[-1]
    enum_front = list(enumerate(frontier.queue))
    enum_front = [(x[0], x[1][-1][-1]) for x in enum_front]  # index, frontier label
    alternates = [i[0] for i in enum_front if i[1] == edge_label]
    alternates_pruned = False
    if alternates:
        # if len(alternates) > 2:
        #     raise Exception("Apparently multiple paths with this on frontier??")
        index = alternates[0]
        if frontier.queue[index][0] > item_cost:
            frontier.remove(index)
            frontier.append((item_cost, item_path))
            alternates_pruned = True
    return alternates_pruned


def prune_alternates(frontier, item, h=0):
    item_cost = item[0]
    item_path = item[1]
    edge_label = item_path[-1]
    enum_front = list(enumerate(frontier.queue))
    enum_front = [(x[0], x[1][-1][-1]) for x in enum_front]  # index, frontier label
    alternates = [i[0] for i in enum_front if i[1] == edge_label]
    alternates_pruned = False
    if alternates:
        # if len(alternates) > 2:
        #     raise Exception("Apparently multiple paths with this on frontier??")
        index = alternates[0]
        if frontier.queue[index][0] > item_cost:
            frontier.remove(index)
            frontier.append((item_cost, item_path))
            alternates_pruned = True
    return alternates_pruned


def uniform_cost_search(graph, start, goal):
    if start == goal:
        return []

    # Path cost is now the 'priority'
    frontier = PriorityQueue()
    frontier.append((0, [start]))
    explored = []

    while True:
        if frontier.size() == 0:
            raise Exception("Frontier can't be empty - Start: {} Goal: {}".format(start, goal))
        path_cost, path = frontier.pop()  # Get next path in queue <(priority, [path])>
        s = path[-1]  # Gets node to expand from
        explored.append(s)

        if s == goal:
            return path

        if frontier.size() > 0:
            if prune_alternates(frontier, (path_cost, path)):
                continue

        neighbors = graph[s]
        for a in neighbors:
            if a not in explored:  # and a not in [f[-1][-1] for f in frontier]:
                tmp_path = list(path)
                tmp_path.append(a)
                # if a == goal:
                #     return tmp_path
                cost = path_cost + graph.get_edge_weight(s, a)
                if not cheaper_path_in_frontier(frontier, (cost, tmp_path)):
                    frontier.append((cost, tmp_path))


def null_heuristic(graph, v, goal):
    """
    Null heuristic used as a base line.

    Args:
        graph (ExplorableGraph): Undirected graph to search.
        v (str): Key for the node to calculate from.
        goal (str): Key for the end node to calculate to.

    Returns:
        0
    """

    return 0


def euclidean_dist_heuristic(graph, v, goal):
    """
    Warm-up exercise: Implement the euclidean distance heuristic.

    See README.md for exercise description.

    Args:
        graph (ExplorableGraph): Undirected graph to search.
        v (str): Key for the node to calculate from.
        goal (str): Key for the end node to calculate to.

    Returns:
        Euclidean distance between `v` node and `goal` node
    """
    x1, y1 = graph.nodes[v]['pos']
    x2, y2 = graph.nodes[goal]['pos']
    return math.sqrt((x2 - x1) ** 2 + (y2 - y1) ** 2)


def calculate_path_weight(graph, path):
    if not path:
        return float('inf')
    cost = 0
    if len(path) > 1:
        for n in range(0, len(path) - 1):
            cost = cost + graph.get_edge_weight(path[n], path[n + 1])
    return cost


def a_star(graph, start, goal, heuristic=euclidean_dist_heuristic):
    """
    Warm-up exercise: Implement A* algorithm. See README.md for exercise description.
    Args:
        graph (ExplorableGraph): Undirected graph to search.
        start (str): Key for the start node.
        goal (str): Key for the end node.
        heuristic: Function to determine distance heuristic.
            Default: euclidean_dist_heuristic.
    Returns:
        The best path as a list from the start and goal nodes (including both).
    """

    if start == goal:
        return []

    # Path cost is now the 'priority'
    frontier = PriorityQueue()
    frontier.append((0, [start]))
    explored = []

    while True:
        if frontier.size() == 0:
            raise Exception("Frontier can't be empty - Start: {} Goal: {}".format(start, goal))
        path_cost, path = frontier.pop()  # Get next path in queue <(priority, [path])>
        s = path[-1]  # Gets node to expand from
        explored.append(s)

        if s == goal:
            return path

        if frontier.size() > 0:
            if prune_alternates(frontier, (path_cost, path)):
                continue

        neighbors = graph[s]

        for a in neighbors:
            if a not in explored:  # and a not in [f[-1][-1] for f in frontier]:
                tmp_path = list(path)
                tmp_path.append(a)
                # if a == goal:
                #     return tmp_path

                h = euclidean_dist_heuristic(graph, a, goal)
                cost = calculate_path_weight(graph, tmp_path) + h
                if not cheaper_path_in_frontier(frontier, (cost, tmp_path)):
                    frontier.append((cost, tmp_path))


def bidirectional_ucs(graph, start, goal):
    if start == goal:
        return []

    # Path cost is now the 'priority'
    forward_queue = PriorityQueue()
    reverse_queue = PriorityQueue()
    forward_queue.append((0, [start]))
    reverse_queue.append((0, [goal]))
    forward_explored = []
    reverse_explored = []
    other_queue = None
    other_explored = None
    mu_path = []
    mu_cost = float('inf')

    d = calculate_path_weight  # graph, path

    direction = -1  # 1 forward, -1 reverse
    while True:
        direction = direction * -1
        topf = d(graph, forward_queue.queue[0][2])
        topr = d(graph, reverse_queue.queue[0][2])
        if topf + topr >= mu_cost:
            return mu_path

        # Alternate which queue worked on
        if direction == 1:
            frontier = forward_queue
            explored = forward_explored
            other_queue = reverse_queue
            other_explored = reverse_explored
        else:
            frontier = reverse_queue
            explored = reverse_explored
            other_queue = forward_queue
            other_explored = forward_explored

        # Expand nodes
        if frontier.size() == 0:
            raise Exception("Frontier can't be empty - Start: {} Goal: {}".format(start, goal))
        path_cost, path = frontier.pop()  # Get next path in queue <(priority, [path])>
        s = path[-1]
        explored.append(s)

        # Check for match before expanding node.....
        if s in other_explored:
            for i in range(0, other_queue.size()):
                tmp_other_path = other_queue.queue[i][2]
                if tmp_other_path[-1] == s:
                    joined_path = tmp_path[:-1] + tmp_other_path[::-1]
                    joined_path_cost = path_weight(graph, joined_path)
                    if joined_path_cost < mu_cost:
                        mu_cost = joined_path_cost
                        mu_path = joined_path
                        if joined_path[0] == goal:
                            mu_path = joined_path[::-1]

        # if s == goal:
        #     return path

        # Remove duplicate paths from v...w if their cost is worse
        if frontier.size() > 0:
            if prune_alternates(frontier, (path_cost, path)):
                continue

        neighbors = graph[s]
        for a in neighbors:
            if a not in explored:
                tmp_path = list(path)
                tmp_path.append(a)

                cost = path_cost + graph.get_edge_weight(s, a)

                # peek in the other frontier, and see if paths connect and update mu
                for i in range(0, other_queue.size()):
                    tmp_other_path = other_queue.queue[i][2]
                    if tmp_other_path[-1] == a:
                        joined_path = tmp_path[:-1] + tmp_other_path[::-1]
                        joined_path_cost = path_weight(graph, joined_path)
                        if joined_path_cost < mu_cost:
                            mu_cost = joined_path_cost
                            mu_path = joined_path
                            if joined_path[0] == goal:
                                mu_path = joined_path[::-1]

                if not cheaper_path_in_frontier(frontier, (cost, tmp_path)):
                    frontier.append((cost, tmp_path))


def bidirectional_a_star(graph, start, goal,
                         heuristic=euclidean_dist_heuristic):
    """
    Exercise 2: Bidirectional A*.

    See README.md for exercise description.

    Args:
        graph (ExplorableGraph): Undirected graph to search.
        start (str): Key for the start node.
        goal (str): Key for the end node.
        heuristic: Function to determine distance heuristic.
            Default: euclidean_dist_heuristic.

    Returns:
        The best path as a list from the start and goal nodes (including both).
    """
    if start == goal:
        return []

    # Path cost is now the 'priority'
    forward_queue = PriorityQueue()
    reverse_queue = PriorityQueue()
    forward_queue.append((0, [start]))
    reverse_queue.append((0, [goal]))
    forward_explored = []
    reverse_explored = []
    other_queue = None
    other_explored = None
    mu_path = []
    mu_cost = float('inf')
    distances = {}

    d = calculate_path_weight  # graph, path

    direction = -1  # 1 forward, -1 reverse
    while forward_queue.queue and reverse_queue.queue:
        direction = direction * -1
        topf = d(graph, forward_queue.queue[0][2])
        topr = d(graph, reverse_queue.queue[0][2])
        if topf + topr >= mu_cost:
            return mu_path

        # Alternate which queue worked on
        if direction == 1:
            frontier = forward_queue
            explored = forward_explored
            other_queue = reverse_queue
            other_explored = reverse_explored
        else:
            frontier = reverse_queue
            explored = reverse_explored
            other_queue = forward_queue
            other_explored = forward_explored

        # Expand nodes
        if frontier.size() == 0:
            raise Exception("Frontier can't be empty - Start: {} Goal: {}".format(start, goal))
        path_cost, path = frontier.pop()  # Get next path in queue <(priority, [path])>
        s = path[-1]
        explored.append(s)

        # if s == goal:
        #     return path

        # Remove duplicate paths from v...w if their cost is worse
        if frontier.size() > 0:
            if prune_alternates(frontier, (path_cost, path)):
                continue

        # if set(path).issubset(set(mu_path)):
        #     neighbors = []
        # else:
        neighbors = graph[s]
        for a in neighbors:
            if a not in explored:
                tmp_path = list(path)
                tmp_path.append(a)

                # cost = path_cost + graph.get_edge_weight(s, a)
                h = 0
                if direction == 1:
                    if (a, goal) in distances:
                        h = distances[(a, goal)]
                    else:
                        h = euclidean_dist_heuristic(graph, a, goal)
                        distances[(a, goal)] = h
                else:
                    if (a, start) in distances:
                        h = distances[(a, start)]
                    else:
                        h = euclidean_dist_heuristic(graph, a, start)
                        distances[(a, start)] = h
                cost = calculate_path_weight(graph, tmp_path) + h

                # peek in the other frontier, and see if paths connect and update mu
                for i in range(0, other_queue.size()):
                    tmp_other_path = other_queue.queue[i][2]
                    if tmp_other_path[-1] == a:
                        joined_path = tmp_path[:-1] + tmp_other_path[::-1]
                        joined_path_cost = path_weight(graph, joined_path)
                        if joined_path_cost < mu_cost:
                            mu_cost = joined_path_cost
                            mu_path = joined_path
                            if joined_path[0] == goal:
                                mu_path = joined_path[::-1]
                                explored.append(a)
                                # other_explored.append(a)

                if not cheaper_path_in_frontier(frontier, (cost, tmp_path)):
                    frontier.append((cost, tmp_path))


# def tridirectional_search(graph, goals):
#
#     ab_path = bidirectional_ucs(graph, goals[0], goals[1])
#     bc_path = bidirectional_ucs(graph, goals[1], goals[2])
#     ca_path = bidirectional_ucs(graph, goals[2], goals[0])
#
#     mu_segments = dict()
#     mu_segments[goals[0]] = ab_path
#     mu_segments[goals[1]] = bc_path
#     mu_segments[goals[2]] = ca_path
#
#     mu_path, mu_score = update_mu(graph, mu_segments, None, goals)
#     print(mu_path, mu_score)
#     return mu_path

class Mu(object):
    def __init__(self, graph, goals):
        """Initialize a new Priority Queue."""
        self.mu_segments = dict()
        self.mu_segment_cost = dict()
        self.goals = goals
        self.graph = graph

        self.mu_segments[goals[0]] = []
        self.mu_segments[goals[1]] = []
        self.mu_segments[goals[2]] = []

        self.mu_segment_cost[goals[0]] = float('inf')
        self.mu_segment_cost[goals[1]] = float('inf')
        self.mu_segment_cost[goals[2]] = float('inf')

    def has_all_segments(self):
        if self.mu_segments[self.goals[0]] and self.mu_segments[self.goals[1]] and self.mu_segments[self.goals[2]]:
            return True
        return False

    def get_all_segments(self):
        return self.mu_segments[self.goals[0]], self.mu_segments[self.goals[1]], self.mu_segments[self.goals[2]]

    def get_path_and_score(self):

        mu_path = []
        mu_score = float('inf')

        mu_seg0 = self.mu_segments[self.goals[0]]
        mu_seg1 = self.mu_segments[self.goals[1]]
        mu_seg2 = self.mu_segments[self.goals[2]]

        path_count = [len(mu_seg0) > 0, len(mu_seg1) > 0, len(mu_seg2) > 0].count(
            True)

        if path_count >= 2:
            mu_paths = PriorityQueue()
            mu_paths.append((calculate_path_weight(self.graph, mu_seg0), mu_seg0, 0))
            mu_paths.append((calculate_path_weight(self.graph, mu_seg1), mu_seg1, 1))
            mu_paths.append((calculate_path_weight(self.graph, mu_seg2), mu_seg2, 2))

            result1 = mu_paths.pop()
            result2 = mu_paths.pop()

            # [a,b,c] + [c,d,e]
            if result1[1][-1] == result2[1][0]:
                mu_path = intersect_paths(result1[1], result2[1])
            # [c,d,e] + [a,b,c]???
            else:
                mu_path = intersect_paths(result2[1], result1[1])

            # score2 = calculate_path_weight(self.graph, result1[1])
            # score1 = calculate_path_weight(self.graph, result2[1])

            mu_score = calculate_path_weight(self.graph, mu_path)

            if not contains_all_goals(mu_path, self.goals):
                mu_score = float('inf')

        return mu_path, mu_score

    def get_segment_by_name(self, name):
        return self.mu_segments[name]

    def get_segment_score_by_idx(self, index):
        return self.mu_segment_cost[self.goals[index]]

    def get_segment_score_by_name(self, name):
        return self.mu_segment_cost[name]

    def update_mu_by_idx(self, path, goal_index):
        self.update_mu_by_name(path, self.goals[goal_index])

    def update_mu_by_name(self, path, goal_name):
        # Update it if it's an improvement!
        weight = calculate_path_weight(self.graph, path)
        if weight < self.mu_segment_cost[goal_name]:
            self.mu_segments[goal_name] = path
            self.mu_segment_cost[goal_name] = weight

    # returns target goal of node g
    def target_goal(self, g):
        index = self.goals.index(g)
        next_index = (index + 1) % 3
        return self.goals[next_index]

    # returns goal of which given goal node is target goal
    def whos_goal_am_i(self, g):
        index = self.goals.index(g)
        next_index = (index + 2) % 3
        return self.goals[next_index]


def path_subseq(path1, path2):
    path1_len = len(path1)
    path2_len = len(path2)
    if len(path1) > len(path2):
        return False
    diff = path2_len - path1_len
    for i in range(0, diff + 1):
        subseq = path2[i:i + path1_len]
        if path1 == subseq:
            return True
    return False


def tridirectional_search(graph, goals):
    goal1 = goals[0]
    goal2 = goals[1]
    goal3 = goals[2]

    if goal1 == goal2 == goal3:
        return []

    # Path cost is now the 'priority'
    goal1_queue = PriorityQueue()
    goal2_queue = PriorityQueue()
    goal3_queue = PriorityQueue()
    goal1_queue.append((0, [goal1]))
    goal2_queue.append((0, [goal2]))
    goal3_queue.append((0, [goal3]))
    goal1_explored = []
    goal2_explored = []
    goal3_explored = []
    other_goal_name = None
    other_queue = None
    other_explored = None

    mu = Mu(graph, goals)
    d = calculate_path_weight  # graph, path
    direction = -1  # 0 = goal1, 1 = goal2, 2 = goal3
    frontier_wt = 0
    unintersected_mu_wt = 0
    critical_subset_index = 0

    while True:

        direction = (direction + 1) % 3
        top1, top2, top3 = 0, 0, 0
        if goal1_queue.queue:
            top1 = d(graph, goal1_queue.queue[0][2])
        if goal2_queue.queue:
            top2 = d(graph, goal2_queue.queue[0][2])
        if goal3_queue.queue:
            top3 = d(graph, goal3_queue.queue[0][2])

        frontier_wt = top1 + top2 + top3  # for debugging
        mu_path, mu_score = mu.get_path_and_score()

        scores = [float('inf') if not goal1_queue.queue else goal1_queue.queue[0][0],
                  float('inf') if not goal2_queue.queue else goal2_queue.queue[0][0],
                  float('inf') if not goal3_queue.queue else goal3_queue.queue[0][0]]
        min_score = min(scores)
        min_index = scores.index(min_score)
        direction = min_index

        # ALTERNATES FRONTIER QUEUES
        if direction == 0:
            frontier = goal1_queue
            explored = goal1_explored
            other_queue = goal2_queue
            other_explored = goal2_explored
            other_goal_name = goal2
        elif direction == 1:
            frontier = goal2_queue
            explored = goal2_explored
            other_queue = goal3_queue
            other_explored = goal3_explored
            other_goal_name = goal3
        elif direction == 2:
            frontier = goal3_queue
            explored = goal3_explored
            other_queue = goal1_queue
            other_explored = goal1_explored
            other_goal_name = goal1

        # TERMINATE, NEXT DOOR NEIGHBORS SPECIAL CASE
        if mu_score < float('inf') and mu.has_all_segments():
            if len(mu_path) == 3:
                return mu_path

        front1_exceeds = top1 >= mu.get_segment_score_by_idx(0)
        front2_exceeds = top2 >= mu.get_segment_score_by_idx(1)
        front3_exceeds = top3 >= mu.get_segment_score_by_idx(2)

        if (front1_exceeds and front2_exceeds) or (front2_exceeds and front3_exceeds) or (
                front1_exceeds and front3_exceeds) or (front1_exceeds and front2_exceeds and front3_exceeds):
            return mu_path

        if frontier_wt >= mu_score and mu_score != float('inf'):
            return mu_path

        # TERMINATE, QUEUE ALL GO EMPTY
        if frontier.size() == 0:
            if not goal3_queue.queue and not goal2_queue.queue and not goal1_queue.queue:
                print("Explored node count: ", sum(graph.explored_nodes.values()))
                return mu_path
            continue
        # raise Exception("Frontier can't be empty - Goals: {}".format(goals))

        path_cost, path = frontier.pop()  # Get next path in queue <(priority, [path])>
        s = path[-1]
        explored.append(s)

        # Optimization: Remove duplicate paths from v...w if their cost is worse
        #
        if frontier.size() > 0:
            if prune_alternates(frontier, (path_cost, path)):
                continue

        # Don't process if already processed by mu
        mu_segs = mu.get_all_segments()
        is_subset = False
        for seg in mu_segs:
            if len(seg) >= 2 and len(path) == 2 and path_subseq(seg, path):
                is_subset = True
                break
        if is_subset:
            continue

        # Optimization:  Subsequence pruning, very finicky optimization though
        # Seems to bomb out on nodes with 3 or more in path connections (back tracking)
        #
        mu_segs = mu.get_all_segments()
        is_subset = False
        # for p in mu_segs:  # maybe we should do only in our own path??
        p = mu.get_segment_by_name(path[0])
        if path_subseq(path, p[::-1]):
            is_subset = True
        if is_subset:
            continue

        # Optimization: If path is more expensive than what we already have in mu, then don't expand
        #
        new_path_shorter_than_mu = calculate_path_weight(graph, path) < mu.get_segment_score_by_name(path[0])
        if new_path_shorter_than_mu:
            neighbors = graph[s]
        else:
            continue  # we have no neighbors so skip!

        for curr_neighbor in neighbors:

            # Optimization:  Skip if we are overlapping a goal node that's not ours (REVERSED),
            #
            if curr_neighbor in goals and not curr_neighbor == other_goal_name:

                neighbor_goal_target = mu.target_goal(curr_neighbor)
                if neighbor_goal_target == path[0]:
                    tmp_path = path.copy()
                    tmp_path.append(curr_neighbor)
                    tmp_path.reverse()

                    # Update MU value
                    mu.update_mu_by_name(tmp_path, tmp_path[0])

            elif curr_neighbor not in explored:
                tmp_path = list(path)
                tmp_path.append(curr_neighbor)
                cost = path_cost + graph.get_edge_weight(s, curr_neighbor)

                """
                   TRI-DIRECTIONAL SEARCH, THE FINAL FRONTIER
                """
                if not cheaper_path_in_frontier(frontier, (cost, tmp_path)):

                    # Optimization:  IF WE HAVE BEEN FOUND AS A GOAL, DON'T CONSIDER HEAVIER WEIGHTS
                    #
                    upstream_goal = mu.whos_goal_am_i(tmp_path[0])
                    upstream_mu = mu.get_segment_by_name(upstream_goal)
                    upstream_path_cost = mu.get_segment_score_by_name(upstream_goal)
                    if not upstream_mu or (upstream_mu[-1] != tmp_path and upstream_path_cost > cost):

                        tmp_mu_segment = mu.get_segment_by_name(tmp_path[0])
                        if tmp_mu_segment != tmp_path:
                            frontier.append((cost, tmp_path))

                # Optimization:  SEE IF OUR TMP PATH CAN CONNECT TO A MU SEGMENT
                #
                if mu.get_segment_by_name(other_goal_name) and tmp_path[-1] == \
                        mu.get_segment_by_name(other_goal_name)[-1]:
                    mu.update_mu_by_name(tmp_path, goals[direction])

                # Optimization:  SEE IF OUR PATH CAN CONNECT TO A MU SEGMENT
                #
                elif mu.get_segment_by_name(other_goal_name) and path[-1] == \
                        mu.get_segment_by_name(other_goal_name)[-1]:
                    mu.update_mu_by_name(path, goals[direction])

                # Optimization:  SEE IF TARGET GOAL IS OUR IMMEDIATE NEIGHBOR
                #
                elif other_goal_name == curr_neighbor:
                    mu.update_mu_by_name(tmp_path, goals[direction])

                else:

                    # Optimization:  aggressive frontier scanning for expanded paths (TMP_PATH)
                    #
                    queues = [goal1_queue, goal2_queue, goal3_queue]
                    for q in queues:
                        for i in range(0, q.size()):
                            tmp_other_path = q.queue[i][2]
                            if len(tmp_other_path) > 1:
                                # don't operate on our own queue
                                if tmp_other_path[0] == path[0]:
                                    break
                                # test queues can attach
                                if tmp_other_path[-1] == tmp_path[-1]:
                                    # my target goal
                                    if other_goal_name == tmp_other_path[0]:
                                        joined_path = intersect_paths(tmp_path, tmp_other_path)
                                        if joined_path:
                                            mu.update_mu_by_name(joined_path, joined_path[0])
                                    # i'm their target goal
                                    else:
                                        joined_path = intersect_paths(tmp_other_path, tmp_path)
                                        if joined_path:
                                            mu.update_mu_by_name(joined_path, joined_path[0])

                    # Optimization:  aggressive frontier scanning for unexpanded paths (PATH)
                    #
                    queues = [goal1_queue, goal2_queue, goal3_queue]
                    for q in queues:
                        for i in range(0, q.size()):
                            tmp_other_path = q.queue[i][2]
                            if len(tmp_other_path) > 1:
                                # don't operate on our own queue
                                if tmp_other_path[0] == path[0]:
                                    break
                                # test queues can attach
                                if tmp_other_path[-1] == path[-1]:
                                    # my target goal
                                    if other_goal_name == tmp_other_path[0]:
                                        joined_path = intersect_paths(path, tmp_other_path)
                                        if joined_path:
                                            mu.update_mu_by_name(joined_path, joined_path[0])
                                    # i'm their target goal
                                    else:
                                        joined_path = intersect_paths(tmp_other_path, path)
                                        if joined_path:
                                            mu.update_mu_by_name(joined_path, joined_path[0])


def contains_all_goals(lst, goals):
    count = [0] * len(goals)
    for i in range(0, len(goals)):
        if goals[i] in lst:
            count[i] = count[i] + 1
    if 0 in count:
        return False
    return True


def intersect_paths(path1, path2):
    a = path1
    b = path2

    # # Backtrack and cycling edge case
    # if len(list(set(path1).intersection(path2))) > 1:
    #     return []

    # Subseq edge case
    if path_subseq(a, b) or path_subseq(a[::-1], b):
        return b
    if path_subseq(b, a) or path_subseq(b[::-1], a):
        return a

    mu_path = []

    if a[-1] == b[0]:
        mu_path = a + b[1:]
    elif a[-1] == b[-1]:
        b = b[:-1]
        b.reverse()
        mu_path = a + b
    else:
        raise Exception("Trouble intersecting")
    return mu_path


def tridirectional_upgraded(graph, goals, heuristic=euclidean_dist_heuristic):
    goal1 = goals[0]
    goal2 = goals[1]
    goal3 = goals[2]

    if goal1 == goal2 == goal3:
        return []

    # Path cost is now the 'priority'
    goal1_queue = PriorityQueue()
    goal2_queue = PriorityQueue()
    goal3_queue = PriorityQueue()
    goal1_queue.append((0, [goal1]))
    goal2_queue.append((0, [goal2]))
    goal3_queue.append((0, [goal3]))
    goal1_explored = []
    goal2_explored = []
    goal3_explored = []
    other_goal_name = None
    other_queue = None
    other_explored = None

    mu = Mu(graph, goals)
    d = calculate_path_weight  # graph, path
    direction = -1  # 0 = goal1, 1 = goal2, 2 = goal3
    frontier_wt = 0
    unintersected_mu_wt = 0
    critical_subset_index = 0
    queues = [goal1_queue, goal2_queue, goal3_queue]

    while True:

        direction = (direction + 1) % 3
        top1, top2, top3 = 0, 0, 0
        if goal1_queue.queue:
            top1 = d(graph, goal1_queue.queue[0][2])
        if goal2_queue.queue:
            top2 = d(graph, goal2_queue.queue[0][2])
        if goal3_queue.queue:
            top3 = d(graph, goal3_queue.queue[0][2])

        frontier_wt = top1 + top2 + top3  # for debugging
        mu_path, mu_score = mu.get_path_and_score()

        scores = [float('inf') if not goal1_queue.queue else goal1_queue.queue[0][0],
                  float('inf') if not goal2_queue.queue else goal2_queue.queue[0][0],
                  float('inf') if not goal3_queue.queue else goal3_queue.queue[0][0]]
        min_score = min(scores)
        min_index = scores.index(min_score)
        direction = min_index

        # ALTERNATES FRONTIER QUEUES
        if direction == 0:
            frontier = goal1_queue
            explored = goal1_explored
            other_queue = goal2_queue
            other_explored = goal2_explored
            other_goal_name = goal2
        elif direction == 1:
            frontier = goal2_queue
            explored = goal2_explored
            other_queue = goal3_queue
            other_explored = goal3_explored
            other_goal_name = goal3
        elif direction == 2:
            frontier = goal3_queue
            explored = goal3_explored
            other_queue = goal1_queue
            other_explored = goal1_explored
            other_goal_name = goal1

        # TERMINATE, NEXT DOOR NEIGHBORS SPECIAL CASE
        if mu_score < float('inf') and mu.has_all_segments():
            if len(mu_path) == 3:
                return mu_path

        front1_exceeds = top1 >= mu.get_segment_score_by_idx(0)
        front2_exceeds = top2 >= mu.get_segment_score_by_idx(1)
        front3_exceeds = top3 >= mu.get_segment_score_by_idx(2)

        if (front1_exceeds and front2_exceeds) or (front2_exceeds and front3_exceeds) or (
                front1_exceeds and front3_exceeds) or (front1_exceeds and front2_exceeds and front3_exceeds):
            return mu_path

        if frontier_wt >= mu_score and mu_score != float('inf'):
            return mu_path

        # TERMINATE, QUEUE ALL GO EMPTY
        if frontier.size() == 0:
            if not goal3_queue.queue and not goal2_queue.queue and not goal1_queue.queue:
                print("Explored node count: ", sum(graph.explored_nodes.values()))
                return mu_path
            continue
        # raise Exception("Frontier can't be empty - Goals: {}".format(goals))

        #-----------------------
        # Add distances just before popping
        # And remove them afterwards... it's a little psychotic but I'm tired
        #-----------------------

        # Dynamically follow last explored node of target frontier
        last_explored = other_goal_name
        if other_explored:
            last_explored = other_explored[-1]
        for i in range(0, len(frontier.queue)):
            f = frontier.queue[i]
            dist_to_explored = heuristic(graph, f[2][-1], last_explored)
            distcost = calculate_path_weight(graph, f[2]) + dist_to_explored
            frontier.queue[i] = (distcost, f[1], f[2])
        heapq.heapify(frontier.queue)

        path_cost, path = frontier.pop()  # Get next path in queue <(priority, [path])>
        s = path[-1]
        explored.append(s)

        # Reset weights for next go around
        for i in range(0, len(frontier.queue)):
            f = frontier.queue[i]
            distcost = calculate_path_weight(graph, f[2])
            frontier.queue[i] = (distcost, f[1], f[2])
        heapq.heapify(frontier.queue)

        # # Optimization: Remove duplicate paths from v...w if their cost is worse
        # #
        # if frontier.size() > 0:
        #     if prune_alternates(frontier, (path_cost, path)):
        #         continue

        # # Don't process if already processed by mu
        # mu_segs = mu.get_all_segments()
        # is_subset = False
        # for seg in mu_segs:
        #     if len(seg) >= 2 and len(path) == 2 and path_subseq(seg, path):
        #         is_subset = True
        #         break
        # if is_subset:
        #     continue

        # Optimization:  Subsequence pruning, very finicky optimization though
        # Seems to bomb out on nodes with 3 or more in path connections (back tracking)
        #
        mu_segs = mu.get_all_segments()
        is_subset = False
        # for p in mu_segs:  # maybe we should do only in our own path??
        p = mu.get_segment_by_name(path[0])
        if path_subseq(path, p[::-1]):
            is_subset = True
        if is_subset:
            continue

        # Optimization: If path is more expensive than what we already have in mu, then don't expand
        #
        new_path_shorter_than_mu = calculate_path_weight(graph, path) < mu.get_segment_score_by_name(path[0])
        if new_path_shorter_than_mu:
            neighbors = graph[s]
        else:
            continue  # we have no neighbors so skip!

        for curr_neighbor in neighbors:

            # Optimization:  Skip if we are overlapping a goal node that's not ours (REVERSED),
            #
            if curr_neighbor in goals and not curr_neighbor == other_goal_name:

                neighbor_goal_target = mu.target_goal(curr_neighbor)
                if neighbor_goal_target == path[0]:
                    tmp_path = path.copy()
                    tmp_path.append(curr_neighbor)
                    tmp_path.reverse()

                    # Update MU value
                    mu.update_mu_by_name(tmp_path, tmp_path[0])

            elif curr_neighbor not in explored:
                tmp_path = list(path)
                tmp_path.append(curr_neighbor)

                # q1 = queues[(direction + 1) % 3]
                # f1 = goals[(direction + 1) % 3]
                # if q1.queue and len(q1.queue) >= 1:
                #     f1 = q1.queue[0][2][-1]
                #
                # q2 = queues[(direction + 2) % 3]
                # f2 = goals[(direction + 2) % 3]
                # if q2.queue and len(q2.queue) >= 1:
                #     f2 = q2.queue[0][2][-1]
                #
                # h1 = heuristic(graph, curr_neighbor, f1)
                # h2 = heuristic(graph, curr_neighbor, f2)
                # hval = h1 if h1 > h2 else h2

                last_explored = other_goal_name
                if other_explored:
                    last_explored = other_explored[-1]
                hval = heuristic(graph, goals[direction], last_explored)
                cost = path_cost + graph.get_edge_weight(s, curr_neighbor)


                """
                   TRI-DIRECTIONAL SEARCH, THE FINAL FRONTIER
                """
                #if not cheaper_path_in_frontier(frontier, (cost, tmp_path)):

                # Optimization:  IF WE HAVE BEEN FOUND AS A GOAL, DON'T CONSIDER HEAVIER WEIGHTS
                #
                upstream_goal = mu.whos_goal_am_i(tmp_path[0])
                upstream_mu = mu.get_segment_by_name(upstream_goal)
                upstream_path_cost = mu.get_segment_score_by_name(upstream_goal)
                if not upstream_mu or (upstream_mu[-1] != tmp_path and upstream_path_cost > cost):

                    tmp_mu_segment = mu.get_segment_by_name(tmp_path[0])
                    if tmp_mu_segment != tmp_path:
                        frontier.append((cost, tmp_path))

                # Optimization:  SEE IF OUR TMP PATH CAN CONNECT TO A MU SEGMENT
                #
                if mu.get_segment_by_name(other_goal_name) and tmp_path[-1] == \
                        mu.get_segment_by_name(other_goal_name)[-1]:
                    mu.update_mu_by_name(tmp_path, goals[direction])

                # Optimization:  SEE IF OUR PATH CAN CONNECT TO A MU SEGMENT
                #
                elif mu.get_segment_by_name(other_goal_name) and path[-1] == \
                        mu.get_segment_by_name(other_goal_name)[-1]:
                    mu.update_mu_by_name(path, goals[direction])

                # Optimization:  SEE IF TARGET GOAL IS OUR IMMEDIATE NEIGHBOR
                #
                elif other_goal_name == curr_neighbor:
                    mu.update_mu_by_name(tmp_path, goals[direction])

                else:

                    # Optimization:  aggressive frontier scanning for expanded paths (TMP_PATH)
                    #
                    for q in queues:
                        for i in range(0, q.size()):
                            tmp_other_path = q.queue[i][2]
                            if len(tmp_other_path) > 1:
                                # don't operate on our own queue
                                if tmp_other_path[0] == path[0]:
                                    break
                                # test queues can attach
                                if tmp_other_path[-1] == tmp_path[-1]:
                                    # my target goal
                                    if other_goal_name == tmp_other_path[0]:
                                        joined_path = intersect_paths(tmp_path, tmp_other_path)
                                        if joined_path:
                                            mu.update_mu_by_name(joined_path, joined_path[0])
                                    # i'm their target goal
                                    else:
                                        joined_path = intersect_paths(tmp_other_path, tmp_path)
                                        if joined_path:
                                            mu.update_mu_by_name(joined_path, joined_path[0])

                    # Optimization:  aggressive frontier scanning for unexpanded paths (PATH)
                    #
                    for q in queues:
                        for i in range(0, q.size()):
                            tmp_other_path = q.queue[i][2]
                            if len(tmp_other_path) > 1:
                                # don't operate on our own queue
                                if tmp_other_path[0] == path[0]:
                                    break
                                # test queues can attach
                                if tmp_other_path[-1] == path[-1]:
                                    # my target goal
                                    if other_goal_name == tmp_other_path[0]:
                                        joined_path = intersect_paths(path, tmp_other_path)
                                        if joined_path:
                                            mu.update_mu_by_name(joined_path, joined_path[0])
                                    # i'm their target goal
                                    else:
                                        joined_path = intersect_paths(tmp_other_path, path)
                                        if joined_path:
                                            mu.update_mu_by_name(joined_path, joined_path[0])


def return_your_name():
    return "dward45"


def custom_heuristic(graph, v, goal):
    """
       Feel free to use this method to try and work with different heuristics and come up with a better search algorithm.
       Args:
           graph (ExplorableGraph): Undirected graph to search.
           v (str): Key for the node to calculate from.
           goal (str): Key for the end node to calculate to.
       Returns:
           Custom heuristic distance between `v` node and `goal` node
       """


# Extra Credit: Your best search method for the race
def custom_search(graph, start, goal, data=None):
    """
    Race!: Implement your best search algorithm here to compete against the
    other student agents.

    If you implement this function and submit your code to bonnie, you'll be
    registered for the Race!

    See README.md for exercise description.

    Args:
        graph (ExplorableGraph): Undirected graph to search.
        start (str): Key for the start node.
        goal (str): Key for the end node.
        data :  Data used in the custom search.
            Will be passed your data from load_data(graph).
            Default: None.

    Returns:
        The best path as a list from the start and goal nodes (including both).
    """

    # TODO: finish this function!
    raise NotImplementedError


def load_data(graph, time_left):
    """
    Feel free to implement this method. We'll call it only once
    at the beginning of the Race, and we'll pass the output to your custom_search function.
    graph: a networkx graph
    time_left: function you can call to keep track of your remaining time.
        usage: time_left() returns the time left in milliseconds.
        the max time will be 10 minutes.

    * To get a list of nodes, use graph.nodes()
    * To get node neighbors, use graph.neighbors(node)
    * To get edge weight, use graph.get_edge_weight(node1, node2)
    """

    # nodes = graph.nodes()
    return None


def haversine_dist_heuristic(graph, v, goal):
    """
    Note: This provided heuristic is for the Atlanta race.

    Args:
        graph (ExplorableGraph): Undirected graph to search.
        v (str): Key for the node to calculate from.
        goal (str): Key for the end node to calculate to.

    Returns:
        Haversine distance between `v` node and `goal` node
    """

    # Load latitude and longitude coordinates in radians:
    vLatLong = (math.radians(graph.nodes[v]["pos"][0]), math.radians(graph.nodes[v]["pos"][1]))
    goalLatLong = (math.radians(graph.nodes[goal]["pos"][0]), math.radians(graph.nodes[goal]["pos"][1]))

    # Now we want to execute portions of the formula:
    constOutFront = 2 * 6371  # Radius of Earth is 6,371 kilometers
    term1InSqrt = (math.sin((goalLatLong[0] - vLatLong[0]) / 2)) ** 2  # First term inside sqrt
    term2InSqrt = math.cos(vLatLong[0]) * math.cos(goalLatLong[0]) * (
            (math.sin((goalLatLong[1] - vLatLong[1]) / 2)) ** 2)  # Second term
    return constOutFront * math.asin(math.sqrt(term1InSqrt + term2InSqrt))  # Straight application of formula
