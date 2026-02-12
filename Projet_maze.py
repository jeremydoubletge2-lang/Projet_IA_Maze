# =========================================
# MAZE + A* + DIJKSTRA + VISUALISATION
# =========================================

import heapq
import random
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.colors import ListedColormap


class Maze:
    def __init__(self, width, height, start, goal):
        self.width = width
        self.height = height
        self.start = start
        self.goal = goal
        self.grid = [[0 for _ in range(width)] for _ in range(height)]

    # ----------------------------
    # OUTILS
    # ----------------------------

    def in_bounds(self, x, y):
        return 0 <= x < self.width and 0 <= y < self.height

    def is_free(self, x, y):
        return self.grid[y][x] == 0

    def neighbors(self, x, y):
        directions = [(1, 0), (-1, 0), (0, 1), (0, -1)]
        result = []
        for dx, dy in directions:
            nx, ny = x + dx, y + dy
            if self.in_bounds(nx, ny) and self.is_free(nx, ny):
                result.append((nx, ny))
        return result

    def generate_random_obstacles(self, obstacle_prob=0.3):
        for y in range(self.height):
            for x in range(self.width):
                if (x, y) != self.start and (x, y) != self.goal:
                    if random.random() < obstacle_prob:
                        self.grid[y][x] = 1

    def heuristic(self, x, y):
        gx, gy = self.goal
        return abs(x - gx) + abs(y - gy)

    def reconstruct_path(self, came_from):
        current = self.goal
        path = [current]
        while current in came_from:
            current = came_from[current]
            path.append(current)
        path.reverse()
        return path

    # ----------------------------
    # A*
    # ----------------------------

    def solve_astar(self):
        open_set = []
        heapq.heappush(open_set, (0, self.start))

        g_score = {self.start: 0}
        came_from = {}
        explored = set()

        while open_set:
            _, current = heapq.heappop(open_set)
            explored.add(current)

            if current == self.goal:
                return self.reconstruct_path(came_from), explored

            for neighbor in self.neighbors(*current):
                tentative_g = g_score[current] + 1

                if neighbor not in g_score or tentative_g < g_score[neighbor]:
                    g_score[neighbor] = tentative_g
                    f_score = tentative_g + self.heuristic(*neighbor)
                    heapq.heappush(open_set, (f_score, neighbor))
                    came_from[neighbor] = current

        return None, explored

    # ----------------------------
    # DIJKSTRA
    # ----------------------------

    def solve_dijkstra(self):
        open_set = []
        heapq.heappush(open_set, (0, self.start))

        dist = {self.start: 0}
        came_from = {}
        explored = set()

        while open_set:
            current_dist, current = heapq.heappop(open_set)
            explored.add(current)

            if current == self.goal:
                return self.reconstruct_path(came_from), explored

            for neighbor in self.neighbors(*current):
                tentative_dist = current_dist + 1

                if neighbor not in dist or tentative_dist < dist[neighbor]:
                    dist[neighbor] = tentative_dist
                    heapq.heappush(open_set, (tentative_dist, neighbor))
                    came_from[neighbor] = current

        return None, explored

    # ----------------------------
    # AFFICHAGE
    # ----------------------------

    def plot(self, path, explored, title):
        display_grid = np.array(self.grid)

        # Codes :
        # 0 blanc = libre
        # 1 noir = obstacle
        # 2 bleu clair = exploré
        # 3 rouge = chemin final
        # 4 vert = start
        # 5 jaune = goal

        for (x, y) in explored:
            if (x, y) != self.start and (x, y) != self.goal:
                display_grid[y][x] = 2

        if path:
            for (x, y) in path:
                if (x, y) != self.start and (x, y) != self.goal:
                    display_grid[y][x] = 3

        sx, sy = self.start
        gx, gy = self.goal
        display_grid[sy][sx] = 4
        display_grid[gy][gx] = 5

        cmap = ListedColormap([
            "white",
            "black",
            "lightblue",
            "red",
            "green",
            "yellow"
        ])

        plt.figure()
        plt.imshow(display_grid, cmap=cmap)
        plt.title(title)
        plt.xticks(range(self.width))
        plt.yticks(range(self.height))
        plt.grid(True)
        plt.show()


# =========================================
# TEST COMPARAISON
# =========================================

maze = Maze(10, 10, (0, 0), (9,9))
maze.generate_random_obstacles(0.3)

# A*
path_a, explored_a = maze.solve_astar()
maze.plot(path_a, explored_a, "A* : exploration vs chemin final")

print("A* - cases explorées :", len(explored_a))
print("A* - longueur chemin :", len(path_a) if path_a else "Aucun chemin")

# Dijkstra
path_d, explored_d = maze.solve_dijkstra()
maze.plot(path_d, explored_d, "Dijkstra : exploration vs chemin final")

print("Dijkstra - cases explorées :", len(explored_d))
print("Dijkstra - longueur chemin :", len(path_d) if path_d else "Aucun chemin")
