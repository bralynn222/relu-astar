"""Exact A* for a fixed-size 4-neighbor unit-cost grid, executed as a
fixed-weight recurrent ReLU circuit.

No training. No search/queue/branch/comparison/argmin/argmax in inference.
The recurrent circuit uses ONLY fixed linear maps, addition of biases,
ReLU, and a fixed number of iterations (N = H*W). Its state stores OPEN,
CLOSED, g-scores, parent links, and success. Both the expansion order and
parent chain are neural outputs.

The A* variant implemented here is:
  - 4-connected movement, unit edge cost, static walls
  - Manhattan heuristic
  - Priority tie-break: (f, h, row-major index), smallest first
  - Update parent only when tentative_g < existing_g
  - Stop when goal is popped, or no open node exists

The weights are constructed analytically for the selected H,W and never
learned. Like a normal fixed-size neural network, changing H,W requires
instantiating a new fixed-size circuit.

Input encoding is a flattened wall mask, a start one-hot vector and a
goal one-hot vector. Valid inputs must have binary walls and exactly one
start and goal, both on free cells.

Required: numpy
"""

import numpy as np


def relu(x):
    """The sole nonlinear activation used by the circuit."""
    return np.maximum(x, 0.0)


class PureWeightNeuralAStar:
    def __init__(self, height: int, width: int):
        self.H = int(height)
        self.W = int(width)
        self.N = self.H * self.W
        if self.H <= 0 or self.W <= 0:
            raise ValueError('height and width must be positive')

        n = self.N
        # All parameters below are fixed weights, assembled once.
        rows, cols = np.divmod(np.arange(n), self.W)
        self.W_neighbor = (
            np.abs(rows[:, None] - rows[None, :])
            + np.abs(cols[:, None] - cols[None, :]) == 1
        ).astype(np.float64)
        self.W_heuristic = (
            np.abs(rows[:, None] - rows[None, :])
            + np.abs(cols[:, None] - cols[None, :])
        ).astype(np.float64)
        self.b_index = np.arange(n, dtype=np.float64)
        self.b_ones = np.ones(n, dtype=np.float64)
        self.b_ones_matrix = np.ones((n, n), dtype=np.float64)

        # Distinct integer priorities encode exact lexicographic (f,h,index).
        self.K_h = n + 1
        self.K_f = (self.H + self.W) * self.K_h + n + 1
        self.INF = n + 1
        self.GATE = n + 3

        # Differences in priority must be representable exactly by float64.
        max_priority = (self.INF + self.H + self.W) * self.K_f
        if max_priority >= 2**52:
            raise ValueError('Grid is too large for exact float64 priorities')

    def forward(self, walls, start_onehot, goal_onehot):
        """Pure fixed-weight recurrent neural inference.

        Returns:
            success: scalar (0 or 1)
            expansion_onehot: [N, N] - one selected node per A* step, then zeros
            reverse_path_onehot: [N, N] - goal -> ... -> start, then zeros
            parent_matrix: [N, N] - parent[child, parent]
            g_scores: [N]

        The body below contains no data-dependent branches, loops, dynamic
        indexing, min/max except ReLU, sorting, or queue operations.
        """
        wall = np.asarray(walls, dtype=np.float64).reshape(self.N)
        start = np.asarray(start_onehot, dtype=np.float64).reshape(self.N)
        goal = np.asarray(goal_onehot, dtype=np.float64).reshape(self.N)

        # Fixed input projection: Manhattan distance to the one-hot goal.
        h = self.W_heuristic @ goal

        # Recurrent state vectors/matrix.
        g = self.INF * (self.b_ones - start)
        opened = start.copy()
        closed = np.zeros(self.N, dtype=np.float64)
        parent = np.zeros((self.N, self.N), dtype=np.float64)
        done = np.float64(0.0)

        expansion = np.zeros((self.N, self.N), dtype=np.float64)

        # Exactly N recurrent ticks; recurrence freezes after goal is popped
        # or OPEN becomes empty. Every operation is affine or ReLU.
        for tick in range(self.N):
            available = relu(opened - done)

            # A* node priority: (f,h,row-major-index) encoded as a scalar.
            priority = (g + h) * self.K_f + h * self.K_h + self.b_index

            # Each ReLU pair implements the Boolean integer comparator
            # [priority[j] < priority[i]], without a comparison operator.
            diff = priority[:, None] - priority[None, :]
            smaller = relu(diff) - relu(diff - 1.0)
            smaller_and_open = relu(smaller + available[None, :] - 1.0)

            # Unique winner: OPEN[i] AND no other OPEN[j] has lower priority.
            winner = relu(available - smaller_and_open.sum(axis=1))
            expansion[tick] = winner

            # Goal detection is an AND gate followed by a fixed sum.
            goal_popped = relu(winner + goal - 1.0).sum()
            expand_winner = relu(winner - goal_popped)

            closed_next = relu(closed + winner) - relu(closed + winner - 1.0)

            # Exactly one winner or none; fixed adjacency matrix fans out to
            # its four neighboring cells. Walls and CLOSED inhibit expansion.
            candidate = relu(self.W_neighbor @ expand_winner - wall - closed_next)

            # Gating a nonnegative number x by a binary bit a, using ReLU:
            # gate(x,a) = ReLU(x - GATE * (1-a)), where GATE > all possible x.
            selected_g = relu(g - self.GATE * (1.0 - expand_winner))
            tentative = self.W_neighbor @ selected_g + candidate

            # Strict improvement predicate: candidate AND (tentative < g).
            delta = g - tentative
            is_better = relu(delta) - relu(delta - 1.0)
            improve = relu(candidate + is_better - 1.0)

            # g_new = improve ? tentative : g  (two ReLU gates)
            g = (relu(g - self.GATE * improve)
                 + relu(tentative - self.GATE * (1.0 - improve)))

            # parent_new[i,j] = improve[i] ? winner[j] : parent_old[i,j]
            parent = (relu(parent - improve[:, None])
                      + relu(improve[:, None] + winner[None, :] - 1.0))

            # OPEN = (OPEN union newly improved) minus popped node.
            opened = (relu(opened + improve) - relu(opened + improve - 1.0)) - winner
            closed = closed_next
            done = relu(done + goal_popped) - relu(done + goal_popped - 1.0)

        # Neural path reconstruction: fixed N more recurrent ticks. A pair
        # of ReLU gates computes each (current_onehot AND parent_bit), then
        # sums the child dimension to follow parent pointers. Once the start
        # is reached, its zero parent row automatically ends the sequence.
        current = relu(goal + done - 1.0)
        reverse_path = np.zeros((self.N, self.N), dtype=np.float64)
        for tick in range(self.N):
            reverse_path[tick] = current
            current = relu(parent + current[:, None] - 1.0).sum(axis=0)

        return {
            'success': done,
            'expansion_onehot': expansion,
            'reverse_path_onehot': reverse_path,
            'parent_matrix': parent,
            'g_scores': g,
        }


def encode_inputs(grid, start, goal):
    """Outside the network: encode human-readable inputs as tensor inputs."""
    grid = np.asarray(grid)
    if grid.ndim != 2:
        raise ValueError('grid must be a 2-D array')
    h, w = grid.shape
    if not np.all((grid == 0) | (grid == 1)):
        raise ValueError('grid must contain only 0 (free) or 1 (wall)')
    sy, sx = start
    gy, gx = goal
    if not (0 <= sy < h and 0 <= sx < w and 0 <= gy < h and 0 <= gx < w):
        raise ValueError('start and goal must be on the grid')
    if grid[sy, sx] or grid[gy, gx]:
        raise ValueError('start and goal must be free cells')
    start_hot = np.zeros(h*w, dtype=np.float64)
    goal_hot = np.zeros(h*w, dtype=np.float64)
    start_hot[sy*w + sx] = 1.0
    goal_hot[gy*w + gx] = 1.0
    return grid.astype(np.float64).ravel(), start_hot, goal_hot


def decode_output(result, width):
    """Outside the network: convert fixed-size one-hot outputs to coordinates.

    These readout operations don't compute the path. The neural circuit
    already generated every position in the correct parent order.
    """
    expanded = [int(np.argmax(row)) for row in result['expansion_onehot'] if row.sum() == 1.0]
    expanded = [(index // width, index % width) for index in expanded]
    if result['success'] == 0.0:
        return None, expanded
    reverse = result['reverse_path_onehot']
    indices = [int(np.argmax(row)) for row in reverse if row.sum() == 1.0]
    path = [(index // width, index % width) for index in reversed(indices)]
    return path, expanded


def neural_astar(grid, start, goal):
    """Convenience wrapper. Model construction can be cached for reuse."""
    grid = np.asarray(grid)
    net = PureWeightNeuralAStar(*grid.shape)
    walls, start_hot, goal_hot = encode_inputs(grid, start, goal)
    result = net.forward(walls, start_hot, goal_hot)
    path, _ = decode_output(result, grid.shape[1])
    return path


if __name__ == '__main__':
    grid = np.array([
        [0, 0, 0, 0, 0],
        [0, 1, 1, 1, 0],
        [0, 0, 0, 1, 0],
        [0, 1, 0, 0, 0],
        [0, 0, 0, 0, 0],
    ], dtype=np.int64)
    net = PureWeightNeuralAStar(*grid.shape)
    out = net.forward(*encode_inputs(grid, (0, 0), (2, 2)))
    path, expanded = decode_output(out, grid.shape[1])
    print('Path:', path)
    print('Expanded:', expanded)
    print('Success:', bool(out['success']))
