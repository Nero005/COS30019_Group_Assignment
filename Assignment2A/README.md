
Group Assignment 2A - Tree Based Search
----------------------------------------------

1. OVERVIEW
This project implements six graph search algorithms and a step-by-step visualiser for them.

The graph is given in a simple text file (see the tests/ folder) with node coordinates, directed edges (with costs), an origin node and one or more destination nodes.

The six algorithms are:
  DFS   - Depth-First Search (U)
  BFS   - Breadth-First Search (U)
  GBFS  - Greedy Best-First Search (I)
  AS    - A* Search (I)
  CUS1  - Uniform Cost Search (UCS) (U)
  CUS2  - Iterative Deepening A* (IDA*) (I)

The informed algorithms use the straight-line distance to the closest destination as their heuristic.
Ties are broken in a consistent way. Neighbors are expanded in ascending node-number order, and priority queues use insertion order to break equal-priority ties.


2. FILES
core.py       Reads a problem file. Also holds the heuristic function (Euclidean distance to the nearest goal).
search.py     Command-line entry point for running ONE method. Usage: python search.py <file> <method>
run_test.py   Runs EVERY method on EVERY file in tests/ and prints a results table. Also writes results.csv.
animate.py    Replays one method step by step and saves an animated GIF to visPhoto/ file. Shows a window with the animation.

dfs.py        Depth-First Search
bfs.py        Breadth-First Search
gbfs.py       Greedy Best-First Search
astar.py      A* Search
cus1.py       Uniform Cost Search
cus2.py       Iterative Deepening A* (IDA*)

tests/        Example problem files (test_01.txt to test_16.txt)
visPhoto/     Output folder for the animation GIFs
results.csv   Output table produced by run_test.py


3. INPUT FILE FORMAT
Example:

  Nodes:
  1: (4,1)
  2: (2,2)
  ...
  Edges:
  (2,1): 4
  (3,1): 5
  ...
  Origin:
  2
  Destinations:
  5; 4

What they mean:
  Nodes: Each line is "node: (x, y)" - a coordinate
  Edges: Each line is "(from,to): cost" - a DIRECTED edge
  Origin: A single node number
  Destinations: One or more node numbers separated by ";"


4. HOW TO RUN
Meet requirements below first
Run a single search method:    python search.py tests/test_01.txt AS
  Output:
    tests/test_01.txt AS
    <goal node> <nodes created>
    <path, space separated>

Run all methods on all test files and make the table:      python run_test.py
(This prints the results table and saves results.csv. Each cell is:  goal / nodes created / path cost)

Make an animation GIF for one method:
  python animate.py tests/test_01.txt BFS
  python animate.py tests/test_01.txt BFS noshow

The GIF is saved to visPhoto/<testname>_<method>.gif. If "noshow" is given, no window will open. If given, a window also plays the animation.

Valid method names in this program: DFS, BFS, GBFS, AS, CUS1, CUS2.

5. THE ANIMATION
Each frame is one step of the search:

  green ring   - origin node
  orange ring  - destination node
  tomato       - node being expanded right now
  gold         - node in the frontier (waiting to be expanded)
  light blue   - node already expanded
  light grey   - node not reached yet
  blue arrows  - path to the current node
  red arrows   - final solution path

The title shows the step number, the node being expanded, how many nodes have been created, and (for informed searches) the g, h or f values and the current limit.

Note: animate.py re-runs the algorithm step by step and checks the result against the real search() function. If they differ, a warning is printed.


----------------------------------------------
Requirements

Python 3.x
matplotlib  (only needed for animate.py, install with: python -m pip install matplotlib)


