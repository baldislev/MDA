from .graph_problem_interface import *
from .astar import AStar
from typing import Optional, Callable
import numpy as np
import math


class AStarEpsilon(AStar):
    """
    This class implements the (weighted) A*Epsilon search algorithm.
    A*Epsilon algorithm basically works like the A* algorithm, but with
    another way to choose the next node to expand from the open queue.
    """

    solver_name = 'A*eps'

    def __init__(self,
                 heuristic_function_type: HeuristicFunctionType,
                 within_focal_priority_function: Callable[[SearchNode, GraphProblem, 'AStarEpsilon'], float],
                 heuristic_weight: float = 0.5,
                 max_nr_states_to_expand: Optional[int] = None,
                 focal_epsilon: float = 0.1,
                 max_focal_size: Optional[int] = None):
        # A* is a graph search algorithm. Hence, we use close set.
        super(AStarEpsilon, self).__init__(heuristic_function_type, heuristic_weight,
                                           max_nr_states_to_expand=max_nr_states_to_expand)
        self.focal_epsilon = focal_epsilon
        if focal_epsilon < 0:
            raise ValueError(f'The argument `focal_epsilon` for A*eps should be >= 0; '
                             f'given focal_epsilon={focal_epsilon}.')
        self.within_focal_priority_function = within_focal_priority_function
        self.max_focal_size = max_focal_size

    def _init_solver(self, problem):
        super(AStarEpsilon, self)._init_solver(problem)

    def _extract_next_search_node_to_expand(self, problem: GraphProblem) -> Optional[SearchNode]:
        """
        Extracts the next node to expand from the open queue,
         by focusing on the current FOCAL and choosing the node
         with the best within_focal_priority from it.
        """

        focal_list = []
        if self.open.is_empty():
            return None
        else:
            min_exp_priority = self.open.peek_next_node().expanding_priority
            max_exp_priority = min_exp_priority*(1 + self.focal_epsilon)
            if self.max_focal_size is not None:
                while len(focal_list) < self.max_focal_size and not self.open.is_empty():
                    if self.open.peek_next_node().expanding_priority <= max_exp_priority:
                        focal_list.append(self.open.pop_next_node())
                    else:
                        break
            else:
                while not self.open.is_empty():
                    if self.open.peek_next_node().expanding_priority <= max_exp_priority:
                        focal_list.append(self.open.pop_next_node())
                    else:
                        break
            index = np.argmin(np.array([self.within_focal_priority_function(n, problem, self) for n in focal_list]))
            node = focal_list.pop(index)

            for x in focal_list:
                self.open.push_node(x)
            if self.use_close:
                self.close.add_node(node)
            return node





