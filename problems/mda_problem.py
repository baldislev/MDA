import math
from typing import *
from dataclasses import dataclass
from enum import Enum

from framework import *
from .map_heuristics import AirDistHeuristic
from .cached_map_distance_finder import CachedMapDistanceFinder
from .mda_problem_input import *

__all__ = ['MDAState', 'MDACost', 'MDAProblem', 'MDAOptimizationObjective']


@dataclass(frozen=True)
class MDAState(GraphProblemState):
    """
    An instance of this class represents a state of MDA problem.
    This state includes:
        `current_site`:
            The current site where the ambulate is at.
            The initial state stored in this field the initial ambulance location (which is a `Junction` object).
            Other states stores the last visited reported apartment (object of type `ApartmentWithSymptomsReport`),
             or the last visited laboratory (object of type `Laboratory`).
        `tests_on_ambulance`:
            Stores the reported-apartments (objects of type `ApartmentWithSymptomsReport`) which had been visited,
             and their tests are still stored on the ambulance (hasn't been transferred to a laboratory yet).
        `tests_transferred_to_lab`:
            Stores the reported-apartments (objects of type `ApartmentWithSymptomsReport`) which had been visited,
             and their tests had already been transferred to a laboratory.
        `nr_matoshim_on_ambulance`:
            The number of matoshim currently stored on the ambulance.
            Whenever visiting a reported apartment, this number is decreased by the #roommates in this apartment.
            Whenever visiting a laboratory for the first time, we transfer the available matoshim from this lab
             to the ambulance.
        `visited_labs`:
            Stores the laboratories (objects of type `Laboratory`) that had been visited at least once.
    """

    current_site: Union[Junction, Laboratory, ApartmentWithSymptomsReport]
    tests_on_ambulance: FrozenSet[ApartmentWithSymptomsReport]
    tests_transferred_to_lab: FrozenSet[ApartmentWithSymptomsReport]
    nr_matoshim_on_ambulance: int
    visited_labs: FrozenSet[Laboratory]

    @property
    def current_location(self):
        if isinstance(self.current_site, ApartmentWithSymptomsReport) or isinstance(self.current_site, Laboratory):
            return self.current_site.location
        assert isinstance(self.current_site, Junction)
        return self.current_site

    def get_current_location_short_description(self) -> str:
        if isinstance(self.current_site, ApartmentWithSymptomsReport):
            return f'test @ {self.current_site.reporter_name}'
        if isinstance(self.current_site, Laboratory):
            return f'lab {self.current_site.name}'
        return 'initial-location'

    def __str__(self):
        return f'(' \
               f'loc: {self.get_current_location_short_description()} ' \
               f'tests on ambulance: ' \
               f'{[f"{reported_apartment.reporter_name} ({reported_apartment.nr_roommates})" for reported_apartment in self.tests_on_ambulance]} ' \
               f'tests transferred to lab: ' \
               f'{[f"{reported_apartment.reporter_name} ({reported_apartment.nr_roommates})" for reported_apartment in self.tests_transferred_to_lab]} ' \
               f'#matoshim: {self.nr_matoshim_on_ambulance} ' \
               f'visited labs: {[lab.name for lab in self.visited_labs]}' \
               f')'

    def __eq__(self, other):
        """
        This method is used to determine whether two given state objects represent the same state.
        """
        assert isinstance(other, MDAState)

        sites_equal = self.current_site == other.current_site
        matoshim_equal = self.nr_matoshim_on_ambulance == other.nr_matoshim_on_ambulance
        taken_equal = self.tests_on_ambulance == other.tests_on_ambulance
        transferred_equal = self.tests_transferred_to_lab == other.tests_transferred_to_lab
        visitedLabs_equal = self.visited_labs == other.visited_labs
        return sites_equal and matoshim_equal and taken_equal and transferred_equal and visitedLabs_equal

    def __hash__(self):
        """
        This method is used to create a hash of a state instance.
        The hash of a state being is used whenever the state is stored as a key in a dictionary
         or as an item in a set.
        It is critical that two objects representing the same state would have the same hash!
        """
        return hash((self.current_site, self.tests_on_ambulance, self.tests_transferred_to_lab,
                     self.nr_matoshim_on_ambulance, self.visited_labs))

    def get_total_nr_tests_taken_and_stored_on_ambulance(self) -> int:
        """
        This method returns the total number of of tests that are stored on the ambulance in this state.
        """
        return sum([ap.nr_roommates for ap in self.tests_on_ambulance])


class MDAOptimizationObjective(Enum):
    Distance = 'Distance'
    Monetary = 'Monetary'
    TestsTravelDistance = 'TestsTravelDistance'


@dataclass(frozen=True)
class MDACost(ExtendedCost):
    """
    An instance of this class is returned as an operator cost by the method
     `MDAProblem.expand_state_with_costs()`.
    The `SearchNode`s that will be created during the run of the search algorithm are going
     to have instances of `MDACost` in SearchNode's `cost` field (instead of float values).
    The reason for using a custom type for the cost (instead of just using a `float` scalar),
     is because we want the cumulative cost (of each search node and particularly of the final
     node of the solution) to be consisted of 3 objectives:
     (i) distance, (ii) money, and (iii) tests-travel.
    The field `optimization_objective` controls the objective of the problem (the cost we want
     the solver to minimize). In order to tell the solver which is the objective to optimize,
     we have the `get_g_cost()` method, which returns a single `float` scalar which is only the
     cost to optimize.
    This way, whenever we get a solution, we can inspect the 2 different costs of that solution,
     even though the objective was only one of the costs.
    Having said that, note that during this assignment we will mostly use the distance objective.
    """
    distance_cost: float = 0.0
    monetary_cost: float = 0.0
    tests_travel_distance_cost: float = 0.0
    optimization_objective: MDAOptimizationObjective = MDAOptimizationObjective.Monetary

    def __add__(self, other):
        assert isinstance(other, MDACost)
        assert other.optimization_objective == self.optimization_objective
        return MDACost(
            optimization_objective=self.optimization_objective,
            distance_cost=self.distance_cost + other.distance_cost,
            monetary_cost=self.monetary_cost + other.monetary_cost,
            tests_travel_distance_cost=self.tests_travel_distance_cost + other.tests_travel_distance_cost)

    def get_g_cost(self) -> float:
        if self.optimization_objective == MDAOptimizationObjective.Distance:
            return self.distance_cost
        elif self.optimization_objective == MDAOptimizationObjective.Monetary:
            return self.monetary_cost
        assert self.optimization_objective == MDAOptimizationObjective.TestsTravelDistance
        return self.tests_travel_distance_cost

    def __repr__(self):
        return f'MDACost(' \
               f'dist={self.distance_cost:11.3f}m, ' \
               f'money={self.monetary_cost:11.3f}NIS, ' \
               f'tests-travel={self.tests_travel_distance_cost:11.3f}m)'


class MDAProblem(GraphProblem):
    """
    An instance of this class represents an MDA problem.
    """

    name = 'MDA'

    def __init__(self,
                 problem_input: MDAProblemInput,
                 streets_map: StreetsMap,
                 optimization_objective: MDAOptimizationObjective = MDAOptimizationObjective.Distance):
        self.name += f'({problem_input.input_name}({len(problem_input.reported_apartments)}):{optimization_objective.name})'
        initial_state = MDAState(
            current_site=problem_input.ambulance.initial_location,
            tests_on_ambulance=frozenset(),
            tests_transferred_to_lab=frozenset(),
            nr_matoshim_on_ambulance=problem_input.ambulance.initial_nr_matoshim,
            visited_labs=frozenset())
        super(MDAProblem, self).__init__(initial_state)
        self.problem_input = problem_input
        self.streets_map = streets_map
        self.map_distance_finder = CachedMapDistanceFinder(
            streets_map, AStar(AirDistHeuristic))
        self.optimization_objective = optimization_objective

    def expand_state_with_costs(self, state_to_expand: GraphProblemState) -> Iterator[OperatorResult]:
        """
        For each successor, an object of type `OperatorResult` is yielded. This object describes the
            successor state, the cost of the applied operator and its name. Look for its definition
            and use the correct fields in its c'tor. The operator name should be in the following
            format: `visit ReporterName` (with the correct reporter name) if an reported-apartment
            visit operator was applied (to take tests from the roommates of an apartment), or
            `go to lab LabName` if a laboratory visit operator was applied.
            The apartment-report object stores its reporter-name in one of its fields.
        """

        assert isinstance(state_to_expand, MDAState)
        successors = []
        tests_on_amb = state_to_expand.tests_on_ambulance
        transed = state_to_expand.tests_transferred_to_lab
        amb_capacity = self.problem_input.ambulance.total_fridges_capacity
        matoshim = state_to_expand.nr_matoshim_on_ambulance

        for app in self.get_reported_apartments_waiting_to_visit(state_to_expand):
            if ((app.nr_roommates <= matoshim) and
            app.nr_roommates <= amb_capacity - state_to_expand.get_total_nr_tests_taken_and_stored_on_ambulance()):
                succ = MDAState(current_site=app, tests_on_ambulance=frozenset(tests_on_amb | {app}),
                                tests_transferred_to_lab=frozenset(transed), nr_matoshim_on_ambulance=matoshim - app.nr_roommates,
                                visited_labs=frozenset(state_to_expand.visited_labs))
                op_name = 'visit ' + app.reporter_name
                successors.append((succ, op_name))

        for lab in self.problem_input.laboratories:
            if len(state_to_expand.tests_on_ambulance) > 0 or lab not in state_to_expand.visited_labs:
                if lab not in state_to_expand.visited_labs:
                    extra_matoshim = lab.max_nr_matoshim
                else:
                    extra_matoshim = 0

                succ = MDAState(current_site=lab, tests_on_ambulance=frozenset(),
                                tests_transferred_to_lab=frozenset(transed | tests_on_amb),
                                nr_matoshim_on_ambulance=matoshim + extra_matoshim,
                                visited_labs=frozenset((state_to_expand.visited_labs | {lab})))
                op_name = 'go to ' + lab.name
                successors.append((succ, op_name))

        for succ, op_name in successors:
            op_cost = self.get_operator_cost(state_to_expand, succ)
            yield OperatorResult(succ, op_cost, op_name)

    def get_operator_cost(self, prev_state: MDAState, succ_state: MDAState) -> MDACost:
        """
        Calculates the operator cost (of type `MDACost`) of an operator (moving from the `prev_state`
         to the `succ_state`). The `MDACost` type is defined above in this file (with explanations).
        Use the formal MDA problem's operator costs definition presented in the assignment-instructions.

        map_distance_finder() takes objects of type Junction, while field current_site can be Junction as well
        as other type that has field location of type Junction:
        """
        if type(prev_state.current_site) is Junction:
            src_junction = prev_state.current_site
        else:
            src_junction = prev_state.current_site.location

        dist_cost = self.map_distance_finder.get_map_cost_between(src_junction, succ_state.current_site.location)
        if dist_cost is None:
            dist_cost = float('inf')
            monetary_cost = float('inf')
            tests_travel_cost = float('inf')
        else:
            total_tests = prev_state.get_total_nr_tests_taken_and_stored_on_ambulance()
            active_fridge = math.ceil(total_tests / self.problem_input.ambulance.fridge_capacity)
            fridges_gas_consumption = sum(
                self.problem_input.ambulance.fridges_gas_consumption_liter_per_meter[:active_fridge])
            monetary_cost_amb = self.problem_input.gas_liter_price * (
                self.problem_input.ambulance.drive_gas_consumption_liter_per_meter + fridges_gas_consumption) * dist_cost
            monetary_cost_lab = 0
            if type(succ_state.current_site) is Laboratory:
                if total_tests > 0:
                    monetary_cost_lab += succ_state.current_site.tests_transfer_cost
                if succ_state.current_site in prev_state.visited_labs:
                    monetary_cost_lab += succ_state.current_site.revisit_extra_cost
            monetary_cost = monetary_cost_amb + monetary_cost_lab
            tests_travel_cost = total_tests * dist_cost
        return MDACost(dist_cost, monetary_cost, tests_travel_cost, self.optimization_objective)

    def is_goal(self, state: GraphProblemState) -> bool:
        """
        This method receives a state and returns whether this state is a goal.
        """
        assert isinstance(state, MDAState)
        return (state.current_site in self.problem_input.laboratories) and (len(state.tests_on_ambulance) == 0) \
               and (state.tests_transferred_to_lab == frozenset(self.problem_input.reported_apartments))

    def get_zero_cost(self) -> Cost:
        """
        Overridden method of base class `GraphProblem`. For more information, read
         documentation in the default implementation of this method there.
        In this problem the accumulated cost is not a single float scalar, but an
         extended cost, which actually includes 2 scalar costs.
        """
        return MDACost(optimization_objective=self.optimization_objective)

    def get_reported_apartments_waiting_to_visit(self, state: MDAState) -> List[ApartmentWithSymptomsReport]:
        """
        This method returns a list of all reported-apartments that haven't been visited yet.
        For the sake of determinism considerations, the returned list has to be sorted by
         the apartment's report id in an ascending order.
        """

        return sorted(list((set(self.problem_input.reported_apartments) - set(state.tests_on_ambulance)) - set(
                                                              state.tests_transferred_to_lab)), key=lambda ap: ap.report_id)

    def get_all_certain_junctions_in_remaining_ambulance_path(self, state: MDAState) -> List[Junction]:
        """
        This method returns a list of junctions that are part of the remaining route of the ambulance.
        This includes the ambulance's current location, and the locations of the reported apartments
         that hasn't been visited yet.
        The list should be ordered by the junctions index ascendingly (small to big).
        """

        aps = [ap.location for ap in self.get_reported_apartments_waiting_to_visit(state)]
        if type(state.current_site) is Junction:
            to_append = state.current_site
        else:
            to_append = state.current_site.location
        aps.append(to_append)
        return sorted(aps, key=lambda x: x.index)
