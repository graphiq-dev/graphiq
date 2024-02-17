# Copyright (c) 2022-2024 Quantum Bridge Technologies Inc.
# Copyright (c) 2022-2024 Ki3 Photonics Technologies
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
"""
Classes to compute metrics on a circuit and/or system states
"""

from abc import ABC, abstractmethod

import numpy as np
import networkx as nx

import graphiq.backends.density_matrix.functions as dmf
import graphiq.backends.stabilizer.functions.metric as sfm
from graphiq.backends.stabilizer.state import Stabilizer, MixedStabilizer


class MetricBase(ABC):
    """
    Base class for a metric.

    Metrics should be scalar values computed on the circuit and/or system states
    If a metric is used as a cost function, we aim to minimize it (i.e. smaller metric means better performance); this
    is not, however, required of metrics in general.
    """

    def __init__(self, log_steps=1, *args, **kwargs):
        """
        Create a MetricBase object

        :param log_steps: the metric values are computed at every log_steps optimization step
        :type log_steps: int
        :return: the function returns nothing
        :rtype: None
        """
        self.log_steps = log_steps
        self.name = "base"
        self.differentiable = False

        self.log = []  # will store the metric evaluations
        self._inc = 0  #

    @abstractmethod
    def evaluate(self, state, circuit):
        raise NotImplementedError(
            "Please use an inherited class, not the base metric class"
        )

    def increment(self):
        """
        Counts up the number of times a given metric has been evaluated

        :return: this function returns nothing
        :rtype: None
        """
        self._inc += 1


class Infidelity(MetricBase):
    def __init__(self, target, log_steps=1, *args, **kwargs):
        """
        Creates an Infidelity Metric object, which computes 1-fidelity with respect to the ideal_state

        :param target: the ideal state against which we compute fidelity
        :type target: QuantumState
        :param log_steps: the metric values are computed at every log_steps optimization step
        :type log_steps: int
        :raises AssertionError: if targe is not a valid density matrix
        :return: nothing
        :rtype: None
        """
        super().__init__(log_steps=log_steps, *args, **kwargs)
        self.target = target
        self.differentiable = False

    def evaluate(self, state, circuit):
        r"""
        Evaluates the infidelity between a state, $\rho$, and a target state, $\rho_{t}$.

        The infidelity is $1- F(\rho, \rho_{t})$

        For density matrices the fidelity is:
        $$
        F(\rho, \rho_{t}):=\text{Tr}[\sqrt{\sqrt{\rho} \rho_{t} \sqrt{\rho}}]^2
        $$
        or if either $\rho$ or $\rho_{t}$ is pure, then it simplifies to:
        $$
        F(\rho, \rho_{t}):=\text{Tr}[\rho \rho_{t}]
        $$

        Using the branched mixed stabilizer representation, the fidelity is:
        $$
        F(\rho, T_t) := \sum_i p_i F(T_i, T_{t})
        $$
        which assumes the target state is pure and represented by a single tableau $T_t$.

        :param state: the state to evaluate
        :type state: QuantumState
        :param circuit: circuit which generated state
                        Not used for the fidelity evaluation, but argument is provided for API consistency
        :type circuit: CircuitBase (or subclass of it)
        :raises AssertionError: if the state is not a valid density matrix
        :return: infidelity = 1 - fidelity
        :rtype: float
        """

        if self.target.rep_type == "s":
            if state.rep_type == "s":
                rep_data = state.rep_data
            else:
                tmp_state = state.copy()
                tmp_state.convert_representation("s")
                rep_data = tmp_state.rep_data
            if isinstance(self.target.rep_data, MixedStabilizer):
                assert len(self.target.rep_data.mixture) == 1
                assert self.target.rep_data.mixture[0][0] == 1.0
                tableau = self.target.rep_data.mixture[0][1]
            elif isinstance(self.target.rep_data, Stabilizer):
                tableau = self.target.rep_data.tableau

            if isinstance(rep_data, Stabilizer):
                fid = sfm.fidelity(tableau, rep_data.data)
            elif isinstance(state.rep_data, MixedStabilizer):
                fid = sum(
                    [p_i * sfm.fidelity(tableau, t_i) for p_i, t_i in rep_data.mixture]
                )

        elif self.target.rep_type == "dm":
            if state.rep_type == "dm":
                fid = dmf.fidelity(self.target.rep_data.data, state.rep_data.data)
            else:
                tmp_state = state.copy()
                tmp_state.convert_representation("dm")
                fid = dmf.fidelity(self.target.rep_data.data, tmp_state.rep_data.data)
        else:
            raise ValueError("Cannot compute the infidelity.")

        self.increment()

        if self._inc % self.log_steps == 0:
            self.log.append(1 - fid)

        return 1 - fid


class TraceDistance(MetricBase):
    def __init__(self, target, log_steps=1, *args, **kwargs):
        """
        Creates a TraceDistance Metric object, which computes the trace distance between the current state and the
        target state.

        :param target: the ideal state against which we compute fidelity
        :type target: QuantumState
        :param log_steps: the metric values are computed at every log_steps optimization step
        :type log_steps: int
        :return: nothing
        :rtype: None
        """
        super().__init__(log_steps=log_steps, *args, **kwargs)
        self.target = target
        self.differentiable = False

    def evaluate(self, state, circuit):
        r"""
        Evaluates the trace distance between the state to the target state.

        The trace distance is computed between two density matrices $\rho$ and $\sigma$ as:
        $$
        T(\rho, \sigma) = \frac{1}{2} \text{Tr}\left( \sqrt{ (\rho - \sigma)^2 } \right)
         = \\frac{1}{2} \sum_i | \lambda_i |
         $$
        :param state: the state to evaluate
        :type state: QuantumState
        :param circuit: circuit which generated state
                        Not used for the trace distance evaluation, but argument is provided for API consistency
        :type circuit: CircuitBase (or subclass of it)
        :return: the trace distance
        :rtype: float
        """
        if self.target.rep_type == "dm":
            if state.rep_type == "dm":
                trace_distance = dmf.trace_distance(
                    self.target.rep_data.data, state.rep_data.data
                )
            else:
                tmp_state = state.copy()
                tmp_state.convert_representation("dm")
                trace_distance = dmf.trace_distance(
                    self.target.rep_data.data, tmp_state.rep_data.data
                )

        else:
            raise ValueError("Cannot compute the trace distance.")

        self.increment()

        if self._inc % self.log_steps == 0:
            self.log.append(trace_distance)

        return trace_distance


class CircuitDepth(MetricBase):
    """
    A metric which calculates the circuit depth
    """

    def __init__(self, log_steps=1, depth_penalty=None, *args, **kwargs):
        """

        :param log_steps: the metric values are computed at every log_steps optimization step
        :type log_steps: int
        :param depth_penalty: a function which calculates a "cost"/penalty as a function of circuit depth
        :type depth_penalty: function
        :return: the function returns nothing
        :rtype: None
        """
        super().__init__(log_steps=log_steps, *args, **kwargs)
        self.differentiable = False
        if depth_penalty is None:
            self.depth_penalty = (
                lambda x: x
            )  # by default, the penalty for depth is the depth itself
        else:
            self.depth_penalty = depth_penalty

    def evaluate(self, state, circuit):
        """
        Calculates a scalar function of the circuit depth

        :param state: state which was created by the circuit. This is not actually used by this metric object,
                      but is nonetheless provided to guarantee a uniform API between Metric-type objects
        :type state: QuantumState
        :param circuit: the circuit to evaluate
        :type circuit: CircuitBase (or a subclass of it)
        :return: the scalar penalty resulting from circuit depth. By default, this is the circuit depth itself
        :rtype: float or int
        """

        depth = circuit.depth
        val = self.depth_penalty(depth)
        self.increment()

        if self._inc % self.log_steps == 0:
            self.log.append(val)

        return val


class Metrics(MetricBase):
    """
    Wraps around one or more metric functions, evaluating each and logging the values
    """

    _all = {  # metrics that can be used, which can be specified by the dictionary keys or as a class instance
        "Infidelity": Infidelity,
        "TraceDistance": TraceDistance,
        "CircuitDepth": CircuitDepth,
    }

    def __init__(
        self, metrics_list: list, metric_weight=None, log_steps=1, *args, **kwargs
    ):
        """
        Create a Metrics object which acts as a wrapper around Metric functions

        :param metrics_list: metrics to evaluate
        :type metrics_list: list of strings (strings should be metric names) OR MetricBase objects
                            MetricBase objects may be preferable, since we cannot set initial parameters via the list of strings.
        :param metric_weight: some representation of how to weigh the different metric results against one another
                              if None, all metrics provided are weighted equally (by 1)
                              if a list or ndarray, the metrics are a linear combination weighted by the list/ndarray values
                              Otherwise, metric_weight is a function, that can make any mathematical function of the individual
                              metric values.
        :type metric_weight: None, numpy.ndarray, list, or Function
        :param log_steps: the metric values are computed at every log_steps optimization step
        :type log_steps: int
        :return: function returns nothing
        :rtype: None
        """
        super().__init__(log_steps=log_steps, *args, **kwargs)
        # pass in either list of either strings or of specific Metric instance (must be an accepted Metric)
        _metrics = []
        for metric in metrics_list:
            if metric.__class__ in self._all.values():
                _metrics.append(metric)
            elif metric in self._all.keys():
                _metrics.append(self._all[metric]())
            else:
                raise UserWarning(
                    f"{metric} is not a recognized metric - it will not be evaluated."
                )
        self._metrics = _metrics

        if (
            metric_weight is None
            or isinstance(metric_weight, list)
            or isinstance(metric_weight, np.ndarray)
        ):
            if metric_weight is None:
                metric_weight = np.ones(len(metrics_list)).flatten()

            elif isinstance(metric_weight, list):
                metric_weight = np.array([metric_weight]).flatten()

            else:
                metric_weight = metric_weight.flatten()

            def weighting_func(state, circuit):
                return np.dot(
                    metric_weight,
                    np.array(
                        [met.evaluate(state, circuit) for met in self._metrics]
                    ).flatten(),
                )

            self.weighting_func = weighting_func
        elif callable(metric_weight):
            self.weighting_func = metric_weight

    def evaluate(self, state, circuit):
        """
        Evaluate each metric function contained by the Metrics object

        :param state: the state on which to evaluate the metrics
        :type state: QuantumState
        :param circuit: the circuit on which to evaluate the metrics
        :type circuit: CircuitBase (or a subclass of it)
        :return: this function returns nothing
        :rtype: None
        """
        val = self.weighting_func(state, circuit)
        self.increment()

        if self._inc % self.log_steps == 0:
            self.log.append(val)

        return val

    @property
    def per_metric_log(self):
        """
        The joint log of all metric functions

        :return: the log itself
        :rtype: dict (keys are metric class names, values are the logs)
        """
        m = {}
        for i, metric in enumerate(self._metrics):
            m[metric.__class__.__name__] = metric.log

        return m


# circuit metrics


class CircuitEmitterCount(MetricBase):
    """
    A metric which calculates the circuit's number of emitters'
    """

    def __init__(self, log_steps=1, n_emitter_penalty=None, *args, **kwargs):
        """

        :param log_steps: the metric values are computed at every log_steps optimization step
        :type log_steps: int
        :param n_emitter_penalty: a function which calculates a "cost"/penalty as a function of circuit's number of
        emitters
        :type n_emitter_penalty: function
        :return: the function returns nothing
        :rtype: None
        """
        super().__init__(log_steps=log_steps, *args, **kwargs)
        self.differentiable = False
        if n_emitter_penalty is None:
            self.n_emitter_penalty = (
                lambda x: x
            )  # by default, the number emitters itself
        else:
            self.n_emitter_penalty = n_emitter_penalty

    def evaluate(self, state, circuit):
        """
        Calculates a scalar function of the number of emitters

        :param state: state which was created by the circuit. This is not actually used by this metric object,
                      but is nonetheless provided to guarantee a uniform API between Metric-type objects
        :type state: QuantumState
        :param circuit: the circuit to evaluate
        :type circuit: CircuitBase (or a subclass of it)
        :return: the scalar penalty resulting from number of emitters. By default, this is the emitter count itself
        :rtype: float or int
        """

        n = circuit.n_emitters
        val = self.n_emitter_penalty(n)
        self.increment()

        if self._inc % self.log_steps == 0:
            self.log.append(val)

        return val


class CircuitCnotCount(MetricBase):
    """
    A metric which calculates the circuit's CNOT count
    """

    def __init__(self, log_steps=1, n_cnot_penalty=None, *args, **kwargs):
        """

        :param log_steps: the metric values are computed at every log_steps optimization step
        :type log_steps: int
        :param n_cnot_penalty: a function which calculates a "cost"/penalty as a function of circuit's number of
        CNOTs
        :type n_cnot_penalty: function
        :return: the function returns nothing
        :rtype: None
        """
        super().__init__(log_steps=log_steps, *args, **kwargs)
        self.differentiable = False
        if n_cnot_penalty is None:
            self.n_emitter_penalty = (
                lambda x: x
            )  # by default, the number emitters itself
        else:
            self.n_cnot_penalty = n_cnot_penalty

    def evaluate(self, state, circuit):
        """
        Calculates a scalar function of the number of emitter-emitter CNOT gates

        :param state: state which was created by the circuit. This is not actually used by this metric object,
                      but is nonetheless provided to guarantee a uniform API between Metric-type objects
        :type state: QuantumState
        :param circuit: the circuit to evaluate
        :type circuit: CircuitBase (or a subclass of it)
        :return: the scalar penalty resulting from number of CNOTs. By default, this is the CNOT count itself
        :rtype: float or int
        """

        if "Emitter-Emitter" in circuit.node_dict:
            n = len(circuit.get_node_by_labels(["Emitter-Emitter", "CNOT"]))
        else:
            n = 0
        val = self.n_cnot_penalty(n)
        self.increment()

        if self._inc % self.log_steps == 0:
            self.log.append(val)

        return val


class CircuitUnitaryCount(MetricBase):
    """
    A metric which calculates the circuit depth
    """

    def __init__(self, log_steps=1, n_unitary_penalty=None, *args, **kwargs):
        """

        :param log_steps: the metric values are computed at every log_steps optimization step
        :type log_steps: int
        :param n_unitary_penalty: a function which calculates a "cost"/penalty as a function of circuit's number of
        unitary gates
        :type n_unitary_penalty: function
        :return: the function returns nothing
        :rtype: None
        """
        super().__init__(log_steps=log_steps, *args, **kwargs)
        self.differentiable = False
        if n_unitary_penalty is None:
            self.n_unitary_penalty = (
                lambda x: x
            )  # by default, the number emitters itself
        else:
            self.n_unitary_penalty = n_unitary_penalty

    def evaluate(self, state, circuit):
        """
        Calculates a scalar function of the number of unitary gates

        :param state: state which was created by the circuit. This is not actually used by this metric object,
                      but is nonetheless provided to guarantee a uniform API between Metric-type objects
        :type state: QuantumState
        :param circuit: the circuit to evaluate
        :type circuit: CircuitBase (or a subclass of it)
        :return: the scalar penalty resulting from number of unitaries. By default, this is the unitary count itself
        :rtype: float or int
        """
        circuit = circuit.copy()
        circuit.unwrap_nodes()
        circuit.remove_identity()
        n_u = 0
        for label in [
            "SigmaX",
            "SigmaX",
            "SigmaX",
            "Phase",
            "PhaseDagger",
            "Hadamard",
            "CNOT",
        ]:
            if label in circuit.node_dict:
                n_u += len(circuit.get_node_by_labels([label]))
        val = self.n_unitary_penalty(n_u)
        self.increment()

        if self._inc % self.log_steps == 0:
            self.log.append(val)

        return val


class CircuitMaxEmitDepth(MetricBase):
    """
    A metric which calculates the circuit's maximum emitter depth
    """

    def __init__(self, log_steps=1, depth_penalty=None, *args, **kwargs):
        """

        :param log_steps: the metric values are computed at every log_steps optimization step
        :type log_steps: int
        :param depth_penalty: a function which calculates a "cost"/penalty as a function of circuit depth
        :type depth_penalty: function
        :return: the function returns nothing
        :rtype: None
        """
        super().__init__(log_steps=log_steps, *args, **kwargs)
        self.differentiable = False
        if depth_penalty is None:
            self.depth_penalty = (
                lambda x: x
            )  # by default, the penalty for depth is the depth itself
        else:
            self.depth_penalty = depth_penalty

    def evaluate(self, state, circuit):
        """
        Calculates a scalar function of the circuit depth

        :param state: state which was created by the circuit. This is not actually used by this metric object,
                      but is nonetheless provided to guarantee a uniform API between Metric-type objects
        :type state: QuantumState
        :param circuit: the circuit to evaluate
        :type circuit: CircuitBase (or a subclass of it)
        :return: the scalar penalty resulting from circuit depth. By default, this is the circuit depth itself
        :rtype: float or int
        """
        c = circuit.copy()
        c.unwrap_nodes()
        c.remove_identity()
        e_depth = {}
        for e_i in range(c.n_emitters):
            e_depth[e_i] = len(c.reg_gate_history(reg=e_i)[1]) - 2
        depth = max(e_depth.values())
        val = self.depth_penalty(depth)
        self.increment()

        if self._inc % self.log_steps == 0:
            self.log.append(val)

        return val


class CircuitMaxEmitResetDepth(MetricBase):
    """
    A metric which calculates the circuit's maximum emitter reset depth
    """

    def __init__(self, log_steps=1, depth_penalty=None, *args, **kwargs):
        """

        :param log_steps: the metric values are computed at every log_steps optimization step
        :type log_steps: int
        :param depth_penalty: a function which calculates a "cost"/penalty as a function of circuit depth
        :type depth_penalty: function
        :return: the function returns nothing
        :rtype: None
        """
        super().__init__(log_steps=log_steps, *args, **kwargs)
        self.differentiable = False
        if depth_penalty is None:
            self.depth_penalty = (
                lambda x: x
            )  # by default, the penalty for depth is the depth itself
        else:
            self.depth_penalty = depth_penalty

    def evaluate(self, state, circuit):
        """
        Calculates a scalar function of the circuit depth

        :param state: state which was created by the circuit. This is not actually used by this metric object,
                      but is nonetheless provided to guarantee a uniform API between Metric-type objects
        :type state: QuantumState
        :param circuit: the circuit to evaluate
        :type circuit: CircuitBase (or a subclass of it)
        :return: the scalar penalty resulting from circuit depth. By default, this is the circuit depth itself
        :rtype: float or int
        """
        c = circuit.copy()
        c.unwrap_nodes()
        c.remove_identity()
        reset_depths = {}
        for e_i in range(c.n_emitters):
            m_list = []  # list of indices of measurement nodes in emitters gate history
            for i, oper in enumerate(c.reg_gate_history(reg=e_i)[0]):
                # first find a list of nodes in DAG corresponding to measurements
                if type(oper).__name__ in [
                    "Input",
                    "MeasurementCNOTandReset",
                    "Output",
                ]:
                    m_list.append(i)
            reset_intervals = [
                m_list[j + 1] - m_list[j] for j in range(len(m_list) - 1)
            ]
            reset_depths[e_i] = max(reset_intervals)
        depth = max(reset_depths.values())
        val = self.depth_penalty(depth)
        self.increment()

        if self._inc % self.log_steps == 0:
            self.log.append(val)

        return val


class CircuitMaxEmitEffDepth(MetricBase):
    """
    A metric which calculates the circuit's maximum emitter effective depth
    """

    def __init__(self, log_steps=1, depth_penalty=None, *args, **kwargs):
        """

        :param log_steps: the metric values are computed at every log_steps optimization step
        :type log_steps: int
        :param depth_penalty: a function which calculates a "cost"/penalty as a function of circuit depth
        :type depth_penalty: function
        :return: the function returns nothing
        :rtype: None
        """
        super().__init__(log_steps=log_steps, *args, **kwargs)
        self.differentiable = False
        if depth_penalty is None:
            self.depth_penalty = (
                lambda x: x
            )  # by default, the penalty for depth is the depth itself
        else:
            self.depth_penalty = depth_penalty

    def evaluate(self, state, circuit):
        """
        Calculates a scalar function of the circuit depth

        :param state: state which was created by the circuit. This is not actually used by this metric object,
                      but is nonetheless provided to guarantee a uniform API between Metric-type objects
        :type state: QuantumState
        :param circuit: the circuit to evaluate
        :type circuit: CircuitBase (or a subclass of it)
        :return: the scalar penalty resulting from circuit depth. By default, this is the circuit depth itself
        :rtype: float or int
        """
        c = circuit.copy()
        c.unwrap_nodes()
        c.remove_identity()
        eff_depth = {}
        for e_i in range(c.n_emitters):
            node_list = []
            for i, oper in enumerate(c.reg_gate_history(reg=e_i)[0]):
                # first find a list of nodes in DAG corresponding to measurements
                if type(oper).__name__ in [
                    "Input",
                    "MeasurementCNOTandReset",
                    "Output",
                ]:
                    node_list.append(c.reg_gate_history(reg=e_i)[1][i])
            node_depth_list = [c._max_depth(n) for n in node_list]
            depth_diff = [
                node_depth_list[j + 1] - node_depth_list[j]
                for j in range(len(node_list) - 1)
            ]
            eff_depth[e_i] = max(depth_diff)
        depth = max(eff_depth.values())
        val = self.depth_penalty(depth)
        self.increment()

        if self._inc % self.log_steps == 0:
            self.log.append(val)

        return val


class CircuitMeasureCount(MetricBase):
    """
    A metric which calculates the circuit's number of measurements
    """

    def __init__(self, log_steps=1, m_penalty=None, *args, **kwargs):
        """

        :param log_steps: the metric values are computed at every log_steps optimization step
        :type log_steps: int
        :param m_penalty: a function which calculates a "cost"/penalty as a function of number of measurement
        :type m_penalty: function
        :return: the function returns nothing
        :rtype: None
        """
        super().__init__(log_steps=log_steps, *args, **kwargs)
        self.differentiable = False
        if m_penalty is None:
            self.m_penalty = (
                lambda x: x
            )  # by default, the penalty for depth is the depth itself
        else:
            self.m_penalty = m_penalty

    def evaluate(self, state, circuit):
        """
        Calculates a scalar function of the number of measurements

        :param state: state which was created by the circuit. This is not actually used by this metric object,
                      but is nonetheless provided to guarantee a uniform API between Metric-type objects
        :type state: QuantumState
        :param circuit: the circuit to evaluate
        :type circuit: CircuitBase (or a subclass of it)
        :return: the scalar penalty resulting from number of measurements. By default, this is the measurement count
        itself
        :rtype: float or int
        """
        c = circuit.copy()
        n = len(c.get_node_by_labels(["MeasurementCNOTandReset"]))
        val = self.m_penalty(n)
        self.increment()

        if self._inc % self.log_steps == 0:
            self.log.append(val)

        return val


class GraphMetric(MetricBase):
    """
    A class to calculate a given graph metric
    """

    def __init__(self, graph: nx.Graph, log_steps=1, penalty=None, *args, **kwargs):
        """

        :param log_steps: the metric values are computed at every log_steps optimization step
        :type log_steps: int
        :param m_penalty: a function which calculates a "cost"/penalty as a function of number of measurement
        :type m_penalty: function
        :return: the function returns nothing
        :rtype: None
        """
        super().__init__(log_steps=log_steps, *args, **kwargs)
        self.differentiable = False
        assert isinstance(graph, nx.Graph), "input graph must be an nx.Graph object"
        self.graph = graph
        if penalty is None:
            self.penalty = (
                lambda x: x
            )  # by default, the penalty for depth is the depth itself
        else:
            self.penalty = penalty

    def evaluate(self, graph_metric: str):
        """
        Calculates a metric for a given graph state.

        :param graph_metric: The list of valid metrics are
        "max_between",
        "max_close",
        "min_close",
        "mean_nei_deg",
        "max_deg",
        "node_connect",
        "edge_connect",
        "assort",
        "radius",
        "diameter",
        "periphery",
        "center",
        "cluster",
        "local_efficiency",
        "global_efficiency",
        "node",
        "avg_shortest_path",
        "n_edges",
        "pop"
        :type graph_metric: str
        :return: the resulting value for the metric
        :rtype: float or int
        """
        g = self.graph.copy()
        met = graph_met_value(graph_metric, g)
        val = self.penalty(met)
        self.increment()

        if self._inc % self.log_steps == 0:
            self.log.append(val)

        return val


def graph_met_value(graph_metric, g):
    """
    Evaluates the graph metric for the given graph.
    :param graph_metric: the abbreviation for the graph metric to be evaluated
    :type graph_metric: str
    :param g: graph at study
    :type g: nx.Graph
    :return: the graph metric value
    :rtype: int or float
    """
    if graph_metric == "max_between":
        dict_centrality = nx.betweenness_centrality(g)
        graph_value = max(dict_centrality.values())
    elif graph_metric == "max_close":
        dict_centrality = nx.closeness_centrality(g)
        graph_value = max(dict_centrality.values())
    elif graph_metric == "min_close":
        dict_centrality = nx.closeness_centrality(g)
        graph_value = min(dict_centrality.values())
    elif graph_metric == "mean_nei_deg":
        # the mean of the "average neighbors degree" over all nodes in graph
        dict_met = nx.average_neighbor_degree(g)
        graph_value = np.mean(list(dict_met.values()))
    elif graph_metric == "max_deg":
        dict_met = dict(g.degree())
        graph_value = max(list(dict_met.values()))
    elif graph_metric == "node_connect":
        graph_value = nx.node_connectivity(g)
    elif graph_metric == "edge_connect":
        graph_value = nx.edge_connectivity(g)
    elif graph_metric == "assort":
        graph_value = nx.degree_assortativity_coefficient(g)
    elif graph_metric == "radius":
        graph_value = nx.radius(g)
    elif graph_metric == "diameter":
        graph_value = nx.diameter(g)
    elif graph_metric == "periphery":
        # num of nodes with distance equal to diameter
        graph_value = len(nx.periphery(g))
    elif graph_metric == "center":
        # num of nodes with distance equal to radius
        graph_value = len(nx.center(g))
    elif graph_metric == "cluster":
        graph_value = nx.average_clustering(g)
    elif graph_metric == "local_efficiency":
        graph_value = nx.local_efficiency(g)
    elif graph_metric == "global_efficiency":
        graph_value = nx.global_efficiency(g)
    elif graph_metric == "node":
        graph_value = g.number_of_nodes()
    elif graph_metric == "avg_shortest_path":
        graph_value = nx.average_shortest_path_length(g)
    elif graph_metric == "n_edges":
        graph_value = nx.number_of_edges(g)
    elif graph_metric == "pop":
        nodes = g.number_of_nodes()
        edges = g.size()
        graph_value = edges / ((nodes * (nodes - 1)) / 2)
    else:
        raise ValueError(
            f"Graph metric {graph_metric} not found. It may not be implemented"
        )

    return graph_value
