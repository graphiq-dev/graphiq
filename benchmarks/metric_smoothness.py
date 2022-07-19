"""
This module is meant to facilitate "smoothness" testing of various metrics which may serve as a
cost function for our solver optimization.

We currently plan the support the following studies:

1. Characterization of metric sensitivity to individual operations
2. Characterization of metric evolution over a sequence of operations

For each study, we will consider:
1. Different initial graphs/circuits
2. Different operations from both graph ops (e.g. complementation, merging) and circuit operations
   (e.g. single qubit, two-qubit ops)
3. Different places on which to apply the operations on the circuit

We will gather the following statistics
1. Initial metric value (baseline)
2. The individual operations applied and their corresponding metric value
   (should define an encoding for each type of operation)
3. Mean/std dev
4. min, 25th percentile, median, 75th percentile, max

# TODO: consider how best to use "ray" for supporting multiple processes. Currently, the methods we're running are
# quick enough that it feels like a lot of unnecessary overhead...
"""
from abc import ABC, abstractmethod
import math
import itertools
import random
import copy
import pandas as pd
import time
import numpy as np

import src.metrics as met
from src.io import IO
import benchmarks.graph_states as gs


class MetricSmoothnessTestBase(ABC):
    """
    Base class for metric smoothness assessment
    """

    def __init__(
        self,
        metric_iter,
        allowed_ops,
        test_representations,
        save_dir="",
        folder_name="metric-benchmark",
        max_per_op_sample=100,
    ):
        self.metric_iter = metric_iter
        self.allowed_ops = allowed_ops
        self.test_rep_name = [
            n[0] for n in test_representations
        ]  # the first index is the graph/circuit name
        self.test_rep = [
            n[1] for n in test_representations
        ]  # the second index is the actual graph/circuit
        self.max_per_op_sample = max_per_op_sample

        # set up saving
        self.save_dir = save_dir
        self.io = IO.new_directory(
            path=IO.default_path.joinpath(self.save_dir),
            folder=folder_name,
            include_date=True,
            include_time=True,
            include_id=True,
        )

    @abstractmethod
    def metric_sensitivity_to_op(self):
        raise NotImplementedError(f"MetricSmoothnessBase class is abstract")

    @abstractmethod
    def metric_along_path(self):
        raise NotImplementedError(f"MetricSmoothnessBase class is abstract")

    @staticmethod
    def apply_operation(test_obj, op_name, loc_info):
        """
        Applies an operation by name to the given test object
        """
        getattr(test_obj, op_name)(*loc_info)


class GraphOpMetricSmoothnessTest(MetricSmoothnessTestBase):
    """
    Class for metric smoothness assessment. This class focuses on operations which are
    valid on graph states rather than operations on circuits.
    """

    def __init__(self, metric_iter, test_iter, save_dir="", folder_name="metric-benchmarking-graph-smoothness"):
        # allowed ops are described by tuples (a, b)
        allowed_ops = [
            ("measure_x", 1),
            ("measure_y", 1),
            ("measure_z", 1),
            ("local_complementation", 1),
            # ("merge", 2),
            ("add_edge", 2),
        ]
        self.metric_sensitivity_op_df = None
        super().__init__(metric_iter, allowed_ops, test_iter, save_dir=save_dir, folder_name=folder_name)

    def metric_sensitivity_to_op(self, save=True, verbose=True):
        tuples = []
        for metric_class in self.metric_iter:
            for graph_name, graph in zip(self.test_rep_name, self.test_rep):
                for op_name, n_op in self.allowed_ops:
                    for i in range(
                        min(self.max_per_op_sample, math.comb(graph.n_node, n_op))
                    ):
                        tuples.append((metric_class.__name__, graph_name, op_name, i))

        row_index = pd.MultiIndex.from_tuples(
            tuples, names=("metric", "graph", "op", "index")
        )
        column_index = [
            "init score",
            "op location",
            "updated score",
            "metric runtime (ms)",
        ]
        df = pd.DataFrame(index=row_index, columns=column_index)
        for metric_class in self.metric_iter:
            for graph_name, graph in zip(self.test_rep_name, self.test_rep):
                metric = metric_class(
                    target=graph
                )  # TODO: refactor so that all metrics have target arg
                # TODO: account for case where metrics depend on the circuit
                df.loc[
                    (metric_class.__name__, graph_name), "init score"
                ] = metric.evaluate(graph, None)

                for (op_name, n_op) in self.allowed_ops:
                    if verbose:
                        print(f'Testing metric: {metric_class.__name__}, graph: {graph_name}, operation: {op_name}')
                    option_num = math.comb(graph.n_node, n_op)
                    location_iter = itertools.combinations(
                        graph.get_nodes_id_form(), n_op
                    )
                    if option_num > self.max_per_op_sample:
                        location_iter = random.sample(
                            list(location_iter), self.max_per_op_sample
                        )
                    for i, node_info in enumerate(location_iter):
                        tmp_graph = copy.deepcopy(graph)
                        if verbose:
                            print(f'Applying op...')
                        self.apply_operation(tmp_graph, op_name, node_info)
                        df.loc[
                            (metric_class.__name__, graph_name, op_name, i),
                            "op location",
                        ] = node_info
                        if verbose:
                            print(f'Evaluating metric...')
                        start_time = time.time()
                        new_score = metric.evaluate(
                            tmp_graph, None
                        )  # assumes for now that metric doesn't depend on circuit
                        runtime = time.time() - start_time
                        df.loc[
                            (metric_class.__name__, graph_name, op_name, i),
                            "updated score",
                        ] = new_score
                        df.loc[
                            (metric_class.__name__, graph_name, op_name, i),
                            "metric runtime (ms)",
                        ] = (
                            runtime / 1000
                        )
        if save:
            self.io.save_dataframe(df, "metric_sensitivity_to_op.csv", index=True)

        self.metric_sensitivity_op_df = df
        return df

    def get_stats_op_sensitivity2(self, filter_dict={}):
        data_df = self.filter_op_sensitivity(filter_dict)
        stats = {
            "updated score": pd.to_numeric(data_df["updated score"]).describe(),
            "metric runtime (ms)": pd.to_numeric(
                data_df["metric runtime (ms)"]
            ).describe(),
        }
        return stats

    def filter_op_sensitivity(self, filter_dict):
        """
        Allows you to get the last computed dataframe, filtered by metric, target graph, OR operation name

        Reference:
        https://stackoverflow.com/questions/25224545/filtering-multiple-items-in-a-multi-index-python-panda-dataframe

        :param filter_dict: dictionary which explains the data we want to keep.
                            Key = category being filtered, Value = expected category value (list or single val)
        :type filter_dict: dict
        :return: the filtered dataframe
        :rtype: panda.dataframe
        """
        # TODO: figure out if we can filter all at once on an arbitrary number of conditions, instead of sequentially
        df = self.metric_sensitivity_op_df
        for category, val in filter_dict.items():
            if not isinstance(val, list):
                val = [val]
            df = df.loc[(df.index.get_level_values(category).isin(val))]

        return df

    def metric_along_path(self):
        # TODO
        pass


if __name__ == "__main__":
    # linear cluster state 3 qubits
    linear3 = gs.linear_cluster_state(3)
    linear9 = gs.linear_cluster_state(9)
    linear24 = gs.linear_cluster_state(27)

    # TODO: save the graphs and seed as well
    rng = np.random.default_rng(seed=0)
    random3 = gs.random_graph_state(3, p_edge=0.05)
    random9 = gs.random_graph_state(9, p_edge=0.05)
    random24 = gs.random_graph_state(24, p_edge=0.05)

    graph_smoothness_test = GraphOpMetricSmoothnessTest(
        [met.ExactGED], [("linear3", linear3), ("linear9", linear9), ("linear24", linear24),
                         ("random3", random3), ("random9", random9), ("random24", random24)],
        folder_name='metric-benchmarking-graph-smoothness-exact-GED'
    )

    df_test = graph_smoothness_test.metric_sensitivity_to_op()
    print(df_test)
    #
    # filter_test = {"graph": "linear3", "op": ["local_complementation", "merge", "add_edge"]}
    # print(graph_smoothness_test.filter_op_sensitivity(filter_test))
    # info_summary = graph_smoothness_test.get_stats_op_sensitivity2(filter_test)
    # for key, val in info_summary.items():
    #     print(f"For category {key}")
    #     print(val)
