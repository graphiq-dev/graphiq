import networkx as nx
import numpy as np
import json
import warnings
import pandas as pd


class SolverResult:
    """
    Class to keep track of the solver result.

    The data structure of the class is a simple table structure format. The data is a dictionary with key of the dict is
    the name of the column and the value is a list of data of that column.
    """

    def __init__(self, circuit_list, properties=None):
        self._data = {
            "circuit": circuit_list,
            "circuit_id": [f"c{i}" for i in range(len(circuit_list))],
        }
        self._properties = [] if properties is None else properties
        assert isinstance(self._properties, list)
        for p in properties:
            self._data[p] = [None] * len(circuit_list)

    def __len__(self):
        """
        Return the size of the result

        :return: size of the class
        :rtype: int
        """
        return len(self._data["circuit"])

    def __getitem__(self, key):
        """
        Get the column of the result

        :param key: name of the column
        :type key: str
        :return: the column in the result class
        :rtype: list
        """
        if key not in self._data:
            raise ValueError(f"Can not find property {key} in data")
        return self._data[key]

    def __str__(self):
        """
        Return string that is print in the print() function. It is helpful to inspect the result

        :return: return string
        :rtype: str
        """
        r_str = ""

        for i in range(len(self._data["circuit"])):
            for p in self._data:
                r_str += f"{self._data[p][i]} "
            r_str += "\n"
        return r_str

    def __setitem__(self, key, value):
        """
        Function to set a column of the result class. It is also used to add new column

        :param key: name of the column
        :type key: str
        :param value: list of value of the column
        :type value: list
        :return: nothing
        :rtype: None
        """
        if isinstance(type, (list, np.ndarray)):
            raise TypeError(f"Data should be a list or numpy array")
        else:
            if len(value) != len(self._data["circuit"]):
                raise ValueError(
                    f"length of data provided must match number of circuits"
                )

        self._data[key] = value

    def add_properties(self, new_property):
        """

        :param new_property:
        :type new_property:
        :return:
        :rtype:
        """
        assert isinstance(new_property, str)
        if new_property not in self._properties:
            self._properties.append(new_property)
            self._data[new_property] = [None] * len(self._data["circuit"])

    def get_index_data(self, index):
        """
        Get index data in a form of dictionary.

        :param index: index of the row
        :type index: int
        :return: data of the index
        :rtype: dict
        """
        data = {}

        for d in self._data:
            value = self._data[d][index]
            data[d] = value
        return data

    def get_circuit_index(self, index):
        """
        Get the circuit of the index row

        :param index: index of the row
        :type index: int
        :return: the circuit corresponding to the provided index
        :rtype: CircuitDAG
        """
        return self._data["circuit"][index]

    def get_index_with_column_value(self, column: str, value):
        """
        Get the index list of the row that the value of column match the value provided

        :param column: name of the column
        :type column: str
        :param value: value to check
        :type value: object
        :return: list of all indices that match the value
        :rtype: list
        """

        if column not in self._data:
            raise ValueError(f"Column {column} can not be found")

        r_list = []

        for i, v in enumerate(self._data[column]):
            if v == value:
                r_list.append(i)
        return r_list

    def sort_by(self, prop):
        """
        Sort the results by a given property

        :param prop: a property in the list of properties of the result
        :type prop: str
        :return: Nothing
        :rtype: None
        """
        data_dict = self._data
        n_result = len(self)
        p_index = list(data_dict.keys()).index(prop)
        data_tuple_list = [
            tuple(x[i] for x in data_dict.values()) for i in range(n_result)
        ]
        sorted_data_tuple = sorted(data_tuple_list, key=lambda x: x[p_index])
        for j, p in enumerate(data_dict.keys()):
            self._data[p] = [sorted_data_tuple[i][j] for i in range(n_result)]

    def to_df(self):
        """
        Function to convert SolverResult to pandas DataFrame
        :return: dataframe
        :rtype: pandas.DataFrame
        """
        df = pd.DataFrame(data=self._data)

        return df

    def save2json(self, path, filename):
        data_dict = self._data.copy()
        data_dict["g"] = [list(nx.to_edgelist(g)) for g in self._data["g"]]
        data_dict["circuit"] = [c.to_openqasm() for c in self._data["circuit"]]
        with open(f"{path}/{filename}.json", "w") as f:
            json.dump(data_dict, f)

    def load_json(self, path, filename):
        with open(f"{path}/{filename}.json", "r") as f:
            loaded_dict = json.load(f)
        for properties in loaded_dict.keys():
            if properties == "g":
                self._data["g"] = [nx.from_edgelist(g) for g in loaded_dict["g"]]
            else:
                self._data[properties] = loaded_dict[properties]
        warnings.warn(
            "The circuits in the loaded result object are now openqasm strings!"
        )
        diff = (
            set(self.properties) - set(loaded_dict.keys()) - {"circuit", "circuit_id"}
        )
        if diff:
            warnings.warn("")
        # TODO: from openqasm to circuit object
        # self._data['circuit'] = [QuantumCircuit.from_qasm_str(c) for c in loaded_dict['circuit']]
