import numpy as np
import pandas as pd


class SolverResult:
    """
    Class to keep track of the solver result
    """

    def __init__(self, circuit_list, properties=None):
        self._data = {
            "circuit": circuit_list,
        }

        if properties and type(properties) == list:
            for p in properties:
                self._data[p] = [0] * len(circuit_list)

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
        if type(value) is not list:
            raise TypeError(f"Data should be a list or numpy array")
        else:
            if len(value) != len(self._data["circuit"]):
                raise ValueError(
                    f"length of data provided must match number of circuits"
                )

        self._data[key] = value

    def append(self, data):
        return

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
        :return: circuit
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
        :return: list of index
        :rtype: list
        """

        if column not in self._data:
            raise ValueError(f"Column {column} can not be found")

        r_list = []

        for i, v in enumerate(self._data[column]):
            if v == value:
                r_list.append(i)
        return r_list

    def to_df(self):
        df = pd.DataFrame(data=self._data)

        return df

