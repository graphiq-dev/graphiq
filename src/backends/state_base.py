"""
The state representation base summarizes the methods which should be available for every quantum state representation
subclass

Currently planned subclasses are:
1. DensityMatrix
2. Graph
3. Stabilizer
"""
from abc import ABC


class StateRepresentationBase(ABC):
    """
    Base class for state representation
    """
    def __init__(self, data, *args, **kwargs):
        """
        Construct an empty state representation base

        :param data: some input data describing the state
        :type data: any
        :return: function returns nothing
        :rtype: None
        """
        self._data = data

    def __str__(self):
        """
        This defines the string representation (i.e. what gets output by the "print" function) of the class

        :return: string representation of the state representation base class
        :rtype: str
        """
        return f"{self.__class__.__name__}\n{self._data}"

    @property
    def data(self):
        """
        Provides read access to the data of the state representation

        :return: the data describing the state
        :rtype: any
        """
        return self._data

    @data.setter
    def data(self, data):
        """
        Provides write access to the data of the state representation

        :param data: the new data to set as representation data
        :type data: any
        :return: function returns nothing
        :rtype: None
        """
        self._data = data
