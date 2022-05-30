"""

"""
from abc import ABC


class StateRepresentationBase(ABC):
    """
    Base class for state representation
    """
    def __init__(self, data, *args, **kwargs):
        """
        Construct an empty state representation
        :param data: some input data about the state
        """
        self._data = data

    def __str__(self):
        return f"{self.__class__.__name__}\n{self._data}"

    @property
    def data(self):
        return self._data

    @data.setter
    def data(self, data):
        self._data = data

