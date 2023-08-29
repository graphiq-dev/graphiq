class QuNode:
    """
    A class that represents a node of qubit(s). It has the option to enable simple redundancy encoding.

    """

    def __init__(self, id_set, redundancy=False):
        """
        Creates a node of qubits

        :param id_set: id for the QuNode (if the QuNode has a single qubit/no redundant encoding) OR
                       id for each qubit of the redundantly encoded QuNode
        :type id_set: frozenset OR int
        :raises ValueError: if the wrong datatype is passed in as id_set
        :return: nothing
        :rtype: None
        """
        self.redundancy = redundancy

        if isinstance(id_set, frozenset):
            self.id = id_set
        elif isinstance(id_set, int):
            self.id = frozenset([id_set])
        else:
            raise ValueError("QuNode only accepts frozenset and int as id.")
        if (not self.redundancy) and len(self.id) > 1:
            raise ValueError("Redundancy encoding is disabled for this QuNode.")

    def count_redundancy(self):
        """
        Return the number of qubits in the redundancy encoding

        :return: the number of qubits in the redundant encoding
        :rtype: int
        """
        if self.redundancy:
            return len(self.id)
        else:
            return 1

    def set_id(self, id_set):
        """
        Allow one to update the IDs of all qubits in the node.

        :param id_set: the new set of ids for the qubits in the node
        :type id_set: frozenset OR int
        :raises ValueError: if id_set is not the desired datatype
        :return: function returns nothing
        :rtype: None
        """
        if (not self.redundancy) and len(id_set) > 1:
            raise ValueError("Redundancy encoding is disabled for this QuNode.")
        if isinstance(id_set, frozenset):
            self.id = id_set
        elif isinstance(id_set, int):
            self.id = frozenset([id_set])
        else:
            raise ValueError("QuNode only accepts frozenset and int as id.")

    def remove_id(self, photon_id):
        """
        Remove the qubit with the specified id from a redundancy encoding.
        It does nothing if the node is not redundantly encoded.

        :param photon_id: id of the qubit to remove from the redundancy encoding
        :type photon_id: int
        :return: True if the photon of the given ID was removed, False otherwise
        :rtype: bool
        """
        if not self.redundancy:
            return False
        if len(self.id) > 1 and photon_id in self.id:
            tmp_set = set(self.id)
            tmp_set.remove(photon_id)
            self.id = frozenset(tmp_set)
            return True

        return False

    def remove_first_id(self):
        """
        Remove the first qubit from the redundancy encoding.
        It does nothing if the node is not redundantly encoded.

        :return: True if a qubit is removed, False otherwise
        :rtype: bool
        """
        if not self.redundancy:
            return False
        if len(self.id) > 1:
            tmp_set = set(self.id)
            tmp_set.pop()
            self.id = frozenset(tmp_set)
            return True

        return False

    def get_id(self):
        """
        Return the id of the node. This may be either an integer ID
        or a frozenset containing all photon IDs in this node

        :return: the photon(s) id(s)
        :rtype: frozenset
        """
        return self.id
