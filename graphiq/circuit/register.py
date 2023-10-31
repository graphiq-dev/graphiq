class Register:
    """
    Register class object, the class includes a dictionary which map the register type as the key and the register array
    as the value.
    """

    def __init__(self, reg_dict, is_multi_qubit: bool = False):
        """
        Constructor for the register class

        :param reg_dict: register dictionary
        :type reg_dict: dict
        :param is_multi_qubit: variable that indicate support for multi-qubit register
        :type is_multi_qubit: bool
        :return: this function returns nothing
        :rtype: None
        """
        # Check empty data
        if not reg_dict:
            raise ValueError("Register dict can not be None or empty")

        # Check if input data is numerical
        for key in reg_dict:
            if not all([isinstance(item, int) for item in reg_dict[key]]):
                raise ValueError("The input data contains non-numerical value")

        # Check if not multi-qubit register but input value more than 1
        for key in reg_dict:
            if reg_dict[key] and set(reg_dict[key]) != {1} and not is_multi_qubit:
                raise ValueError(
                    f"Register is not multi-qubit register but has value more than 1"
                )

        self._registers = reg_dict
        self.is_multi_qubit = is_multi_qubit

    @property
    def register(self):
        return self._registers.copy()

    def __getitem__(self, key):
        return self._registers[key]

    def __setitem__(self, key, value):
        # Check value is numerical
        if not all([isinstance(item, int) for item in value]):
            raise ValueError("The input data contains non-numerical value")

        # Check if not multi-qubit register but has value more than 1
        if value and set(value) != {1} and not self.is_multi_qubit:
            raise ValueError(f"The register only supports single-qubit registers")
        self._registers[key] = value

    @property
    def n_quantum(self):
        q_sum = 0

        for key in self._registers:
            if key != "c":
                q_sum += len(self._registers[key])
        return q_sum

    def add_register(self, reg_type: str, size: int = 1):
        """
        Function that add a quantum/classical register to the register dict

        :param reg_type: 'p' for a photonic quantum register, 'e' for an emitter quantum register,
                         'c' for a classical register
        :type reg_type: str
        :param size: the new register size
        :type size: int
        :raises ValueError: if new_size is not greater than the current register size
        :return: the index number of the added register
        :rtype: int
        """
        if reg_type not in self._registers:
            raise ValueError(
                f"reg_type must be 'e' (emitter qubit), 'p' (photonic qubit), 'c' (classical bit)"
            )
        if size < 1:
            raise ValueError(f"{reg_type} register size must be at least one")
        if size > 1 and not self.is_multi_qubit:
            raise ValueError(
                f"Can not add register of size {size}, multiple qubit register is not supported"
            )
        self._registers[reg_type].append(size)
        return len(self._registers[reg_type]) - 1

    def expand_register(self, reg_type: str, register: int, new_size: int = 1):
        """
        Function to expand quantum/classical registers

        :param register: the register index of the register to expand
        :type register: int
        :param new_size: the new register size
        :type register: int
        :param reg_type: 'p' for a photonic quantum register, 'e' for an emitter quantum register,
                         'c' for a classical register
        :type reg_type: str
        :raises ValueError: if new_size is not greater than the current register size
        :return: this function returns nothing
        :rtype: None
        """
        if reg_type not in self._registers:
            raise ValueError(
                "reg_type must be 'e' (emitter register), 'p' (photonic register), "
                "or 'c' (classical register)"
            )
        if new_size > 1 and not self.is_multi_qubit:
            raise ValueError(
                f"Can not expand register to size {new_size}, multiple qubit register is not supported"
                f"(they must have a size of 1)"
            )
        curr_reg = self._registers[reg_type]
        curr_size = curr_reg[register]

        if new_size <= curr_size:
            raise ValueError(
                f"New register size {new_size} is not greater than the current size {curr_size}"
            )
        curr_reg[register] = new_size

    def next_register(self, reg_type: str, register: int):
        """
        Provides the index of the next register in the provided register. This allows the user to query
        which register they should add next, should they decide to expand the register

        :param reg_type: indicate register type, can be "p", "e", or "c"
        :type reg_type: str
        :param register: the register index {0, ..., N - 1} for N emitter quantum registers
        :type register: int
        :return: the index of the next register
        :rtype: int (non-negative)
        """
        if reg_type not in self._registers:
            raise ValueError(
                "Register type must be 'p' (quantum photonic), 'e' (quantum emitter), or 'c' (classical)"
            )
        return self._registers[reg_type][register]
