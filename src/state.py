"""
Note: this file should only be used for the state WRAPPER object
      (individual implementations should be placed in backend:
      state_representations.py)
"""
import src.backends.state_representations as srep

# Currently it does nothing other than initialization
class QuantumState:
    """
    Base class for quantum state wrapper
    """
    def __init__(self,state_id,state_properties,state_rep):
        """
        Construct a quantum state
        :param state_id: an identifier for each state
        :param state_properties: a dictionary that contains information about state state_properties
        :param state_rep: a dictionary of available state representations
        """

        self.state_id = state_id
        self.state_properties = state_properties
        self.state_rep = state_rep
    
