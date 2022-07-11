from src.backends.state_base import StateRepresentationBase


class Stabilizer(StateRepresentationBase):
    def __init__(self, data, *args, **kwargs):
        super().__init__(data, *args, **kwargs)
        raise NotImplementedError("")
