from abc import ABC, abstractclassmethod

class BaseSampler(ABC):

    def __init__(self):
        super().__init__()

    @abstractclassmethod
    def sample(self):
        pass
