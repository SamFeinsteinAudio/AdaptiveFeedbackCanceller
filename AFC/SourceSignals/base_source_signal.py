from abc import ABC, abstractmethod
class SourceSignal(ABC):

    @abstractmethod
    def get_next_sample(self, scaling=1.0, **kwargs):
        pass

    @abstractmethod
    def get_buffer(self, num_samples:int):
        pass


