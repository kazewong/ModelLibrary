from abc import ABC, abstractmethod
from jaxtyping import Array

class FeatureExtractor(ABC):

    @abstractmethod
    def __init__(self):
        pass

    @abstractmethod
    def extract_features(self,
                         data: Array) -> Array:
        """
        
        The first dimension should be number of patches and the second dimension should be size of embedding.
        """
        pass

    @abstractmethod
    def embed(self,
            data: Array,) -> Array:
        pass

class ImageFeatureExtractor(FeatureExtractor):

    def __init__(self):
        super().__init__()

    def extract_features(self, data):
        pass

class SeriesFeatureExtractor(FeatureExtractor):
    
    def __init__(self):
        super().__init__()

    def extract_features(self, data):
        pass

class TextFeatureExtractor(FeatureExtractor):

    def __init__(self):
        super().__init__()

    def extract_features(self, data):
        pass