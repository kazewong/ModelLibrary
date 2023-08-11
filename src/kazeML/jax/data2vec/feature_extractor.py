from abc import ABC, abstractmethod

class FeatureExtractor(ABC):

    @abstractmethod
    def __init__(self):
        pass

    @abstractmethod
    def extract_features(self, data):
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