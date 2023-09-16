from typing import Dict, List

import cv2
import numpy as np

from source.services.planet_classifier import PlanetClassifier
# from src.services.rotate_classifier import RotateClassifier, NOT_ROTATE_CODE


class PlanetAnalytics:

    def __init__(self, planet_classifier: PlanetClassifier):
        self._planet_classifier = planet_classifier

    @property
    def planets(self):
        return self._planet_classifier.classes

    def predict(self, image: np.ndarray) -> List[str]:
        """Предсказания списка жанров по постеру.

        :param image: входное RGB изображение;
        :return: список жанров.
        """

        return self._planet_classifier.predict(image)

    def predict_proba(self, image: np.ndarray) -> Dict[str, float]:
        """Предсказание вероятностей принадлежности постера к жанрам.

        :param image: входное RGB изображение;
        :return: словарь вида `жанр фильма`: вероятность.
        """

        return self._planet_classifier.predict_proba(image)
