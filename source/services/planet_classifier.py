import typing as tp

import numpy as np
import onnxruntime as ort

from source.services.preprocess_utils import preprocess_image


class PlanetClassifier:

    def __init__(self, config: tp.Dict):
        self._model_path = config['model_path']

        self._ort_session = ort.InferenceSession(
            self._model_path,
            providers=['CPUExecutionProvider'],
        )
        self._classes: np.ndarray = np.array(
            [
                'agriculture',
                'clear',
                'road',
                'primary',
                'blooming',
                'blow_down',
                'water',
                'partly_cloudy',
                'bare_ground',
                'artisinal_mine',
                'cloudy',
                'habitation',
                'selective_logging',
                'cultivation',
                'haze',
                'conventional_mine',
                'slash_burn',
            ],
        )
        self._size: tp.Tuple[int, int] = (224, 224)
        self._threshold: float = 0.5

    @property
    def classes(self) -> tp.List:
        return list(self._classes)

    def predict(self, image: np.ndarray) -> tp.List[str]:
        """Предсказание списка жанров.

        :param image: RGB изображение;
        :return: список жанров.
        """
        return self._postprocess_predict(self._predict(image))

    def predict_proba(self, image: np.ndarray) -> tp.Dict[str, float]:
        """Предсказание вероятностей принадлежности к жанрам.

        :param image: RGB изображение.
        :return: словарь вида `жанр фильма`: вероятность.
        """
        return self._postprocess_predict_proba(self._predict(image))

    def _predict(self, image: np.ndarray) -> np.ndarray:
        """Предсказание вероятностей.

        :param image: RGB изображение;
        :return: вероятности после прогона модели.
        """
        batch = preprocess_image(image, self._size)

        ort_inputs = {self._ort_session.get_inputs()[0].name: batch}
        ort_outputs = self._ort_session.run(None, ort_inputs)[0][0]

        return 1 / (1 + np.exp(-ort_outputs))

    def _postprocess_predict(self, predict: np.ndarray) -> tp.List[str]:
        """Постобработка для получения списка жанров.

        :param predict: вероятности после прогона модели;
        :return: список жанров.
        """
        return self._classes[predict > self._threshold].tolist()

    def _postprocess_predict_proba(self, predict: np.ndarray) -> tp.Dict[str, float]:
        """Постобработка для получения словаря с вероятностями.

        :param predict: вероятности после прогона модели;
        :return: словарь вида `жанр фильма`: вероятность.
        """
        sorted_idxs = reversed(predict.argsort())
        return {self._classes[i]: float(predict[i]) for i in sorted_idxs}
