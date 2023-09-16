from dependency_injector import containers, providers

from source.services.planet_analysis import PlanetAnalytics
from source.services.planet_classifier import PlanetClassifier


class AppContainer(containers.DeclarativeContainer):
    config = providers.Configuration()

    planet_classifier = providers.Factory(
        PlanetClassifier,
        config=config.services.planet_classifier,
    )

    planet_analytics = providers.Singleton(
        PlanetAnalytics,
        planet_classifier=planet_classifier,
    )
