from dependency_injector import containers, providers

from source.services.planet_classifier import PlanetClassifier


class AppContainer(containers.DeclarativeContainer):
    config = providers.Configuration()

    planet_classifier = providers.Singleton(
        PlanetClassifier,
        config=config.services.planet_classifier,
    )
