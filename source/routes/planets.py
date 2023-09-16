import cv2
import numpy as np
from dependency_injector.wiring import Provide, inject
from fastapi import Depends, File

from source.containers.containers import AppContainer
from source.routes.routers import router
from source.services.planet_analysis import PlanetAnalytics


@router.get('/planets')
@inject
def planets_list(service: PlanetAnalytics = Depends(Provide[AppContainer.planet_analytics])):
    return {
        'planets': service.planets,
    }


@router.post('/predict')
@inject
def predict(
    image: bytes = File(),
    service: PlanetAnalytics = Depends(Provide[AppContainer.planet_analytics]),
):
    img = cv2.imdecode(np.frombuffer(image, np.uint8), cv2.IMREAD_COLOR)
    planets = service.predict(img)

    return {'planets': planets}


@router.post('/predict_proba')
@inject
def predict_proba(
    image: bytes = File(),
    service: PlanetAnalytics = Depends(Provide[AppContainer.planet_analytics]),
):
    img = cv2.imdecode(np.frombuffer(image, np.uint8), cv2.IMREAD_COLOR)
    return service.predict_proba(img)


@router.get('/health_check')
def health_check():
    return 'OK'
