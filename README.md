## Service

Решение задачи мультилейбл классификации изображений со спутника вторая часть (сервис).

### Service link

[Here](http://91.206.15.25:8877/docs) is the link to the `docs` section, you can use it to access all the API.

### How to use service

#### List of all classes
Open the `planet/planets` section. Push `Try it out` and `Execute`. You will see the list with all available classes in the `Response body` section.
It would look like this:

    ```
    {
        "planets": [
            "agriculture",
            "clear",
            "road",
            "primary",
            "blooming",
            "blow_down",
            "water",
            "partly_cloudy",
            "bare_ground",
            "artisinal_mine",
            "cloudy",
            "habitation",
            "selective_logging",
            "cultivation",
            "haze",
            "conventional_mine",
            "slash_burn"
        ]
    }
    ```

#### Predict
Open the `planet/predict` section. Push `Try it out`, `Choose file`. You can use the `tests/images/image.jpg` image as an example. Then push `Execute`.
In the `Response body` you will see something like this:

    ```
    {
        "planets": [
            "agriculture",
            "primary",
            "water"
        ]
    }
    ```

It is all the classes predicted for the image.
#### Predict_proba
Open the `planet/predict_proba` section. Push `Try it out`, `Choose file`. You can use the `tests/images/image.jpg` image as an example. Then push `Execute`.
In the `Response body` you will see something like this:

    ```
    {
        "water": 0.9918007254600525,
        "agriculture": 0.9864889979362488,
        "primary": 0.5149545669555664,
        "selective_logging": 0.3772282302379608,
        "haze": 0.373235285282135,
        "habitation": 0.07654610276222229,
        "cultivation": 0.03305388242006302,
        "artisinal_mine": 0.014107238501310349,
        "blow_down": 0.009212978184223175,
        "road": 0.006693960167467594,
        "blooming": 0.005060070659965277,
        "bare_ground": 0.002448877552524209,
        "slash_burn": 0.00235483655706048,
        "conventional_mine": 0.0006055029225535691,
        "partly_cloudy": 0.0005943151190876961,
        "clear": 0.0004459354095160961,
        "cloudy": 0.0001889267296064645
    }
    ```

These are the probabilities of all the classes for the given image.

### How to run service localy

Install the requirements with terminal command:

```make install```

Generate ssh key and make sure that in the `.dvc/config` in the `['remote "svr"']` section you have the 

```keyfile = /path/to/your/private_key```

You can do it manually or by typing the following command in terminal:

```dvc remote modify svr keyfile /path/to/your/private_key```

Now you can get the model weights with

```make download_weights```

You should see the weights in `weights/onnx_model.onnx`.

#### Using the Python
You can run in the terminal:

```make run_app```

The app will be running at `0.0.0.0:APP_PORT` where `APP_PORT` you can find in the `Makefile`. You can connect to app in your browser with `0.0.0.0:APP_PORT/docs`.

#### Using the Docker

```make build```

Now you have the image `$(DOCKER_IMAGE):$(DOCKER_TAG)` where `DOCKER_IMAGE` and `DOCKER_TAG` are deffined in `Makefile`. 
You can run it with

```docker run --name my-container -d -p 8070:$(APP_PORT) $(DOCKER_IMAGE):$(DOCKER_TAG)```

You can use different name instead of the `my-container` and differnet port instead of the `8070`. This is the port where the service will be available if you go to the browser now: `http://0.0.0.0:8070/docs`. 

You can use `docker ps` command to see these containers and `docker images` to see the images.

To stop and remove the container use:

```docker stop my-container && docker rm my-container```

To remove image:

```docker image rm $(DOCKER_IMAGE):$(DOCKER_TAG)```

### Running tests

```make run_all_tests```
