## Aprendizaje

-   Inicialmente entrenamos todos los modelos con una tasa de aprendizaje constante.
-   En lugar de usar una tasa de aprendizaje constante, implementamos una tasa de aprendizaje cíclica y un buscador de tasas de aprendizaje que proporcionaron un gran impulso en términos de velocidad y precisión para realizar varios experimentos.
-   El aprendizaje de transferencia con resnet-18 tuvo un rendimiento deficiente.
-   A partir de los resultados anteriores de la evaluación de la prueba, podemos ver que el modelo funciona mal en caracteres específicos, ya que puede haber confusión debido a la similitud, como el dígito 1 y la letra l, el dígito 0 y la letra o o O, el dígito 5 y la letra s o S ​​o el dígito 9 y la letra q o Q.
-   Las precisiones en el conjunto de datos del tren son del 78 % en lenet, del 83 % en resnet y del 84 % en personalizado.
-   Las precisiones en el conjunto de datos de val son del 80 % en lenet, 81 % en resnet y 82 % en custom.
-   Las precisiones en el conjunto de datos de prueba son del 62 % en lenet, del 36 % en resnet y del 66 % en personalizado.
-   La arquitectura personalizada funciona bien pero resnet funciona mal (¿Por qué?)
-   Hay mucha brecha en el tren-val y la prueba, incluso cuando la distribución de val es la misma que la distribución de prueba, es decir, el conjunto de val se toma del 10% del conjunto de prueba.
-   Busque nuevas formas de aumentar la precisión

Una breve descripción del proyecto.

## Proyecto Caracteres

    ├── README.md          <- El README de nivel superior para los desarrolladores que utilizan este proyecto.
    ├── data
    │   ├── external       <- Datos de fuentes de terceros.
    │   ├── interim        <- Datos intermedios que han sido transformados.
    │   ├── processed      <- Los conjuntos de datos canónicos finales para el modelado.
    │   └── raw            <- El volcado de datos original e inmutable.
    │
    ├── models             <- Modelos entrenados y serializados, predicciones de modelos o resúmenes de modelos
    │
    |
    ├── src                <- Codigo fuente para usar en este proyecto.
        ├── data
        │   ├── dataset.py
        │   ├── emnist_dataset.py  <- Scripts para descargar o generar datos
        |
        ├── __init__.py            <- Convierte src en un módulo de Python
        |
        ├── models
        │   ├── base_model.py
        │   ├── character_model.py
        |
        ├── networks
        │   ├── custom.py
        │   ├── lenet.py
        │   └── resnet.py
        |
        ├── tests                    <- Scripts para usar modelos climáticos para hacer predicciones
        │   ├── support
        │   │   ├── create_emnist_support_files.py
        │   │   └── emnist
        │   │       ├── 3.png
        │   │       ├── 8.png
        │   │       ├── a.png
        │   │       ├── e.png
        │   │       └── U.png
        │   └── test_character_predictor.py
        ├── training                        <- Scripts para entrenar modelos
        │   ├── character_predictor.py
        │   ├── clr_callback.py
        │   ├── lr_find.py
        │   ├── train_model.py
        │   └── util.py
        ├── util.py
        └── visualization       <- Scripts para crear visualizaciones exploratorias y orientadas a resultados
            └── visualize.py
    ____caracteres.ipynb        <- Jupyter del proyecto probado
    ____PCA_KMEAN.ipynb         <- Reduccion de dimensiones y clusterizacion ejemplo utilizado solo del 0-9

---
