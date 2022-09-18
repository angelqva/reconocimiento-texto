"""
Clase de conjunto de datos que se ampliará con clases específicas de conjuntos de datos.
"""


class Dataset:
    """Clase abstracta simple para conjuntos de datos."""

    def download(self):
        raise NotImplementedError(
            "Esto es una clase abstracta. ¡El método aún no está implementado!")

    def load_data(self):
        raise NotImplementedError(
            "Esto es una clase abstracta. ¡El método aún no está implementado!")
