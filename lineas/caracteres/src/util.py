"""
Funciones de utilidad.
"""

from concurrent.futures import as_completed, ThreadPoolExecutor
from urllib.request import urlopen, urlretrieve
from tqdm import tqdm


class TqdmUpTo(tqdm):
    """From https://github.com/tqdm/tqdm/blob/master/examples/tqdm_wget.py"""

    def update_to(self, blocks=1, bsize=1, tsize=None):
        """
        bloques: int, opcional
            Número de bloques transferidos hasta el momento [predeterminado: 1].
        bsize : int, opcional
            Tamaño de cada bloque (en unidades tqdm) [predeterminado: 1].
        tsize : int, opcional
            Tamaño total (en unidades tqdm). Si [predeterminado: Ninguno] permanece sin cambios.
        """
        if tsize is not None:
            self.total = tsize
        self.update(blocks * bsize - self.n)


def download_url(url, filename):
    """Descargue un archivo de URL a nombre de archivo, con una barra de progreso."""
    with TqdmUpTo(unit='B', unit_scale=True, unit_divisor=1024, miniters=1) as t:
        urlretrieve(url, filename, reporthook=t.update_to, data=None)
