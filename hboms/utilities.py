from pygments import highlight  # type: ignore
from pygments.lexers import StanLexer  # type: ignore
from pygments.formatters import HtmlFormatter  # type: ignore
from IPython.display import HTML, display  # type: ignore
from typing import Union, Optional, Tuple, TypeVar, Any
import numpy as np
import re

## TODO: make IPyton import optional. Remove from dependencies


def indent(text: str, n: int) -> str:
    """indent a piece of text with spaces"""
    lines = text.split("\n")
    ind = " " * n
    ind_lines = [ind + line for line in lines]
    return "\n".join(ind_lines)


## TODO: Ipython has a display code feature


def show_stan_model(
    code: str, lines: Optional[Tuple[int, int]] = None, line_numbers: bool = False
) -> None:
    """
    Show stan model code with syntax highlighting in Jupyter notebooks
    """
    lns = "inline" if line_numbers else False
    f = 1  ## fist line number
    if lines is not None:
        f, l = lines
        code = "\n".join(code.split("\n")[f - 1 : l])
    formatter = HtmlFormatter(full=True, linenos=lns, linenostart=f)
    display(HTML(highlight(code, StanLexer(), formatter)))


def apply_padding(
    xss: list, sdtype: str = "real", val: Union[int, float] = 0
) -> np.ndarray:
    """
    Apply padding to elements xs of xss such that all elements of
    the result have the same shape.
    TODO: lots of testing

    Parameters
    ----------
    xss : list
        list with potentially distinctly shaped elements.
    sdtype : str, optional
        Stan type of the base variable. The default is "real".
    val : Union[int, float], optional
        value used for padding. The default is 0.

    Returns
    -------
    xss_padded : np.ndarray
        A numpy array containing xxs, with val added where needed.

    """
    n = len(xss)
    shapes = [np.array(xs).shape for xs in xss]
    max_shp_len = np.max([len(shp) for shp in shapes])
    ext_shapes = [
        shp + tuple([0 for _ in range(max_shp_len - len(shp))]) for shp in shapes
    ]
    max_shp = tuple(np.max(ext_shapes, axis=0))
    ## map stan types to python types
    map_types = {"real": float, "int": int}
    xss_padded = np.full((n, *max_shp), val, dtype=map_types.get(sdtype))
    for i, xs in enumerate(xss):
        np.add.at(
            xss_padded[i], tuple(slice(0, s) for s in ext_shapes[i]), np.array(xs)
        )
    return xss_padded


GenericType = TypeVar("GenericType")


def flatten(xss: list[list[GenericType]]) -> list[GenericType]:
    return [x for xs in xss for x in xs]


## TODO restrict the type of xs


def unique(xs: list[GenericType]) -> list[GenericType]:
    return sorted(list(set(xs)))


def find_used_names(code: str, names: list[str]) -> list[str]:
    """
    find all names in a piece of code that are words.
    a word is a sting of letters, numbers and underscores
    """
    results = [re.search(r"\b" + name + r"\b", code) for name in names]
    return [name for name, res in zip(names, results) if res is not None]


def merge_timeseries(
    txss: list[list[tuple[float, Any]]]
) -> tuple[list[float], list[list[Any]]]:
    """
    merge timeseries with distinct observation times into
    a single timeseries with missing values.

    TODO:
    - censoring codes
    - what do we want: indices in the array of time points,
      or potentially sparse observations. Or both options?
    """
    time_points = unique([tx[0] for txs in txss for tx in txs])
    xss = []
    for txs in txss:
        xs = [None for _ in time_points]
        for t, x in txs:
            i = time_points.index(t)
            xs[i] = x
        xss.append(xs)
    return time_points, xss
