"""Lossless text exports for the CAS notebook (no calculation changes)."""
from pathlib import Path
from dataclasses import is_dataclass, fields
from contextlib import contextmanager, redirect_stdout
import sys
import numpy as np
import pandas as pd

def save_results_txt(path, **results):
    """Write labeled arrays/tables recursively, without display truncation.
    
    Arrays include shape and dtype; >2D arrays are flattened in C order.
    Complex values retain both real and imaginary components."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    def write(out, name, value):
        out.write(f"\n[{name}]\n")
        if is_dataclass(value):
            for field in fields(value):
                write(out, name + "." + field.name, getattr(value, field.name))
        elif isinstance(value, dict):
            for key, item in value.items():
                write(out, name + "." + str(key), item)
        elif isinstance(value, (pd.DataFrame, pd.Series)):
            value.to_csv(out, sep="\t", float_format="%.17g")
        elif isinstance(value, np.ndarray):
            out.write(f"shape={value.shape}; dtype={value.dtype}; order=C\n")
            data = value.reshape(1, -1) if value.ndim < 2 else value.reshape(value.shape[0], -1) if value.size else value
            if value.size:
                np.savetxt(out, data, fmt="%s" if value.dtype.kind not in "fc" else "%.17g", delimiter="\t")
        elif isinstance(value, (list, tuple)):
            for i, item in enumerate(value):
                write(out, name + f"[{i}]", item)
        elif value is None or isinstance(value, (str, int, float, complex, bool, np.generic)):
            out.write(repr(value) + "\n")
        else:
            raise TypeError(f"Unsupported result type at {name}: {type(value)}")
    with path.open("w", encoding="utf-8") as out:
        for name, value in results.items():
            write(out, name, value)

@contextmanager
def result_log(path):
    """Keep notebook console output while also writing it to UTF-8 text."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    console = sys.stdout
    with path.open("w", encoding="utf-8") as output:
        class Tee:
            def write(self, text):
                console.write(text)
                return output.write(text)
            def flush(self):
                console.flush()
                output.flush()
        with redirect_stdout(Tee()):
            yield
