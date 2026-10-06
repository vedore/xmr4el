import tempfile
import shutil
import json
import uuid

import numpy as np

from pathlib import Path
from scipy.sparse import csr_matrix
from scipy.sparse import save_npz, load_npz


class TempVarStore:
    model_dir: Path = Path(tempfile.mkdtemp(prefix="x_model_dir_"))

    @classmethod
    def save_model_temp(cls, var) -> str:
        """Persist a temporary variable (list, np.ndarray, dict, csr_matrix) and return its folder path."""
        sub_dir = cls.model_dir / uuid.uuid4().hex
        sub_dir.mkdir(parents=True, exist_ok=True)

        if isinstance(var, csr_matrix):
            save_npz(sub_dir / "csr.npz", var, compressed=True)
            (sub_dir / "meta.json").write_text(json.dumps({"type": "csr"}))
        elif isinstance(var, np.ndarray):
            np.save(sub_dir / "array.npy", var)
            (sub_dir / "meta.json").write_text(json.dumps({"type": "ndarray"}))
        elif isinstance(var, list):
            (sub_dir / "list.json").write_text(json.dumps(var))
            (sub_dir / "meta.json").write_text(json.dumps({"type": "list"}))
        elif isinstance(var, dict):
            (sub_dir / "dict.json").write_text(json.dumps(var))
            (sub_dir / "meta.json").write_text(json.dumps({"type": "dict"}))
        else:
            raise TypeError(f"Unsupported type: {type(var)}")

        return str(sub_dir)

    @classmethod
    def load_model_temp(cls, path):
        """Load a variable previously saved with save_model_temp()."""
        p = Path(path)
        meta = json.loads((p / "meta.json").read_text()) if (p / "meta.json").exists() else None
        t = (meta or {}).get("type")

        if not t:
            # fallback: infer by file present
            if (p / "csr.npz").exists():
                t = "csr"
            elif (p / "array.npy").exists(): 
                t = "ndarray"
            elif (p / "list.json").exists(): 
                t = "list"
            elif (p / "dict.json").exists(): 
                t = "dict"
            else:
                raise FileNotFoundError(f"No known artifact in {p}")

        if t == "csr":
            return load_npz(p / "csr.npz")
        elif t == "ndarray":
            return np.load(p / "array.npy", allow_pickle=False)
        elif t == "list":
            return json.loads((p / "list.json").read_text())
        elif t == "dict":
            return json.loads((p / "dict.json").read_text())
        else:
            raise ValueError(f"Unknown saved type: {t}")

    @classmethod
    def delete_from_path(cls, path) -> None:
        """Delete the saved variable folder at `path`."""
        shutil.rmtree(Path(path), ignore_errors=True)

    @classmethod
    def delete_model_temp(cls) -> None:
        """Remove all temporary variables from disk and reset the temp root."""
        shutil.rmtree(cls.model_dir, ignore_errors=True)
        cls.model_dir = Path(tempfile.mkdtemp(prefix="x_model_dir_"))