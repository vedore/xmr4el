"""Dataset Setup model (tui.md §5.4): logical datasets recognized from configs/experiments/*.json, dataset directories
under datasets/ and the user's registered datasets; cheap, bounded file checks; validation of one selection. No UI.
Stdlib only.

A recognized path is a suggestion, never proof: availability comes from `inspect` only. Relative paths are relative
to the project root (the runner's working directory), never the TUI's.
"""
import json
import os
import uuid
from dataclasses import dataclass
from pathlib import Path

from xmr4el_tui import runs

SPECS_DIR = Path("configs", "experiments")
DATASETS_DIR = Path("datasets")
REGISTRY = Path("outputs", "tui", "datasets.json")  # datasets the user registered (outputs/ is not tracked)
SPLITS = {"dev": "Development", "test": "Test"}       # evaluation roles = runner split names
SAMPLE = 64 * 1024  # bytes a format check reads: never the whole corpus
FORMAT = "pubtator"  # the only input format the tracked runner reads
READY, MISSING, INVALID, REVIEW = "Ready", "Missing files", "Invalid mapping", "Needs review"
STATUS = {"missing": MISSING, "invalid": INVALID, "review": REVIEW}  # worst first


@dataclass(frozen=True)
class Candidate:
    path: str
    sources: tuple = ()  # spec file names, "directory", "registered"


@dataclass(frozen=True)
class EvalInput:
    split: str           # runner split name: dev | test
    path: str | None
    enabled: bool


@dataclass(frozen=True)
class Selection:
    """The files one experiment uses: what Dataset Setup edits and the draft confirms."""
    key: str             # Dataset.key
    name: str
    variant: str = ""
    train: str | None = None
    evals: tuple = ()    # EvalInput, one per split
    format: str = FORMAT

    def enabled(self):
        return tuple(e for e in self.evals if e.enabled)

    def evaluation(self, split):
        return next((e for e in self.evals if e.split == split), EvalInput(split, None, False))

    def with_eval(self, split, **changes):
        old = self.evaluation(split)
        new = EvalInput(split, changes.get("path", old.path), changes.get("enabled", old.enabled))
        evals = [new if e.split == split else e for e in self.evals]
        return Selection(self.key, self.name, self.variant, self.train,
                         tuple(evals if split in [e.split for e in self.evals] else [*evals, new]), self.format)


@dataclass(frozen=True)
class Dataset:
    key: str             # its directory as recorded (datasets/BC5CDR/chemical), or local:<id> when registered
    name: str
    variant: str = ""
    train: tuple = ()    # Candidate per recognized training file (train.pubtator and train_plus_ctd.pubtator stay two)
    dev: tuple = ()
    test: tuple = ()
    registered: bool = False

    @property
    def title(self):
        return f"{self.name} — {self.variant}" if self.variant else self.name or "New dataset"

    def candidates(self, role):
        return getattr(self, role) if role in ("train", *SPLITS) else ()

    def sources(self):
        return sorted({s for role in ("train", *SPLITS) for c in self.candidates(role) for s in c.sources})

    def default(self):
        """Recognized mapping: a file only where exactly one is recognized (training: or exactly one a spec names);
        Development enabled when recognized, Test never enabled by default."""
        named = [c for c in self.train if set(c.sources) - {"directory"}]
        train = self.train if len(self.train) == 1 else named if len(named) == 1 else ()
        one = lambda role: self.candidates(role)[0].path if len(self.candidates(role)) == 1 else None
        return Selection(self.key, self.name, self.variant, train[0].path if train else None,
                         (EvalInput("dev", one("dev"), bool(self.dev)), EvalInput("test", one("test"), False)))


def norm(path):
    return os.path.normpath(path)


def resolve(root, path):
    """Absolute path of a recorded `path` (relative: to the project root)."""
    return Path(root, os.path.expanduser(path))


def shown(root, path):
    """`path` as recorded: relative to the project root when inside it, else absolute."""
    path = Path(norm(resolve(root, path)))
    try:
        return path.relative_to(Path(root).resolve()).as_posix()
    except ValueError:
        try:
            return path.relative_to(Path(root)).as_posix()
        except ValueError:
            return str(path)


def read_json(path):
    try:
        return json.loads(Path(path).read_text())
    except (OSError, ValueError):
        return None


def split_role(split):
    """dev / test from a spec's split name ("dev", "dev_knn0" -> dev), else None: a role is read from the split the
    spec declares, never guessed from a file name."""
    if not isinstance(split, str):
        return None
    return next((r for r in SPLITS if split == r or split.startswith(f"{r}_")), None)


def naming(directory):
    """(name, variant) of a dataset directory: datasets/BC5CDR/chemical -> (BC5CDR, Chemical)."""
    parts = Path(directory).parts
    rel = parts[1:] if parts[:1] == (DATASETS_DIR.name,) and len(parts) > 1 else (Path(directory).name,)
    variant = " / ".join(p.capitalize() if p.isalpha() and p.islower() else p for p in rel[1:] if p != "data")
    return rel[0].replace("_", " "), variant


def discover(root):
    """Recognized datasets, by directory: training and evaluation files every experiment spec names (eval-only specs
    too; model configs in the same folder are skipped), plus the split files (`runs.split_files`, `runs.training_files`)
    of those directories and of every datasets/ directory with a dev or test file (`runs.datasets`). Nothing outside
    the project is scanned; files are not opened."""
    root, found = Path(root), {}

    def add(directory, role, path, source):
        files = found.setdefault(norm(directory), {"train": {}, "dev": {}, "test": {}})[role]
        files.setdefault(norm(path), []).append(source)

    for f in sorted(Path(root, SPECS_DIR).glob("*.json")):
        spec = read_json(f)
        evals = spec.get("evals") if isinstance(spec, dict) else None
        if not isinstance(evals, list):
            continue
        top = spec.get("train_path") if isinstance(spec.get("train_path"), str) else None
        if top:
            add(Path(top).parent, "train", top, f.name)
        for e in evals:
            if not isinstance(e, dict):
                continue
            train = e.get("train_path") if isinstance(e.get("train_path"), str) else top
            test, role = e.get("test_path"), split_role(e.get("split"))
            test = test if isinstance(test, str) else None
            home = Path(train or test or ".").parent
            if train:
                add(home, "train", train, f.name)
            if isinstance(test, str) and role:
                add(home, role, test, f.name)
    for d in sorted({norm(d) for d in runs.datasets(root)} | set(found)):
        files = runs.split_files(root, d)
        for path in runs.training_files(root, d):
            add(d, "train", path, "directory")
        for role in SPLITS:
            if role in files:
                add(d, role, files[role], "directory")
    out = [Dataset(d, *naming(d), **{role: tuple(Candidate(p, tuple(dict.fromkeys(s))) for p, s in sorted(c.items()))
                                     for role, c in roles.items()}) for d, roles in found.items()]
    return sorted(out, key=lambda ds: (ds.name.casefold(), ds.variant.casefold()))


# --- registered datasets ------------------------------------------------------------------------------------------
def registered(root):
    """(datasets, problem): the user's registered datasets, and why the registry could not be read (None if fine)."""
    path = Path(root, REGISTRY)
    if not path.exists():
        return [], None
    data = read_json(path)
    entries = data.get("datasets") if isinstance(data, dict) else None
    if not isinstance(entries, list):
        return [], f"{REGISTRY} is not a dataset registry: registering is disabled until it is fixed or removed"
    out = []
    for e in entries:
        if isinstance(e, dict) and isinstance(e.get("id"), str) and isinstance(e.get("name"), str):
            files = {r: (Candidate(e[r], ("registered",)),) if isinstance(e.get(r), str) else ()
                     for r in ("train", *SPLITS)}
            out.append(Dataset(f"local:{e['id']}", e["name"], "", registered=True, **files))
    return out, None


def register(root, selection):
    """Save `selection` as a registered dataset (new, or replacing the one with its key); returns its key. Only the
    registry file is written. Raises ValueError when the registry exists but cannot be read (it is not overwritten)."""
    path = Path(root, REGISTRY)
    data = read_json(path) if path.exists() else {"datasets": []}
    if not isinstance(data, dict) or not isinstance(data.get("datasets"), list):
        raise ValueError(f"{REGISTRY} is not a dataset registry: not overwritten")
    rid = selection.key.removeprefix("local:") if selection.key.startswith("local:") else uuid.uuid4().hex[:8]
    entry = {"id": rid, "name": selection.name.strip(), "train": selection.train,
             **{s: selection.evaluation(s).path for s in SPLITS}}
    data["datasets"] = [e for e in data["datasets"] if not (isinstance(e, dict) and e.get("id") == rid)] + [entry]
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(path.name + ".tmp")
    tmp.write_text(json.dumps(data, indent=2))
    os.replace(tmp, path)
    return f"local:{rid}"


# --- checks -------------------------------------------------------------------------------------------------------
_checked = {}  # path + stat identity -> inspect result. shortcut: unpruned; prune if long sessions inspect many edits


def header(line):
    """(id, t|a) of a PubTator title / abstract line, else None (as xmr4el.data.readers.pubtator_header)."""
    parts = line.split("|", 2)
    return tuple(parts[:2]) if len(parts) == 3 and parts[1] in ("t", "a") and "\t" not in parts[0] else None


def inspect(root, path):
    """(None | "missing" | "invalid", message) from a stat and the first SAMPLE bytes: exists, is a readable text
    file, and reads as PubTator (title before abstract and annotations, tab-separated annotation lines). Not a full
    parse: launch-time checks stay with the runner. Cached by file identity, change times and size."""
    p = resolve(root, path)
    try:
        st = p.stat()
    except FileNotFoundError:
        return "missing", "not found"
    except OSError as err:
        return "invalid", f"unreadable ({err.strerror})"
    if not p.is_file():
        return "invalid", "is not a file"
    key = (str(p), st.st_dev, st.st_ino, st.st_mtime_ns, st.st_ctime_ns, st.st_size)
    if key not in _checked:
        _checked[key] = sniff(p, st.st_size)
    return _checked[key]


def sniff(p, size):
    try:
        with open(p, "rb") as f:
            head = f.read(SAMPLE)
    except OSError as err:
        return "invalid", f"unreadable ({err.strerror})"
    whole = len(head) >= size
    if not head.strip():
        return "invalid", "is empty"
    try:
        text = head.decode("utf-8")
    except UnicodeDecodeError as err:
        if whole or err.start < len(head) - 3:  # a character cut by the sample's end is not an error
            return "invalid", "is not UTF-8 text: not PubTator"
        text = head[:err.start].decode("utf-8")
    if "\x00" in text:
        return "invalid", "is binary: not PubTator"
    lines = text.split("\n") if whole else text.split("\n")[:-1]  # the sample's last line may be cut
    title = annotations = 0
    doc_id, title_text, abstract = None, "", ""
    for n, raw in enumerate(lines, 1):
        line = raw.rstrip("\r")
        if not line.strip():
            continue
        h = header(line)
        if h is None and "\t" not in line or h is None and not title or h and h[1] == "a" and not title:
            return "invalid", f"is not PubTator (line {n} is not '<id>|t|<title>', '<id>|a|...' or a tab-separated " \
                              "annotation)"
        title += bool(h and h[1] == "t")
        if h:
            if h[1] == "t":
                doc_id, title_text, abstract = h[0], line.split("|", 2)[2], ""
            elif h[0] != doc_id:
                return "invalid", f"is not PubTator (line {n}: abstract does not match the document title)"
            else:
                abstract = line.split("|", 2)[2]
            continue
        parts = line.split("\t")
        if len(parts) >= 3 and parts[1].isdigit() and parts[2].isdigit():
            length = len(title_text) + (1 + len(abstract) if abstract else 0)
            if parts[0] != doc_id or not 0 <= int(parts[1]) < int(parts[2]) <= length:
                return "invalid", f"is not PubTator (line {n}: annotation does not fit its document)"
            # Like the loader, ignore annotations without a usable concept; they cannot make a file Ready.
            annotations += len(parts) >= 6 and all(parts[5].split("|")) and parts[5] != "-1"
    if not title:
        return "invalid", "is not PubTator (no '<id>|t|<title>' line)"
    if whole and not annotations:
        return "invalid", "has no PubTator annotation lines"
    return None, "ok"


def same_file(root, a, b):
    pa, pb = resolve(root, a), resolve(root, b)
    try:
        return os.path.samefile(pa, pb)
    except OSError:
        return norm(pa) == norm(pb)


def validate(root, dataset, sel, taken=()):
    """[(field, kind, message)] for `sel` of `dataset`; field = name | train | evals | <split>, kind = missing |
    invalid | review. Empty = every check this screen makes passed (`taken`: other registered datasets' names)."""
    out = []
    if dataset.registered:
        name = sel.name.strip()
        if not name:
            out.append(("name", "review", "Name required"))
        elif name.casefold() in {t.casefold() for t in taken}:
            out.append(("name", "invalid", f"Name already registered: {name}"))
    if sel.format != FORMAT:
        out.append(("train", "invalid", f"Format {sel.format} is not supported: the experiment runner reads PubTator"))

    def check(field, label, path):
        kind, message = inspect(root, path)
        if kind:
            out.append((field, kind, f"{label} file {message}: {path}"))

    if not sel.train:
        many = len(dataset.train)
        out.append(("train", "review", f"Choose a training file: {many} variants recognized") if many > 1
                   else ("train", "missing", "Training file not set"))
    else:
        check("train", "Training", sel.train)
    enabled = sel.enabled()
    if not enabled:
        out.append(("evals", "review", "Enable at least one evaluation split"))
    names = [e.split for e in enabled]
    if len(set(names)) < len(names):
        out.append(("evals", "invalid", "Duplicate evaluation split: "
                    + ", ".join(sorted({s for s in names if names.count(s) > 1}))))
    for e in enabled:
        label = SPLITS.get(e.split, e.split)
        many = len(dataset.candidates(e.split))
        if not e.path:
            out.append((e.split, "review", f"Choose the {label} file: {many} recognized") if many > 1
                       else (e.split, "missing", f"{label} file not set"))
            continue
        check(e.split, label, e.path)
        if sel.train and same_file(root, sel.train, e.path):
            out.append((e.split, "invalid", f"{label} file is the training file: {e.path}"))
    return out


def status(problems):
    """Ready only when nothing failed; else the worst kind: Missing files, Invalid mapping, Needs review."""
    kinds = {kind for _, kind, _ in problems}
    return next((word for kind, word in STATUS.items() if kind in kinds), READY)
