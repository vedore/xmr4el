"""Model Builder registry (tui.md §5.5): the slots of the hybrid pipeline, the components each slot can load, the
attributes each component takes, and the checks the builder applies to them. No UI. Stdlib only.

The TUI never imports the backend, so the rules here mirror it: defaults are the backend's effective ones
(`Vectorizer.fit`, `DimensionModel.fit`, `Transformer.transform`, `BalancedKMeans.DEFAULTS`,
`ClassifierModel.init_model`, `PREDICT_DEFAULTS`, the `XModel` signature) and the ranges are its validators
(`XModel.__init__`, `check_search`, `BalancedKMeans.check_config`, `JointOvRLogistic`, scikit-learn's TF-IDF / SVD
parameters). tests/test_tui_components.py checks both against the backend and round-trips built configs through
`XModel.load_config`. Only the hybrid recipe (`features` "sapbert_char_context") is built; TF-IDF-only configs are
refused, never converted.

A tree's configuration is one plain JSON dict (the model_config the runner reads). A field the config leaves out
takes the backend default and stays left out unless the user sets another value (`assign`).
"""
import copy
import json
import math
import re
from dataclasses import dataclass, field
from pathlib import Path

HYBRID = "sapbert_char_context"
BASE_CONFIG = Path("configs", "xmr4el_base_config.json")  # MedMentions config: the template when no spec names one
SPECS_DIR = Path("configs", "experiments")
OMIT = object()  # "leave the key out" (backend default)
INT = re.compile(r"[+-]?\d+")

# Top-level keys in the order the builder writes them (pipeline order); any other key follows in its own order.
ORDER = ("features", "transformer_config", "vectorizer_config", "abbrev_expansion", "dimension_config",
         "context_vectorizer_config", "context_window", "context_dimension_config", "clustering_config", "depth",
         "min_leaf_size", "cut_half_cluster", "routing_features", "matcher_config", "predict_config")


@dataclass(frozen=True)
class Field:
    """One attribute. `path` is relative to the slot's config key unless `top` (a top-level model key).
    kind: int | float | df (TF-IDF document frequency: 0.0-1.0 share or a count) | text | bool | choice | range.
    `empty`: what a cleared text field means: "omit" (backend default), "null", or None (required)."""
    path: tuple
    label: str
    kind: str
    default: object = None
    low: float | None = None
    high: float | None = None
    open_low: bool = False      # low is exclusive (C > 0)
    choices: tuple = ()          # ((label, value), ...)
    empty: str | None = None
    top: bool = False
    help: str = ""


@dataclass(frozen=True)
class Component:
    id: str
    name: str
    description: str
    type: str | None             # the config's "type" (None: predict_config, a custom transformer)
    fields: tuple
    base: dict = field(default_factory=dict)  # the slot config a newly chosen component starts from
    closed: bool = False         # the backend takes exactly `fields` as kwargs (any other key fails or is ignored)


@dataclass(frozen=True)
class Slot:
    id: str
    name: str                    # compact name in the grid
    stage: str
    key: str                     # its model config key
    components: tuple
    extras: tuple = ()           # top-level fields edited with it (depth, context_window, ...)

    @property
    def keys(self):
        """Every top-level key this slot owns (highlighted together in the JSON preview)."""
        return (self.key, *(f.path[0] for f in self.extras))


def kw(name, *args, **kwargs):
    return Field(("kwargs", name), *args, **kwargs)


# --- attributes ---------------------------------------------------------------------------------------------------
TRANSFORMER_FIELDS = (
    kw("revision", "Revision", "text", None, empty="omit",
       help="Hub branch, tag or commit of the checkpoint; empty = main. Training pins it to the resolved commit hash."),
    kw("pooling", "Pooling", "choice", None, choices=(("checkpoint default", None), ("cls", "cls"), ("mean", "mean")),
       help="How token vectors become one mention vector. Default: SapBERT cls, S-BioBert mean, any other checkpoint mean."),
    kw("max_seq_length", "Max sequence length", "int", None, low=1, empty="omit",
       help="Tokens kept per mention; empty = the checkpoint's own (SapBERT 25, S-BioBert 75), capped at the model's positions."),
    kw("batch_size", "Batch size", "int", 1000, low=0,
       help="Texts per forward pass during embedding inference: larger is faster and needs more memory; 0 = 400."),
    kw("max_oom_retries", "Out-of-memory retries", "int", 3, low=0,
       help="Times a batch is retried at a smaller size after the GPU runs out of memory."),
)
CUSTOM_FIELDS = (kw("model_name", "Model", "text", None,
                    help="Hugging Face model id or local directory of the encoder (required)."),) + TRANSFORMER_FIELDS
TFIDF_FIELDS = (
    kw("analyzer", "Analyzer", "choice", "word",
       choices=(("word", "word"), ("char", "char"), ("char_wb", "char_wb")),
       help="Tokens counted: words, character n-grams, or character n-grams inside word boundaries."),
    kw("ngram_range", "N-gram range", "range", [1, 2], low=1,
       help="Smallest and largest n-gram length (min <= max); characters for char analyzers, words for word."),
    kw("sublinear_tf", "Sublinear TF", "bool", False, help="Replace the term count tf by 1 + log(tf)."),
    kw("max_df", "Max document frequency", "df", 0.98,
       help="Ignore terms in more documents than this: 0.0-1.0 = share of documents, a whole number = count."),
    kw("min_df", "Min document frequency", "df", 0.0,
       help="Ignore terms in fewer documents than this: 0.0-1.0 = share of documents, a whole number = count."),
    kw("max_features", "Max features", "int", None, low=1, empty="omit",
       help="Vocabulary cap (most frequent terms kept); empty = no cap."),
    kw("stop_words", "Stop words", "choice", None, choices=(("none", None), ("english", "english")),
       help="Words removed before counting; scikit-learn applies it to the word analyzer only."),
    kw("norm", "Row norm", "choice", "l2", choices=(("l2", "l2"), ("l1", "l1"), ("none", None)),
       help="Normalisation of each output row; none leaves the TF-IDF values unscaled."),
    kw("binary", "Binary TF", "bool", False, help="Count term presence (1) instead of occurrences."),
    kw("use_idf", "Use IDF", "bool", True, help="Weight terms by inverse document frequency; No = plain term frequency."),
    kw("smooth_idf", "Smooth IDF", "bool", True,
       help="Add one to every document frequency, as if one extra document held every term (avoids zero division)."),
)
SVD_FIELDS = (
    kw("n_components", "Components", "int", 1000, low=1,
       help="Output dimensions. Larger than the TF-IDF vocabulary: the SVD is skipped and the block kept sparse."),
    kw("algorithm", "Algorithm", "choice", "randomized", choices=(("randomized", "randomized"), ("arpack", "arpack")),
       help="SVD solver: randomized (fast, approximate) or arpack (exact, slower)."),
    kw("n_iter", "Iterations", "int", 5, low=0, help="Power iterations of the randomized solver (more = more accurate)."),
    kw("n_oversamples", "Oversamples", "int", 10, low=1,
       help="Extra random vectors the randomized solver samples beyond the components."),
    kw("power_iteration_normalizer", "Power normalizer", "choice", "auto",
       choices=(("auto", "auto"), ("QR", "QR"), ("LU", "LU"), ("none", "none")),
       help="Stabilises the randomized solver's power iterations; auto = none up to 2 iterations, else LU."),
    kw("random_state", "Random seed", "int", 0, low=0, empty="null",
       help="Seed of the randomized solver and arpack's start vector; empty = unseeded (not repeatable)."),
    kw("tol", "Tolerance", "float", 0.0, low=0, help="arpack convergence tolerance; 0 = machine precision."),
)
KMEANS_FIELDS = (
    kw("n_clusters", "Clusters per node", "int", 8, low=1,
       help="Children each internal node splits its labels into (recursive 2-means, then balanced refinement: near-equal sizes). Halved per layer with Cut half cluster."),
    kw("iter_limit", "Iteration limit", "int", 20, low=1,
       help="Maximum iterations of each 2-means split; stops earlier once the split no longer changes."),
    kw("refine_iter", "Refine iterations", "int", 20, low=0,
       help="Rounds of joint balanced reassignment of all labels to the k centroids after the splits; 0 = none."),
    kw("seed", "Seed", "int", 0, low=0, help="Seed of the initial centroids; the same seed repeats the clustering."),
)
LOGREG_FIELDS = (
    kw("C", "C (inverse regularisation)", "float", 1.0, low=0, open_low=True,
       help="Inverse L2 regularisation strength of each label's logistic regression: larger = weaker regularisation."),
    kw("class_weight", "Class weight", "choice", "balanced", choices=(("balanced", "balanced"), ("none", None)),
       help="balanced weights each label's positive and negative rows by class frequency; none weights them equally."),
    kw("tol", "Tolerance", "float", 0.001, low=0, open_low=True, help="Stop when max |gradient| <= tol."),
    kw("max_iter", "Max iterations", "int", 500, low=1, help="Maximum L-BFGS iterations when solving a node's labels jointly."),
)
SEARCH_FIELDS = (
    Field(("beam_size",), "Beam size", "int", 10, low=1,
          help="Clusters kept at each level of the top-down beam search: larger = more recall, slower."),
    Field(("topk",), "Top k", "int", 0, low=0,
          help="Labels returned per mention; 0 = all scored (with kNN fusion all are ranked, then cut)."),
    Field(("knn_beta",), "kNN beta", "float", 0.0, low=0, high=100,
          help="Multiplies each candidate's score by exp(beta * knn), knn = max cosine of the mention to that label's training rows; 0 = off. Saved with the tree; evaluations can override it."),
    Field(("leaf_candidates",), "Leaf candidates", "int", 100, low=1,
          help="Labels each reached leaf contributes as candidates."),
)

MODEL_NAMES = {"sapbert": "cambridgeltl/SapBERT-from-PubMedBERT-fulltext",
               "biobert": "dmis-lab/biobert-base-cased-v1.2",
               "sentencetbiobert": "pritamdeka/S-BioBert-snli-multinli-stsb"}  # xmr4el.features.transformers


def transformer(cid, name, text):
    return Component(cid, name, f"{text} ({MODEL_NAMES[cid]}).", cid, TRANSFORMER_FIELDS,
                     {"type": cid, "kwargs": {}})


TFIDF = Component("tfidf", "TF-IDF", "scikit-learn TfidfVectorizer.", "tfidf", TFIDF_FIELDS,
                  {"type": "tfidf", "kwargs": {}})
SVD = Component("sklearntruncatedsvd", "Truncated SVD",
                "scikit-learn TruncatedSVD; skipped when the vocabulary is smaller than the components.",
                "sklearntruncatedsvd", SVD_FIELDS, {"type": "sklearntruncatedsvd", "kwargs": {}})
SLOTS = (
    Slot("transformer", "Transformer", "Feature construction", "transformer_config", (
        transformer("sapbert", "SapBERT", "Biomedical entity-name encoder"),
        transformer("biobert", "BioBERT", "Biomedical BERT, mean pooling"),
        transformer("sentencetbiobert", "Sentence-T BioBERT", "BioBERT trained for sentence similarity"),
        Component("custom", "Custom Hugging Face model", "Any Hugging Face encoder id or local model directory.",
                  None, CUSTOM_FIELDS, {"kwargs": {}}))),
    Slot("char_tfidf", "Character TF-IDF", "Feature construction", "vectorizer_config", (TFIDF,), (
        Field(("abbrev_expansion",), "Abbreviation expansion", "choice", None, top=True,
              choices=(("off", None), ("append", "append"), ("replace", "replace")),
              help="In-document long forms added to (append) or replacing the mention, for every mention block."),)),
    Slot("mention_svd", "Mention SVD", "Feature construction", "dimension_config", (SVD,)),
    Slot("context_tfidf", "Context TF-IDF", "Feature construction", "context_vectorizer_config", (TFIDF,), (
        Field(("context_window",), "Context window", "int", None, low=0, empty="null", top=True,
              help="Words kept each side of the mention for the context block; empty = the whole document."),)),
    Slot("context_svd", "Context SVD", "Feature construction", "context_dimension_config", (SVD,)),
    Slot("hierarchy", "Hierarchy", "Hierarchy", "clustering_config", (
        Component("balancedkmeans", "Balanced KMeans", "Recursive balanced k-means over the PIFA label embeddings.",
                  "balancedkmeans", KMEANS_FIELDS, {"type": "balancedkmeans", "kwargs": {}}, closed=True),), (
        Field(("depth",), "Tree depth", "int", 1, low=1, top=True,
              help="Layers of the hierarchy: every layer but the last clusters labels and routes; the last scores labels individually. Training fails if a node cannot split into clusters of at least the minimum leaf size."),
        Field(("min_leaf_size",), "Minimum leaf size", "int", 20, low=0, top=True,
              help="Clusters smaller than this are merged into the nearest valid cluster. At an internal layer, a node unable to form two valid clusters makes training fail; lower depth or this minimum."),
        Field(("cut_half_cluster",), "Cut half cluster", "bool", False, top=True,
              help="Below the root the cluster count is halved per layer (layer 1: k // 2, layer 2: k // 4, ...), never under 2."),
        Field(("routing_features",), "Routing features", "bool", True, top=True,
              help="Append the parent's cluster scores (this cluster's, their sum and their maximum) to a node's matcher features, in training and search."))),
    Slot("matcher", "Matcher", "Matcher", "matcher_config", (
        Component("jointlogisticregression", "Joint Logistic Regression",
                  "One-vs-rest L2 logistic regression per node, all labels solved jointly (torch L-BFGS).",
                  "jointlogisticregression", LOGREG_FIELDS, {"type": "jointlogisticregression", "kwargs": {}},
                  closed=True),)),
    Slot("search", "Search", "Search", "predict_config", (
        Component("beam", "Beam search", "Top-down beam search; the tree's default prediction settings.",
                  None, SEARCH_FIELDS, {}),)),
)
BY_ID = {s.id: s for s in SLOTS}


def slot(sid):
    return BY_ID[sid]


# --- help panel ---------------------------------------------------------------------------------------------------
def json_path(slot_, f):
    """The field's configuration key as dotted text ("clustering_config.kwargs.n_clusters")."""
    return ".".join(path_of(slot_, f))


def constraints(f):
    """Accepted values of a field in words (from the same rules `check` applies), with its default."""
    if f.kind == "bool":
        rule = "Yes or No"
    elif f.kind == "choice":
        rule = "one of " + ", ".join(c[0] for c in f.choices)
    elif f.kind == "range":
        rule = f"two whole numbers >= {int(f.low)}, min <= max"
    elif f.kind == "text":
        rule = "text"
    elif f.kind == "df":
        rule = "0.0-1.0 (share) or a whole number >= 1 (count)"
    else:
        rule = ("whole number" if f.kind == "int" else "number") + bound(f)
    if f.empty:
        rule += "; empty = " + {"omit": "backend default", "null": "none"}[f.empty]
    default = "none" if f.default is None else show(f.default) if f.kind != "bool" else "Yes" if f.default else "No"
    if f.kind == "choice":
        default = next((c[0] for c in f.choices if c[1] == f.default), default)
    return f"{rule}. Default: {default}."


# --- reading a configuration --------------------------------------------------------------------------------------
def path_of(slot_, f):
    return f.path if f.top else (slot_.key, *f.path)


def get(config, path, default=OMIT):
    node = config
    for key in path:
        if not isinstance(node, dict) or key not in node:
            return default
        node = node[key]
    return node


def value(config, slot_, f):
    """The field's effective value: the config's, else the backend default."""
    return get(config, path_of(slot_, f), f.default)


def installed(config, slot_):
    """The component the config loads in this slot, or None (unknown type, wrong shape)."""
    part = config.get(slot_.key)
    if slot_.id == "search":
        return slot_.components[0] if part is None or isinstance(part, dict) else None
    if not isinstance(part, dict):
        return None
    if "kwargs" in part and not isinstance(part["kwargs"], dict):
        return None
    if slot_.id == "transformer" and isinstance(part.get("kwargs"), dict) and "model_name" in part["kwargs"]:
        return slot_.components[-1]  # kwargs.model_name overrides the type
    return next((c for c in slot_.components if c.type is not None and c.type == part.get("type")), None)


def ordered(config):
    """Copy with the top-level keys in pipeline order (ORDER), every other key after them in its own order."""
    out = {k: copy.deepcopy(config[k]) for k in ORDER if k in config}
    out.update((k, copy.deepcopy(v)) for k, v in config.items() if k not in out)
    return out


def part(config, slot_):
    """The slot's own keys of `config` (deep copy): what the editor drafts."""
    return {k: copy.deepcopy(config[k]) for k in slot_.keys if k in config}


def merged(config, slot_, draft):
    """`config` with the slot's keys replaced by `draft`'s (a key missing from the draft is removed)."""
    out = {k: copy.deepcopy(v) for k, v in config.items() if k not in slot_.keys}
    out.update(copy.deepcopy(draft))
    return ordered(out)


def assign(draft, origin, slot_, f, new):
    """Set `f` to `new` in `draft`: OMIT removes it; the backend default on a key `origin` leaves out stays left
    out (the config only says what differs)."""
    path = path_of(slot_, f)
    parent = draft
    for key in path[:-1]:
        parent = parent.setdefault(key, {})
    leaf = path[-1]
    if new is OMIT or (get(origin, path) is OMIT and same(new, f.default)):
        parent.pop(leaf, None)
    else:
        parent[leaf] = copy.deepcopy(new)


def same(a, b):
    if type(a) is not type(b):
        return False
    if isinstance(a, dict):
        return a.keys() == b.keys() and all(same(a[k], b[k]) for k in a)
    if isinstance(a, list):
        return len(a) == len(b) and all(same(x, y) for x, y in zip(a, b))
    return a == b


# --- text <-> value -----------------------------------------------------------------------------------------------
def show(v):
    """A value as the editor shows it (floats keep their point: 1.0, not 1)."""
    if v is None or v is OMIT:
        return ""
    return repr(v) if isinstance(v, float) else str(v)


def parse(f, text):
    """(value, None) or (None, message) for the text typed into a text field. Malformed input is an error, never 0."""
    text = text.strip()
    if not text:
        if f.empty == "omit":
            return OMIT, None
        if f.empty == "null":
            return None, None
        return None, "Required."
    if f.kind == "text":
        return text, None
    if f.kind == "int":
        if not INT.fullmatch(text):
            return None, f"Whole number{bound(f)}."
        v = int(text)
    elif f.kind in ("float", "df"):
        if f.kind == "df" and INT.fullmatch(text):
            v = int(text)
        else:
            try:
                v = float(text)
            except ValueError:
                return None, f"Number{bound(f)}."
    else:
        raise ValueError(f"{f.kind} fields are not typed")
    err = check(f, v)
    return (None, err) if err else (v, None)


def bound(f):
    if f.kind == "df":
        return ": 0.0-1.0 (share) or a whole number >= 1 (count)"
    low = "" if f.low is None else f"{'>' if f.open_low else '>='} {show(f.low) if f.kind == 'float' else int(f.low)}"
    high = "" if f.high is None else f"<= {f.high}"
    return (" " + ", ".join(p for p in (low, high) if p)) if low or high else ""


def is_int(v):
    return isinstance(v, int) and not isinstance(v, bool)


def check(f, v):
    """Message when `v` (a value of the config) breaks the field's rule, else None."""
    if v is None:
        ok = f.empty == "null" or (f.kind == "choice" and None in [c[1] for c in f.choices]) or \
            (f.empty == "omit" and f.default is None)
        return None if ok else "Required."
    k = f.kind
    if k == "bool":
        return None if isinstance(v, bool) else "Yes or no."
    if k == "choice":
        return None if v in [c[1] for c in f.choices] else f"One of {', '.join(c[0] for c in f.choices)}."
    if k == "text":
        return None if isinstance(v, str) and v.strip() else "Text required."
    if k == "range":
        ok = isinstance(v, list) and len(v) == 2 and all(is_int(x) and x >= f.low for x in v) and v[0] <= v[1]
        return None if ok else f"Two whole numbers >= {int(f.low)}, min <= max."
    if k == "df":
        if is_int(v):
            return None if v >= 1 else "A count must be >= 1."
        ok = isinstance(v, float) and 0.0 <= v <= 1.0
        return None if ok else f"Number{bound(f)}."
    if k == "int" and not is_int(v):
        return f"Whole number{bound(f)}."
    if k == "float" and not (isinstance(v, (int, float)) and not isinstance(v, bool) and math.isfinite(v)):
        return f"Finite number{bound(f)}."
    if f.low is not None and (v <= f.low if f.open_low else v < f.low) or f.high is not None and v > f.high:
        return f"{'Whole number' if k == 'int' else 'Number'}{bound(f)}."
    return None


def coerce(f, v):
    """A float field keeps float JSON (10 typed -> 10.0); everything else as parsed."""
    return float(v) if f.kind == "float" and is_int(v) else v


# --- validation ---------------------------------------------------------------------------------------------------
def slot_problems(config, slot_, component=None):
    """[(field or None, message)] of one slot in `config` (component validity, not launch readiness)."""
    comp = component or installed(config, slot_)
    if comp is None:
        got = config.get(slot_.key)
        if isinstance(got, dict) and "kwargs" in got and not isinstance(got["kwargs"], dict):
            return [(None, "Component kwargs must be a JSON object.")]
        what = got.get("type") if isinstance(got, dict) else got
        return [(None, f"No supported component ({'missing' if got is None else repr(what)}).")]
    out = [(f, err) for f in comp.fields + slot_.extras if (err := check(f, value(config, slot_, f)))]
    kwargs = (config.get(slot_.key) or {}).get("kwargs") if slot_.key in config else None
    if comp.closed and isinstance(kwargs, dict):
        known = [f.path[-1] for f in comp.fields]
        out += [(None, f"Unknown {comp.name} setting {k!r}; it takes {', '.join(known)}.") for k in kwargs
                if k not in known]
    if comp.fields is TFIDF_FIELDS:
        lo, hi = (value(config, slot_, comp.fields[i]) for i in (4, 3))  # min_df, max_df
        if type(lo) is type(hi) and not check(comp.fields[3], hi) and not check(comp.fields[4], lo) and lo > hi:
            out.append((comp.fields[4], "Must not exceed Max document frequency."))
    if slot_.id == "search":
        out += [(None, f"Unknown search key {k!r}.") for k in config.get("predict_config") or {}
                if k not in [f.path[0] for f in SEARCH_FIELDS]]
    return out


def problems(config):
    """{slot id: [(field, message)]} for every slot with a problem, plus "features" when the recipe is not the
    hybrid one."""
    out = {s.id: p for s in SLOTS if (p := slot_problems(config, s))}
    if config.get("features", HYBRID) != HYBRID:
        out["features"] = [(None, f"features {config.get('features')!r}: the Model Builder builds {HYBRID!r} only.")]
    return out


# --- summaries for the grid ---------------------------------------------------------------------------------------
def fields_of(config, slot_):
    comp = installed(config, slot_)
    return {f.path[-1]: value(config, slot_, f) for f in (comp.fields if comp else ()) + slot_.extras}


def summary(config, slot_):
    """Short applied settings of a slot ("char_wb 2-4 · sublinear")."""
    if installed(config, slot_) is None:
        return ""
    v, sid = fields_of(config, slot_), slot_.id
    if sid == "transformer":
        bits = [v.get("pooling"), f"{v['max_seq_length']} tok" if v.get("max_seq_length") else None,
                f"batch {v['batch_size']}"]
    elif sid in ("char_tfidf", "context_tfidf"):
        rng = v["ngram_range"] if isinstance(v["ngram_range"], list) and len(v["ngram_range"]) == 2 else ["?", "?"]
        bits = [f"{v['analyzer']} {rng[0]}-{rng[1]}", "sublinear" if v["sublinear_tf"] is True else None]
        if sid == "char_tfidf":
            bits.append(f"abbrev {v['abbrev_expansion']}" if v["abbrev_expansion"] else None)
        else:
            bits.append("whole doc" if v["context_window"] is None else f"window {v['context_window']}")
    elif sid in ("mention_svd", "context_svd"):
        bits = [f"{v['n_components']} dims"]
    elif sid == "hierarchy":
        bits = [f"k {v['n_clusters']}", f"depth {v['depth']}", f"leaf {v['min_leaf_size']}"]
    elif sid == "matcher":
        bits = [f"C {show(v['C'])}", v["class_weight"] or "unweighted"]
    else:
        bits = [f"beam {v['beam_size']}", f"knn {show(v['knn_beta'])}"]
    return " · ".join(str(b) for b in bits if b)


def component_label(config, slot_):
    comp = installed(config, slot_)
    if comp is None:
        return "unsupported"
    if comp.id == "custom":
        name = str(get(config, (slot_.key, "kwargs", "model_name"), "") or "custom")
        return name.rsplit("/", 1)[-1]
    return comp.name


# --- preview ------------------------------------------------------------------------------------------------------
def json_lines(config):
    """[(line, top-level key or None)] of `json.dumps(config, indent=2)`: each line knows the key it belongs to, so
    a section is highlighted by its JSON path, not by matching text."""
    if not config:
        return [("{}", None)]
    out, keys = [("{", None)], list(config)
    for i, k in enumerate(keys):
        chunk = json.dumps({k: config[k]}, indent=2).split("\n")[1:-1]
        if i < len(keys) - 1:
            chunk[-1] += ","
        out += [(line, k) for line in chunk]
    return out + [("}", None)]


def path_line(lines, path):
    """Index in `json_lines` output of the line holding the key at `path`, searched only inside its top-level key's
    own lines; None when the config does not write it (a default left out)."""
    own = [i for i, (_, key) in enumerate(lines) if key == path[0]]
    at = own[0] if own else None
    for key in path[1:]:
        at = next((i for i in own if at is not None and i > at and lines[i][0].lstrip().startswith(json.dumps(key) + ":")),
                  None)
    return at


def dumps(config):
    return json.dumps(config, indent=2)


# --- templates ----------------------------------------------------------------------------------------------------
def read(root, rel):
    try:
        data = json.loads(Path(root, rel).read_text(encoding="utf-8"))
    except (OSError, ValueError, UnicodeDecodeError):
        return None
    return data if isinstance(data, dict) else None


def template(root, selection=None):
    """(config, source, note) a new tree starts from: the model config of an experiment spec that trains on the
    confirmed training file (sorted by spec name), else configs/xmr4el_base_config.json. A TF-IDF-only config is
    refused (`note` says which), never converted. (None, None, note) when none is usable."""
    named = []
    if selection is not None and selection.train:
        for spec in sorted(Path(root, SPECS_DIR).glob("*.json")):
            data = read(root, spec.relative_to(root))
            if data and data.get("train_path") == selection.train and isinstance(data.get("model_config"), str):
                named.append(data["model_config"])
    notes = []
    for rel in dict.fromkeys([*named, str(BASE_CONFIG)]):
        cfg = read(root, rel)
        if cfg is None:
            continue
        if cfg.get("features", HYBRID) != HYBRID:
            notes.append(f"{rel} uses features {cfg.get('features')!r} (not the hybrid recipe): not used")
            continue
        return ordered({"features": HYBRID, **cfg}), rel, "; ".join(notes)
    return None, None, "; ".join(notes) or "no model configuration found (configs/xmr4el_base_config.json)"
