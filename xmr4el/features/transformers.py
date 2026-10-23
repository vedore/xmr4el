import hashlib
import logging
import os
import tempfile
import time
import torch
import numpy as np

from functools import lru_cache
from gc import collect
from numpy import array
from torch import no_grad
from torch.cuda import OutOfMemoryError, empty_cache
from huggingface_hub import HfApi, snapshot_download
from transformers import AutoModel, AutoTokenizer
from xmr4el import progress, torch_device


logger = logging.getLogger(__name__)
TICK_SECONDS = 5  # progress ticks (events=True) at most this often, plus the final count

# Short config `type` names -> checkpoints (configs and saved trees use these). Any other model:
# "kwargs": {"model_name": "<HF id or local directory>"}, optionally "pooling", "max_seq_length" and
# "revision" (a Hub branch, tag or commit; None = main). Training resolves it to the commit hash (`resolve_revision`),
# which the saved encoder keeps: predictions use the checkpoint the tree was trained on.
MODEL_NAMES = {
    "biobert": "dmis-lab/biobert-base-cased-v1.2",
    "sentencetbiobert": "pritamdeka/S-BioBert-snli-multinli-stsb",
    "sapbert": "cambridgeltl/SapBERT-from-PubMedBERT-fulltext",
}

# Checkpoint -> (pooling, max tokens) its authors use; any other checkpoint: mean pooling up to its own max length.
# S-BioBert's pair = its sentence-transformers config (1_Pooling, sentence_bert_config.json).
POOLING = {"cambridgeltl/SapBERT-from-PubMedBERT-fulltext": ("cls", 25),
           "pritamdeka/S-BioBert-snli-multinli-stsb": ("mean", 75)}

# Embeddings of a frozen encoder depend only on (model, revision, pooling, max length, dtype, texts): `transform` saves
# them here and reuses them on any later run with the same distinct texts. Training pins the revision (a local model
# directory has none: delete the cache after changing its files). XMR4EL_TRANSFORMER_CACHE=0 turns it off (no read, no
# write), e.g. for a service whose every query batch would add a file.
CACHE_DIR = os.path.join(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))),
                         "outputs", "cache", "transformers")


def resolve_revision(config):
    """Copy of a transformer config with kwargs.revision = the Hub commit hash its revision (None = "main") points
    at now; offline, the locally cached commit. Configs without a model name or with a local directory: unchanged."""
    if config is None:
        return None
    kwargs = config.get("kwargs", {})
    name = kwargs.get("model_name") or MODEL_NAMES.get(config.get("type"))
    if not name or os.path.isdir(name):
        return config
    rev = kwargs.get("revision") or "main"
    try:
        sha = HfApi().model_info(name, revision=rev).sha
    except Exception as e:  # offline or Hub unreachable: the snapshot the local cache holds for rev
        try:
            sha = os.path.basename(snapshot_download(name, revision=rev, local_files_only=True))
        except Exception as local:
            raise RuntimeError(f"cannot resolve revision {rev!r} of {name}: Hub ({e}), local cache ({local})") from e
    logger.info("Transformer revision: model=%s revision=%s commit=%s", name, rev, sha)
    return {**config, "kwargs": {**kwargs, "revision": sha}}


@lru_cache(maxsize=2)  # loaded once per process: a per-call load costs ~1-2 s per predict
def sentence_model(model_name, device="cpu", pooling=None, max_seq_length=None, revision=None):
    """Load the encoder used for every transformer embedding (training, prediction, screening): a function
    texts -> (len(texts), hidden) float32 ndarray, one padded forward pass. pooling "cls" | "mean", None = POOLING's
    (else mean); max_seq_length None = POOLING's (else the tokenizer's), capped at the model's positions. Same
    tokenization and pooling as sentence-transformers' Transformer + Pooling (bitwise equal on CPU for both
    POOLING checkpoints), so earlier cached embeddings stay valid."""
    default_pooling, default_len = POOLING.get(model_name, ("mean", None))
    pooling = pooling or default_pooling
    if pooling not in ("cls", "mean"):
        raise ValueError(f"pooling must be 'cls' or 'mean', got {pooling!r}")
    tok = AutoTokenizer.from_pretrained(model_name, revision=revision)
    model = AutoModel.from_pretrained(model_name, revision=revision).to(device).eval()
    max_len = min(max_seq_length or default_len or tok.model_max_length, model.config.max_position_embeddings)

    def encode(texts):
        batch = tok(list(texts), padding=True, truncation=True, max_length=max_len, return_tensors="pt").to(device)
        with no_grad():
            H = model(**batch).last_hidden_state
            if pooling == "cls":
                out = H[:, 0]
            else:
                mask = batch["attention_mask"].unsqueeze(-1).to(H.dtype)
                out = (H * mask).sum(1) / mask.sum(1).clamp(min=1e-9)
        return out.float().cpu().numpy()

    return encode


class Transformer:
    """Embeds texts with a frozen transformer; config = {"type": <MODEL_NAMES key>, "kwargs": {...}}."""

    @classmethod
    def transform(cls, trn_corpus, config=None, dtype=np.float32, events=False):
        """Returns (run config, embeddings ndarray). kwargs: model_name (overrides type), pooling,
        max_seq_length, revision, batch_size, max_oom_retries. events=True (training encode): cache_hit / tick
        progress events of the "transformer" encode part."""
        config = config if config is not None else {"type": "sentencetbiobert", "kwargs": {}}
        kwargs = {"batch_size": 1000, "dtype": dtype, "max_oom_retries": 3, **config.get("kwargs", {})}
        model_name = kwargs.pop("model_name", None) or MODEL_NAMES.get(config.get("type"))
        if not model_name:
            raise ValueError(f"transformer config {config} needs a known 'type' or kwargs.model_name")
        kwargs.pop("device", None)  # _predict picks cuda, then mps (Apple GPU), then cpu
        if not (isinstance(kwargs["batch_size"], int) and kwargs["batch_size"] >= 0):  # < 0 never ends the batch loop
            raise ValueError(f"batch_size must be an int >= 0 (0 = 400), got {kwargs['batch_size']!r}")
        # Mentions repeat (500 labels: 23512 rows, 5937 distinct strings): embed each distinct text once
        uniq, inv = np.unique(np.asarray(list(trn_corpus), dtype=object).astype(str), return_inverse=True)
        if not len(uniq):
            raise ValueError("Transformer.transform: no texts to embed")
        if any("\0" in t for t in uniq):  # "\0" joins the cache key fields
            raise ValueError("Transformer.transform: texts must not contain NUL")
        if os.environ.get("XMR4EL_TRANSFORMER_CACHE", "1") == "0":
            return kwargs, cls._predict(model_name, uniq.tolist(), **kwargs, events=events)[inv.ravel()]
        # revision joins the key when set (always after training resolves it)
        rev = [f"revision={kwargs['revision']}"] if kwargs.get("revision") else []
        key = "\0".join([model_name, *rev, str(kwargs.get("pooling")), str(kwargs.get("max_seq_length")),
                         np.dtype(kwargs["dtype"]).name, *uniq.tolist()])
        path = os.path.join(CACHE_DIR, hashlib.sha256(key.encode()).hexdigest()[:24] + ".npy")
        if os.path.exists(path):
            emb = np.load(path)
            assert emb.shape[0] == len(uniq), f"cached embeddings {path} have {emb.shape[0]} rows, expected {len(uniq)}"
            logger.info("Transformer cache hit: model=%s rows=%d path=%s", model_name, len(uniq), path)
            if events:
                progress(stage="encode", part="transformer", phase="cache_hit", distinct=len(uniq))
        else:
            emb = cls._predict(model_name, uniq.tolist(), **kwargs, events=events)
            try:  # a read-only install cannot cache: keep the computed embeddings
                os.makedirs(CACHE_DIR, exist_ok=True)
                # write-then-rename: an interrupted save leaves no partial cache file; a private tmp per writer
                fd, tmp = tempfile.mkstemp(dir=CACHE_DIR, suffix=".tmp")
            except OSError as e:
                logger.warning("Transformer cache not written (%s): embeddings not cached", e)
            else:
                try:
                    with os.fdopen(fd, "wb") as f:
                        np.save(f, emb)
                    os.replace(tmp, path)
                except BaseException as e:
                    os.unlink(tmp)
                    if not isinstance(e, OSError):
                        raise
                    logger.warning("Transformer cache not written (%s): embeddings not cached", e)
        return kwargs, emb[inv.ravel()]

    @classmethod
    def _predict(
        cls,
        model_name,
        trn_corpus,
        dtype=np.float32,
        batch_size=100,
        max_oom_retries=3,
        pooling=None,
        max_seq_length=None,
        revision=None,
        events=False,
    ):
        """
        Optimized function for efficient memory usage during CPU or GPU-based embedding extraction.
        """

        device = torch_device()
        
        start_time = time.perf_counter()
        logger.info("Transformer started: model=%s device=%s rows=%d batch_size=%d",
                    model_name, device, len(trn_corpus), batch_size or 400)

        encode = sentence_model(model_name, device, pooling, max_seq_length, revision)
        len_corpus = len(trn_corpus)

        if batch_size == 0:
            batch_size = 400  # A safe default, or you could implement auto-tuning

        # Process batches in row order with OOM recovery; a batch size reduced by an OOM is kept
        batches = []
        start = 0
        last_tick = time.perf_counter()

        while start < len_corpus:
            end = min(start + batch_size, len_corpus)
            logger.debug("Transformer batch: rows=%d:%d/%d batch_size=%d", start, end, len_corpus, batch_size)

            try:
                batches.append(array(encode(trn_corpus[start:end]), dtype=dtype))
                start = end
                if events and (start == len_corpus or time.perf_counter() - last_tick >= TICK_SECONDS):
                    progress(stage="encode", part="transformer", phase="tick", unit="distinct_mentions",
                             done=start, total=len_corpus)
                    last_tick = time.perf_counter()

            except RuntimeError as oom:
                if not isinstance(oom, OutOfMemoryError) and "MPS backend out of memory" not in str(oom):
                    raise
                collect()
                if device.type == "mps":
                    torch.mps.empty_cache()
                else:
                    empty_cache()

                if max_oom_retries <= 0:
                    raise RuntimeError("Failed to process batch after multiple OOM retries") from oom
                max_oom_retries -= 1

                # Reduce batch size more aggressively based on error frequency
                reduction_factor = min(0.5, max(0.1, 1 - (0.2 * max_oom_retries)))
                batch_size = max(1, int(batch_size * reduction_factor))
                logger.warning("Transformer OOM: device=%s row=%d retry_batch_size=%d retries_left=%d",
                               device, start, batch_size, max_oom_retries)

        embeddings = np.vstack(batches).astype(dtype, copy=False)
        logger.info("Transformer completed: shape=%s elapsed=%.1fs", embeddings.shape, time.perf_counter() - start_time)
        return embeddings
