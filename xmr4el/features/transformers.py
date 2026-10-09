import hashlib
import logging
import os
import tempfile
import time
import torch
import numpy as np

from gc import collect
from numpy import array
from torch import no_grad
from torch.cuda import OutOfMemoryError, empty_cache
from sentence_transformers import SentenceTransformer
from xmr4el import torch_device
from sentence_transformers.sentence_transformer.modules import Pooling, Transformer as STTransformer


logger = logging.getLogger(__name__)

# Short config `type` names -> checkpoints (configs and saved trees use these). Any other model:
# "kwargs": {"model_name": "<HF or sentence-transformers id>"}, optionally "pooling", "max_seq_length" and
# "revision" (a Hub commit hash pins the checkpoint; None = the Hub's current main).
MODEL_NAMES = {
    "biobert": "dmis-lab/biobert-base-cased-v1.2",
    "sentencetbiobert": "pritamdeka/S-BioBert-snli-multinli-stsb",
    "sapbert": "cambridgeltl/SapBERT-from-PubMedBERT-fulltext",
}

# Plain HF checkpoints that their authors pool by [CLS], with a max token length.
# SentenceTransformer(name) alone would mean-pool them.
CLS_POOLED = {"cambridgeltl/SapBERT-from-PubMedBERT-fulltext": 25}

# Embeddings of a frozen encoder depend only on (model, revision, pooling, max length, dtype, texts): `transform` saves
# them here and reuses them on any later run with the same distinct texts. Without a pinned revision the key cannot see
# a checkpoint update on the Hub: delete the directory then.
CACHE_DIR = os.path.join(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))),
                         "outputs", "cache", "transformers")


def sentence_model(model_name, device="cpu", pooling=None, max_seq_length=None, revision=None):
    """Load the encoder used for every transformer embedding (training, prediction, screening).
    pooling "cls" | "mean" | ... builds [transformer, pooling] explicitly; None = the checkpoint's own
    sentence-transformers setup (mean pooling for a plain HF checkpoint), or CLS_POOLED's."""
    if pooling is None and model_name in CLS_POOLED:
        pooling, max_seq_length = "cls", max_seq_length or CLS_POOLED[model_name]
    rev = {"revision": revision} if revision else None
    if pooling is None:
        model = SentenceTransformer(model_name, **(rev or {})).to(device)
        if max_seq_length is not None:
            model.max_seq_length = max_seq_length
        return model
    tok = STTransformer(model_name, max_seq_length=max_seq_length, model_kwargs=rev, processor_kwargs=rev,
                        config_kwargs=rev)
    pool = Pooling(tok.get_embedding_dimension(), pooling_mode=pooling)
    return SentenceTransformer(modules=[tok, pool], device=str(device))


class Transformer:
    """Embeds texts with a frozen transformer; config = {"type": <MODEL_NAMES key>, "kwargs": {...}}."""

    @classmethod
    def transform(cls, trn_corpus, config=None, dtype=np.float32):
        """Returns (run config, embeddings ndarray). kwargs: model_name (overrides type), pooling,
        max_seq_length, revision, batch_size, max_oom_retries."""
        config = config if config is not None else {"type": "sentencetbiobert", "kwargs": {}}
        kwargs = {"batch_size": 1000, "dtype": dtype, "max_oom_retries": 3, **config.get("kwargs", {})}
        model_name = kwargs.pop("model_name", None) or MODEL_NAMES.get(config.get("type"))
        assert model_name, f"transformer config {config} needs a known 'type' or kwargs.model_name"
        kwargs.pop("device", None)  # _predict picks cuda, then mps (Apple GPU), then cpu
        # Mentions repeat (500 labels: 23512 rows, 5937 distinct strings): embed each distinct text once
        uniq, inv = np.unique(np.asarray(list(trn_corpus), dtype=object).astype(str), return_inverse=True)
        # revision joins the key only when pinned, so unpinned configs keep their cached embeddings
        rev = [f"revision={kwargs['revision']}"] if kwargs.get("revision") else []
        key = "\0".join([model_name, *rev, str(kwargs.get("pooling")), str(kwargs.get("max_seq_length")),
                         np.dtype(kwargs["dtype"]).name, *uniq.tolist()])
        path = os.path.join(CACHE_DIR, hashlib.sha256(key.encode()).hexdigest()[:24] + ".npy")
        if os.path.exists(path):
            emb = np.load(path)
            assert emb.shape[0] == len(uniq), f"cached embeddings {path} have {emb.shape[0]} rows, expected {len(uniq)}"
            logger.info("Transformer cache hit: model=%s rows=%d path=%s", model_name, len(uniq), path)
        else:
            emb = cls._predict(model_name, uniq.tolist(), **kwargs)
            os.makedirs(CACHE_DIR, exist_ok=True)
            # write-then-rename: an interrupted save leaves no partial cache file; a private tmp per writer
            fd, tmp = tempfile.mkstemp(dir=CACHE_DIR, suffix=".tmp")
            try:
                with os.fdopen(fd, "wb") as f:
                    np.save(f, emb)
                os.replace(tmp, path)
            except BaseException:
                os.unlink(tmp)
                raise
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
    ):
        """
        Optimized function for efficient memory usage during CPU or GPU-based embedding extraction.
        """

        device = torch_device()
        
        start_time = time.perf_counter()
        logger.info("Transformer started: model=%s device=%s rows=%d batch_size=%d",
                    model_name, device, len(trn_corpus), batch_size or 400)

        model = sentence_model(model_name, device, pooling, max_seq_length, revision)
        len_corpus = len(trn_corpus)

        if batch_size == 0:
            batch_size = 400  # A safe default, or you could implement auto-tuning

        # Process batches in row order with OOM recovery; a batch size reduced by an OOM is kept
        batches = []
        start = 0

        while start < len_corpus:
            end = min(start + batch_size, len_corpus)
            logger.debug("Transformer batch: rows=%d:%d/%d batch_size=%d", start, end, len_corpus, batch_size)

            try:
                with no_grad():  # Disable gradient calculation
                    batch_results = model.encode(
                        trn_corpus[start:end],
                        convert_to_tensor=False,
                        device=device,
                        batch_size=batch_size,
                        normalize_embeddings=False,
                        show_progress_bar=False,
                    )
                batches.append(array(batch_results, dtype=dtype))
                start = end

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

        embeddings = np.vstack(batches).astype(dtype)
        logger.info("Transformer completed: shape=%s elapsed=%.1fs", embeddings.shape, time.perf_counter() - start_time)
        return embeddings
