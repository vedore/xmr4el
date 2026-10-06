import logging
import torch
import numpy as np

from gc import collect
from numpy import array
from torch import no_grad
from torch.cuda import OutOfMemoryError, empty_cache
from sentence_transformers import SentenceTransformer


logger = logging.getLogger(__name__)

# Short config `type` names -> checkpoints (configs and saved trees use these). Any other model:
# "kwargs": {"model_name": "<HF or sentence-transformers id>"}, optionally "pooling" and "max_seq_length".
MODEL_NAMES = {
    "biobert": "dmis-lab/biobert-base-cased-v1.2",
    "sentencetbiobert": "pritamdeka/S-BioBert-snli-multinli-stsb",
    "sapbert": "cambridgeltl/SapBERT-from-PubMedBERT-fulltext",
}

# Plain HF checkpoints that their authors pool by [CLS], with a max token length.
# SentenceTransformer(name) alone would mean-pool them.
CLS_POOLED = {"cambridgeltl/SapBERT-from-PubMedBERT-fulltext": 25}


def sentence_model(model_name, device="cpu", pooling=None, max_seq_length=None):
    """Load the encoder used for every transformer embedding (training, prediction, screening).
    pooling "cls" | "mean" | ... builds [transformer, pooling] explicitly; None = the checkpoint's own
    sentence-transformers setup (mean pooling for a plain HF checkpoint), or CLS_POOLED's."""
    from sentence_transformers.sentence_transformer.modules import Pooling, Transformer as STTransformer
    if pooling is None and model_name in CLS_POOLED:
        pooling, max_seq_length = "cls", max_seq_length or CLS_POOLED[model_name]
    if pooling is None:
        model = SentenceTransformer(model_name).to(device)
        if max_seq_length is not None:
            model.max_seq_length = max_seq_length
        return model
    tok = STTransformer(model_name, max_seq_length=max_seq_length)
    pool = Pooling(tok.get_embedding_dimension(), pooling_mode=pooling)
    return SentenceTransformer(modules=[tok, pool], device=str(device))


class Transformer:
    """Embeds texts with a frozen transformer; config = {"type": <MODEL_NAMES key>, "kwargs": {...}}."""

    @classmethod
    def transform(cls, trn_corpus, config=None, dtype=np.float32):
        """Returns (run config, embeddings ndarray). kwargs: model_name (overrides type), pooling,
        max_seq_length, batch_size, max_oom_retries."""
        config = config if config is not None else {"type": "sentencetbiobert", "kwargs": {}}
        kwargs = {"batch_size": 1000, "dtype": dtype, "max_oom_retries": 3, **config.get("kwargs", {})}
        model_name = kwargs.pop("model_name", None) or MODEL_NAMES.get(config.get("type"))
        assert model_name, f"transformer config {config} needs a known 'type' or kwargs.model_name"
        kwargs.pop("device", None)  # _predict picks cuda, then mps (Apple GPU), then cpu
        kwargs.pop("batch_dir", None), kwargs.pop("output_prefix", None)  # obsolete: batches stay in memory
        print(kwargs)
        # Mentions repeat (500 labels: 23512 rows, 5937 distinct strings): embed each distinct text once
        uniq, inv = np.unique(np.asarray(list(trn_corpus), dtype=object).astype(str), return_inverse=True)
        return kwargs, cls._predict(model_name, uniq.tolist(), **kwargs)[inv.ravel()]

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
    ):
        """
        Optimized function for efficient memory usage during CPU or GPU-based embedding extraction.
        """

        device = torch.device("cuda" if torch.cuda.is_available() else "mps" if torch.backends.mps.is_available() else "cpu")
        
        logger.info(f"Using PyTorch device: {device}")

        model = sentence_model(model_name, device, pooling, max_seq_length)
        len_corpus = len(trn_corpus)

        if batch_size == 0:
            batch_size = 400  # A safe default, or you could implement auto-tuning

        # Process batches in row order with OOM recovery; a batch size reduced by an OOM is kept
        batches = []
        start = 0

        while start < len_corpus:
            end = min(start + batch_size, len_corpus)
            print(f"Processing rows {start}-{end} of {len_corpus} (batch size: {batch_size})")

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

            except OutOfMemoryError as oom:
                empty_cache()
                collect()

                if max_oom_retries <= 0:
                    raise RuntimeError("Failed to process batch after multiple OOM retries") from oom
                max_oom_retries -= 1

                # Reduce batch size more aggressively based on error frequency
                reduction_factor = min(0.5, max(0.1, 1 - (0.2 * max_oom_retries)))
                batch_size = max(1, int(batch_size * reduction_factor))

        return np.vstack(batches).astype(dtype)
