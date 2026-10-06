import logging
import os
import glob
import shutil
import torch
import numpy as np

from gc import collect
from numpy import savez_compressed, array
from torch import no_grad
from torch.cuda import OutOfMemoryError, empty_cache
from sentence_transformers import SentenceTransformer
from concurrent.futures import ThreadPoolExecutor


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
        max_seq_length, batch_size, batch_dir, output_prefix, max_oom_retries."""
        config = config if config is not None else {"type": "sentencetbiobert", "kwargs": {}}
        kwargs = {"batch_size": 1000, "batch_dir": "batch_dir", "output_prefix": "st_emb",
                  "dtype": dtype, "max_oom_retries": 3, **config.get("kwargs", {})}
        model_name = kwargs.pop("model_name", None) or MODEL_NAMES.get(config.get("type"))
        assert model_name, f"transformer config {config} needs a known 'type' or kwargs.model_name"
        kwargs.pop("device", None)  # _predict picks cuda, then mps (Apple GPU), then cpu
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
        batch_dir="batch_dir",
        output_prefix="st_emb",
        max_oom_retries=3,
        pooling=None,
        max_seq_length=None,
    ):
        """
        Optimized function for efficient memory usage during CPU or GPU-based embedding extraction.
        """

        device = torch.device("cuda" if torch.cuda.is_available() else "mps" if torch.backends.mps.is_available() else "cpu")
        
        logger.info(f"Using PyTorch device: {device}")

        batch_dir = os.path.abspath(batch_dir)
        emb_file = f"{batch_dir}/{output_prefix}"
        cls._create_batch_dir(batch_dir)
        
        model = sentence_model(model_name, device, pooling, max_seq_length)
        len_corpus = len(trn_corpus)

        if batch_size == 0:
            batch_size = 400  # A safe default, or you could implement auto-tuning

        # Process batches with OOM recovery. Files are keyed by start row, so shrinking the batch
        # size after an OOM cannot skip or duplicate rows.
        original_batch_size = batch_size
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
                savez_compressed(f"{emb_file}_batch{start}.npz", embeddings=array(batch_results, dtype=dtype))
                start = end
                batch_size = original_batch_size

            except OutOfMemoryError as oom:
                empty_cache()
                collect()

                if max_oom_retries <= 0:
                    raise RuntimeError("Failed to process batch after multiple OOM retries") from oom
                max_oom_retries -= 1

                # Reduce batch size more aggressively based on error frequency
                reduction_factor = min(0.5, max(0.1, 1 - (0.2 * max_oom_retries)))
                batch_size = max(1, int(batch_size * reduction_factor))

            except Exception as _:
                cls._del_batch_dir(batch_dir)
                raise

        # Parallel loading of batch files
        def load_embedding_file(file):
            with np.load(file) as data:
                return data["embeddings"]
        
        # Numeric order: lexicographic puts batch10 before batch2 and permutes rows past 10 batches
        batch_files = sorted(
            glob.glob(f"{emb_file}_batch*.npz"),
            key=lambda f: int(f.rsplit("_batch", 1)[1][: -len(".npz")]),
        )
        with ThreadPoolExecutor(max_workers=min(4, os.cpu_count())) as executor:
            all_embeddings = list(executor.map(load_embedding_file, batch_files))
        
        # Clean up
        cls._del_batch_dir(batch_dir)
        
        return np.vstack(all_embeddings).astype(dtype)


    @staticmethod
    def _create_batch_dir(batch_dir):
        """
        Create the batch dir if it does not exist,
        if it exists, remove any file inside
        """

        if os.path.exists(batch_dir):
            for item in os.listdir(batch_dir):
                emb_path = os.path.join(batch_dir, item)
                os.remove(emb_path)
        else:
            # LOGGER.warning("Directory does not exist, Creating")
            os.makedirs(batch_dir)

    @staticmethod
    def _del_batch_dir(batch_dir):
        """Delete the batch directory"""
        
        shutil.rmtree(batch_dir)
