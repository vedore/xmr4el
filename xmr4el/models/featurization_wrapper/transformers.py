import logging
import os
import gc
import json
import glob
import pickle
import shutil
import torch
import numpy as np

from gc import collect
from os.path import exists
from numpy import savez_compressed, array
from torch import no_grad
from torch.cuda import OutOfMemoryError, empty_cache
from abc import ABCMeta
from sentence_transformers import SentenceTransformer
from concurrent.futures import ThreadPoolExecutor


transformer_dict = {}

logger = logging.getLogger(__name__)

class TransformersMeta(ABCMeta):
    """Metaclass for keeping track of all 'Transformer' subclasses"""

    def __new__(cls, name, bases, attr):
        new_cls = super().__new__(cls, name, bases, attr)
        if name != "Transformer":
            transformer_dict[name.lower()] = new_cls
        return new_cls


class Transformer(metaclass=TransformersMeta):
    """Wrapper class for all BERT-based Transformers."""

    def __init__(self, config, model):
        """Initialization

        Args:
            config (dict): Dict with key `"type"` and value being the lower
            -cased name of the specific transformer class to use.
            Also contains keyword arguments to pass to the specified
            transformer.
            model (Berttransformer): Trained Berttransformer.
        """

        self.config = config
        self.model = model


    def save(self, transformer_folder):
        """Save trained transformer to disk.

        Args:
            transformer_folder (str): Folder to save to.
        """

        # LOGGER.info(f"Saving transformer to {transformer_folder}")
        os.makedirs(transformer_folder, exist_ok=True)
        with open(
            os.path.join(transformer_folder, "best_vec_config.json"),
            "w",
            encoding="utf-8",
        ) as fout:
            fout.write(json.dumps(self.config))
        self.model.save(transformer_folder)

    @classmethod
    def load(cls, transformer_folder):
        """Load a saved transformer from disk.

        Args:
            transformer_folder (str): Folder where `Berttransformer` was saved to using `Berttransformer.save`.

        Returns:
            Berttransformer: The loaded object.
        """

        config_path = os.path.join(transformer_folder, "bert_vec_config.json")

        if not os.path.exists(config_path):
            config = {"type": "biobert", "kwargs": {}}
        else:
            with open(config_path, "r", encoding="utf-8") as fin:
                config = json.loads(fin.read())

        transformer_type = config.get("type", None)
        assert (
            transformer_type is not None
        ), f"{transformer_folder} is not a valid transformer folder"
        assert (
            transformer_type in transformer_dict
        ), f"invalid transformer type {config['type']}"
        model = transformer_dict[transformer_type].load(transformer_folder)
        return cls(config, model)

    @classmethod
    def transform(cls, trn_corpus, config=None, dtype=np.float32):
        """Train on a corpus.

        Args:
            trn_corpus (list or str): Training corpus in the form of a list of strings or path to text file.
            config (dict, optional): Dict with key `"type"` and value being the lower-cased name of the specific transformer class to use.
                Also contains keyword arguments to pass to the specified transformer. Default behavior is to use tfidf transformer with default arguments.
            dtype (type, optional): Data type. Default is `numpy.float32`.

        Returns:
            Berttransformer: Trained Berttransformer.
        """

        # LOGGER.info("Starting training for a Transformer")
        config = config if config is not None else {"type": "sentencetbiobert", "kwargs": {}}

        transformer_type = config.get("type", None)
        assert (
            transformer_type is not None
        ), f"config {config} should contain a key 'type' for the transformer type"
        # assert isinstance(trn_corpus, list), "No model supports from file training"

        # LOGGER.info(f"Training transformer of type {transformer_type}")
        defaults = {
            "batch_size": 1000,
            "batch_dir": "batch_dir",
            "output_prefix": "st_emb",
            "dtype": np.float32,
            "max_oom_retries": 3,
            "device": "cpu"
        }
        
        config = {**defaults, **config['kwargs']}
        
        print(config)
        
        model = transformer_dict[transformer_type](config)
        embeddings = cls._predict(
            model.model_name, trn_corpus, **config
        )
        
        model.embeddings = embeddings
        
        # model are the embeddings
        return config, embeddings

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
        device = "cpu"
    ):
        """
        Optimized function for efficient memory usage during CPU or GPU-based embedding extraction.
        """

        # device = torch.device("cuda" if device == "gpu" and torch.cuda.is_available() else "cpu")
        
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        
        logger.info(f"Using PyTorch device: {device}")

        batch_dir = os.path.abspath(batch_dir)
        emb_file = f"{batch_dir}/{output_prefix}"
        cls._create_batch_dir(batch_dir)
        
        model = SentenceTransformer(model_name).to(device)
        len_corpus = len(trn_corpus)

        if batch_size == 0:
            batch_size = 400  # A safe default, or you could implement auto-tuning

        # Process batches with OOM recovery
        current_batch_idx = 0
        processed_indices = set()
        original_batch_size = batch_size

        while current_batch_idx * batch_size < len_corpus:
            batch_idx = current_batch_idx
            start = batch_idx * batch_size
            end = min((batch_idx + 1) * batch_size, len_corpus)
            
            # Skip if this batch was already processed successfully
            if batch_idx in processed_indices:
                current_batch_idx += 1
                continue
                
            print(f"Processing batch {batch_idx + 1}/{(len_corpus + batch_size - 1) // batch_size} (size: {batch_size})")
            
            batch = trn_corpus[start:end]
            batch_filename = f"{emb_file}_batch{batch_idx}.npz"
            
            # Skip if this batch was already processed (from previous OOM)
            if exists(batch_filename):
                current_batch_idx += 1
                processed_indices.add(batch_idx)
                continue
                
            try:
                with no_grad():  # Disable gradient calculation
                    # Process batch
                    batch_results = model.encode(
                        batch,
                        convert_to_tensor=False,
                        device=device,
                        batch_size=batch_size,
                        normalize_embeddings=False,
                        show_progress_bar=False,
                    )
                    
                    # Save results
                    savez_compressed(batch_filename, embeddings=array(batch_results, dtype=dtype))
                    processed_indices.add(batch_idx)
                    current_batch_idx += 1
                    
                    # Reset batch size if we had reduced it previously
                    if batch_size != original_batch_size:
                        batch_size = original_batch_size
                        # LOGGER.info(f"Resetting batch size to original: {batch_size}")
                    
            except OutOfMemoryError as oom:
                # LOGGER.warning(f"OOM error processing batch {batch_idx} (size: {batch_size})")
                
                # Clean up memory
                empty_cache()
                collect()
                
                # Reduce batch size more aggressively based on error frequency
                reduction_factor = min(0.5, max(0.1, 1 - (0.2 * max_oom_retries)))
                new_batch_size = max(1, int(batch_size * reduction_factor))
                
                # LOGGER.info(f"Reducing batch size from {batch_size} to {new_batch_size}")
                batch_size = new_batch_size
                
                # Recalculate where we should be in processing
                current_batch_idx = start // batch_size
                
                # If we've retried too many times, give up
                if max_oom_retries <= 0:
                    raise RuntimeError("Failed to process batch after multiple OOM retries") from oom
                max_oom_retries -= 1

            except Exception as _:
                cls._del_batch_dir(batch_dir)
                raise
        
        # Parallel loading of batch files
        def load_embedding_file(file):
            with np.load(file) as data:
                return data["embeddings"]
        
        batch_files = sorted(glob.glob(f"{emb_file}_batch*.npz"))
        with ThreadPoolExecutor(max_workers=min(4, os.cpu_count())) as executor:
            all_embeddings = list(executor.map(load_embedding_file, batch_files))
        
        # Clean up
        cls._del_batch_dir(batch_dir)
        
        return np.vstack(all_embeddings).astype(dtype)


    @staticmethod
    def load_config_from_args(args):
        """Parse config from a `argparse.Namespace` object.

        Args:
            args (argparse.Namespace): Contains either a `transformer_config_path` (path to a json file) or `transformer_config_json` (a json object in string form).

        Returns:
            dict: The dict resulting from loading the json file or json object.

        Raises:
            Exception: If json object cannot be loaded.
        """

        if args.transformer_config_path is not None:
            with open(args.transformer_config_path, "r", encoding="utf-8") as fin:
                transformer_config_json = fin.read()
        else:
            transformer_config_json = args.transformer_config_json

        try:
            transformer_config = json.loads(transformer_config_json)
        except json.JSONDecodeError as jex:
            raise Exception(
                "Failed to load transformer config json from {} ({})".format(
                    transformer_config_json, jex
                )
            )
        return transformer_config
    
    def embeddings(self):
        return self.model.embeddings
    
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

    def _del_batch_dir(batch_dir):
        """Delete the batch directory"""
        
        shutil.rmtree(batch_dir)

class BioBert(Transformer):
    """BioBERT-based transformer."""

    model_name = "dmis-lab/biobert-base-cased-v1.2"

    def __init__(self, config=None, embeddings=None):
        """Initialization

        Args:
            config (dict): Dict with key `"type"` and value being the lower-cased name of the specific transformer class to use.
                Also contains keyword arguments to pass to the specified transformer.
            embeddings (numpy.ndarray): The Embeddings
            model_name (str): Transformer name
        """

        self.config = config
        self.embeddings = embeddings
        self.model_name = BioBert.model_name

    def save(self, save_dir):
        """Save trained tfidf transformer to disk.

        Args:
            save_dir (str): Folder to save the model.
        """

        # LOGGER.info(f"Saving BioBERT transformer to {save_dir}")
        os.makedirs(save_dir, exist_ok=True)
        with open(os.path.join(save_dir, "transformer.pkl"), "wb") as fout:
            pickle.dump(self.__dict__, fout)

    @classmethod
    def load(cls, load_dir):
        """Load a BioBert Transformer from disk.

        Args:
            load_dir (str): Folder inside which the model is loaded.

        Returns:
            BioBert: The loaded object.
        """

        # LOGGER.info(f"Loading BioBERT transformer from {load_dir}")
        transformer_path = os.path.join(load_dir, "transformer.pkl")
        assert os.path.exists(
            transformer_path
        ), f"transformer path {transformer_path} does not exist"

        with open(transformer_path, "rb") as fin:
            model_data = pickle.load(fin)
        model = cls()
        model.__dict__.update(model_data)
        return model
    
class SentenceTBioBert(Transformer):
    
    model_name = "pritamdeka/S-BioBert-snli-multinli-stsb"

    def __init__(self, config=None, embeddings=None):
        """Initialization

        Args:
            config (dict): Dict with key `"type"` and value being the lower-cased name of the specific transformer class to use.
                Also contains keyword arguments to pass to the specified transformer.
            embeddings (numpy.ndarray): The Embeddings
            model_name (str): Transformer name
        """

        self.config = config
        self.embeddings = embeddings
        self.model_name = SentenceTBioBert.model_name

    def save(self, save_dir):
        """Save trained tfidf transformer to disk.

        Args:
            save_dir (str): Folder to save the model.
        """

        # LOGGER.info(f"Saving Sentence BioBERT transformer to {save_dir}")
        os.makedirs(save_dir, exist_ok=True)
        with open(os.path.join(save_dir, "transformer.pkl"), "wb") as fout:
            pickle.dump(self.__dict__, fout)

    @classmethod
    def load(cls, load_dir):
        """Load a BioBert Transformer from disk.

        Args:
            load_dir (str): Folder inside which the model is loaded.

        Returns:
            BioBert: The loaded object.
        """

        # LOGGER.info(f"Loading BioBERT transformer from {load_dir}")
        transformer_path = os.path.join(load_dir, "transformer.pkl")
        assert os.path.exists(
            transformer_path
        ), f"transformer path {transformer_path} does not exist"

        with open(transformer_path, "rb") as fin:
            model_data = pickle.load(fin)
        model = cls()
        model.__dict__.update(model_data)
        return model

class SentenceTSapBert(Transformer):
    
    # cambridgeltl/SapBERT-UMLS-2020AB-all-lang-from-XLMR Multi Lingual
    
    model_name = "cambridgeltl/SapBERT-from-PubMedBERT-fulltext"

    def __init__(self, config=None, embeddings=None):
        """Initialization

        Args:
            config (dict): Dict with key `"type"` and value being the lower-cased name of the specific transformer class to use.
                Also contains keyword arguments to pass to the specified transformer.
            embeddings (numpy.ndarray): The Embeddings
            model_name (str): Transformer name
        """

        self.config = config
        self.embeddings = embeddings
        self.model_name = SentenceTSapBert.model_name

    def save(self, save_dir):
        """Save trained tfidf transformer to disk.

        Args:
            save_dir (str): Folder to save the model.
        """

        # LOGGER.info(f"Saving Sentence SapBERT transformer to {save_dir}")
        os.makedirs(save_dir, exist_ok=True)
        with open(os.path.join(save_dir, "transformer.pkl"), "wb") as fout:
            pickle.dump(self.__dict__, fout)

    @classmethod
    def load(cls, load_dir):
        """Load a BioBert Transformer from disk.

        Args:
            load_dir (str): Folder inside which the model is loaded.

        Returns:
            BioBert: The loaded object.
        """

        # LOGGER.info(f"Loading Sentence SapBERT transformer from {load_dir}")
        transformer_path = os.path.join(load_dir, "transformer.pkl")
        assert os.path.exists(
            transformer_path
        ), f"transformer path {transformer_path} does not exist"

        with open(transformer_path, "rb") as fin:
            model_data = pickle.load(fin)
        model = cls()
        model.__dict__.update(model_data)
        return model
    
class KRISSBert(Transformer):
    """BioBERT-based transformer."""

    model_name = "microsoft/BiomedNLP-KRISSBERT-PubMed-UMLS-EL"

    def __init__(self, config=None, embeddings=None):
        """Initialization

        Args:
            config (dict): Dict with key `"type"` and value being the lower-cased name of the specific transformer class to use.
                Also contains keyword arguments to pass to the specified transformer.
            embeddings (numpy.ndarray): The Embeddings
            model_name (str): Transformer name
        """

        self.config = config
        self.embeddings = embeddings
        self.model_name = BioBert.model_name

    def save(self, save_dir):
        """Save trained tfidf transformer to disk.

        Args:
            save_dir (str): Folder to save the model.
        """

        # LOGGER.info(f"Saving BioBERT transformer to {save_dir}")
        os.makedirs(save_dir, exist_ok=True)
        with open(os.path.join(save_dir, "transformer.pkl"), "wb") as fout:
            pickle.dump(self.__dict__, fout)

    @classmethod
    def load(cls, load_dir):
        """Load a BioBert Transformer from disk.

        Args:
            load_dir (str): Folder inside which the model is loaded.

        Returns:
            BioBert: The loaded object.
        """

        # LOGGER.info(f"Loading BioBERT transformer from {load_dir}")
        transformer_path = os.path.join(load_dir, "transformer.pkl")
        assert os.path.exists(
            transformer_path
        ), f"transformer path {transformer_path} does not exist"

        with open(transformer_path, "rb") as fin:
            model_data = pickle.load(fin)
        model = cls()
        model.__dict__.update(model_data)
        return model
