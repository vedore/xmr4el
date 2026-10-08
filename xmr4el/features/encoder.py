import os
import pickle
import logging
import numpy as np
from typing import Any, Dict, Optional, Tuple, Sequence, List
from scipy.sparse import csr_matrix, hstack
from sklearn.preprocessing import normalize
from xmr4el.features.vectorizers import Vectorizer
from xmr4el.features.reduction import DimensionModel
from xmr4el.features.transformers import Transformer


FEATURES = ("tfidf", "sapbert_char_context")


class TextEncoder():
    
    def __init__(
        self,
        vectorizer_config: Optional[Dict[str, Any]] = None,
        transformer_config: Optional[Dict[str, Any]] = None,
        dimension_config: Optional[Dict[str, Any]] = None,
        features: str = "sapbert_char_context",
        context_vectorizer_config: Optional[Dict[str, Any]] = None,
        context_dimension_config: Optional[Dict[str, Any]] = None,
    ) -> None:
        """Create a new :class:`TextEncoder`."""
        
        self.logger = logging.getLogger(__name__)
        
        self.vectorizer_config = vectorizer_config
        self.transformer_config = transformer_config
        self.dimension_config = dimension_config
        
        self._vectorizer_model: Optional[Vectorizer] = None
        self._dimension_model: Optional[DimensionModel] = None
        self.features = features  # see `_encode`
        self.context_vectorizer_config = context_vectorizer_config
        self.context_dimension_config = context_dimension_config
        self.context_vectorizer_model = None
        self.context_dimension_model = None
        self.block_widths = None  # "sapbert_char_context": fitted [transformer, char, context] widths
    
    @property
    def vectorizer_model(self) -> Optional[Vectorizer]:
        """Return the fitted vectorizer model, if any."""
        return self._vectorizer_model
    
    @vectorizer_model.setter
    def vectorizer_model(self, value: Vectorizer) -> None:
        """Set the fitted vectorizer model."""
        self._vectorizer_model = value
        
    @property
    def dimension_model(self) -> Optional[DimensionModel]:
        """Return the fitted dimensionality reduction model."""
        return self._dimension_model
    
    @dimension_model.setter
    def dimension_model(self, value: DimensionModel) -> None:
        """Set the fitted dimensionality reduction model."""
        self._dimension_model = value
        
    def save(self, save_dir: str) -> None:
        """Persist the encoder and its models to ``save_dir``."""
        models = ["vectorizer_model", "dimension_model", "context_vectorizer_model", "context_dimension_model"]
        models_data = [getattr(self, model_name, None) for model_name in models]
        for name, model in zip(models, models_data):
            if model is not None and not callable(getattr(model, "save", None)):
                raise TypeError(f"{name} ({type(model).__name__}) has no save()")
        os.makedirs(save_dir, exist_ok=True)

        state = self.__dict__.copy()
        for idx, model in enumerate(models_data):
            if model is not None:
                model.save(os.path.join(save_dir, models[idx]))
                state.pop("_" + models[idx] if idx < 2 else models[idx], None)

        with open(os.path.join(save_dir, "text_encoder.pkl"), "wb") as fout:
            pickle.dump(state, fout)


    @classmethod
    def load(cls, load_dir: str) -> "TextEncoder":
        """Load a previously saved :class:`TextEncoder` instance."""
        text_encoder_path = os.path.join(load_dir, "text_encoder.pkl")
        if not os.path.exists(text_encoder_path):
            raise FileNotFoundError(f"Text Encoder path {text_encoder_path} does not exist")

        with open(text_encoder_path, "rb") as fin:
            model_data = pickle.load(fin)

        model = cls()
        model.__dict__.update(model_data)
        model._validate_features()
        
        # Load models
        model_files = {
            "vectorizer_model": Vectorizer if hasattr(Vectorizer, "load") else None,
            "dimension_model": DimensionModel if hasattr(DimensionModel, "load") else None,
            "context_vectorizer_model": Vectorizer,
            "context_dimension_model": DimensionModel,
        }
        
        for model_name, model_class in model_files.items():
            model_path = os.path.join(load_dir, model_name)
            if os.path.exists(model_path) and model_class is not None:
                setattr(model, model_name, model_class.load(model_path))
            else:
                if model_name.startswith("context_") and model.features != "sapbert_char_context":
                    setattr(model, model_name, None)
                else:
                    raise Exception("Something with the loading the models is not right")
            
        return model
    
    @staticmethod
    def _reduce_dimensions(
        X_emb: csr_matrix, dim_config: Optional[Dict[str, Any]]
    ) -> Tuple[csr_matrix, DimensionModel]:
        """Reduce dimensionality of sparse embeddings."""
        if dim_config is None:
            logging.getLogger(__name__).debug("Using default TruncatedSVD configuration")
        
        model = DimensionModel.fit(X_emb, dim_config)
        
        if model is None:
            return X_emb, None
        
        reduced_emb = model.transform(X_emb)
        return reduced_emb, model
    
    @staticmethod 
    def _predict_dimension(X_emb: csr_matrix, dim_model: DimensionModel) -> csr_matrix:
        """Project embeddings using an existing dimensionality reduction model."""
        if dim_model is None:
            raise AttributeError("No model found in dim_model")
        return dim_model.transform(X_emb)
    
    @staticmethod
    def _encode_text_using_text_vectorizer(
        X_test: Sequence[str], vec_config: Optional[Dict[str, Any]]
    ) -> Tuple[csr_matrix, Vectorizer]:
        """Fit a text vectorizer and transform input texts."""
        if vec_config is None:
            logging.getLogger(__name__).debug("Using default TF-IDF configuration")
            
        model = Vectorizer.fit(X_test, vec_config)
        sparse_emb = model.transform(X_test) # CSR_MATRIX    
        return sparse_emb, model
    
    @staticmethod
    def _predict_text_using_text_vectorizer(
        X_test: Sequence[str], vec_model: Vectorizer
    ) -> csr_matrix:
        """Transform texts using an existing vectorizer."""
        if vec_model is None:
            raise AttributeError("No model found in vec_model")
        return vec_model.transform(X_test)
    
    @staticmethod
    def _encode_text_using_transformer(
        X_test: Sequence[str], transformer_config: Optional[Dict[str, Any]]
    ) -> Any:
        """Embed texts with the configured transformer (no fitting: same call for train and query)."""
        if transformer_config is None:
            raise AttributeError("features 'sapbert_char_context' needs a transformer_config")
        _, transformer_embeddings = Transformer.transform(X_test, transformer_config)
        return transformer_embeddings

    def _tfidf_block(self, texts: Sequence[str], prefix: str, fit: bool) -> csr_matrix:
        """TF-IDF (-> dimension model) block; prefix "" = mention/whole text, "context_" = context.
        fit=True fits and stores `{prefix}vectorizer_model` / `{prefix}dimension_model`."""
        if fit:
            X, vec = self._encode_text_using_text_vectorizer(texts, getattr(self, f"{prefix}vectorizer_config"))
            X, dim = self._reduce_dimensions(X, getattr(self, f"{prefix}dimension_config"))
            setattr(self, f"{prefix}vectorizer_model", vec)
            setattr(self, f"{prefix}dimension_model", dim)
            return csr_matrix(X)
        X = self._predict_text_using_text_vectorizer(texts, getattr(self, f"{prefix}vectorizer_model"))
        dim = getattr(self, f"{prefix}dimension_model")
        return csr_matrix(X if dim is None else self._predict_dimension(X, dim))

    def _validate_features(self):
        if self.features not in FEATURES:
            raise ValueError(f"features must be one of {FEATURES}, got {self.features!r}")

    def _encode(self, X_text: Sequence[str], fit: bool) -> csr_matrix:
        """
        "tfidf" → TF-IDF (-> dimension model) of the whole text (plain TSV input)
        "sapbert_char_context" → [transformer(mention) | TF-IDF -> SVD(mention) | TF-IDF -> SVD(context)], text split at
                 [SEP] (mention [SEP] context); each block L2-normalised before the concat, so each
                 carries an equal share of the row norm. Any transformer (`transformer_config`).
        """
        self._validate_features()
        if self.features == "tfidf":
            return normalize(self._tfidf_block(X_text, "", fit))
        if any("[SEP]" not in t for t in X_text):
            raise ValueError("features 'sapbert_char_context' needs 'mention [SEP] context' input")
        mentions, contexts = zip(*(t.split("[SEP]", 1) for t in X_text)) if X_text else ((), ())
        blocks = [csr_matrix(self._encode_text_using_transformer(list(mentions), self.transformer_config)),
                  self._tfidf_block(list(mentions), "", fit),
                  self._tfidf_block(list(contexts), "context_", fit)]
        if fit:  # an SVD can be skipped or give fewer columns than its n_components
            self.block_widths = [b.shape[1] for b in blocks]
        return normalize(hstack([normalize(b) for b in blocks]))

    def encode(self, X_test: Sequence[str]) -> csr_matrix:
        """Fit the feature models on the training texts and return their normalised features."""
        return self._encode(X_test, fit=True)

    def predict(self, X_text_query: Sequence[str]) -> csr_matrix:
        """Features for query texts with the models fitted by `encode`."""
        return self._encode(X_text_query, fit=False)
