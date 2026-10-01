import os
import pickle
import joblib
import logging

from typing import Any, Dict, Optional, Tuple, Sequence
from scipy.sparse import csr_matrix, hstack
from sklearn.preprocessing import normalize
from xmr4el.models.featurization_wrapper.dimension_model import DimensionModel
from xmr4el.models.featurization_wrapper.transformers import Transformer
from xmr4el.models.featurization_wrapper.vectorizers import Vectorizer


class TextEncoder():
    
    def __init__(
        self,
        vectorizer_config: Optional[Dict[str, Any]] = None,
        transformer_config: Optional[Dict[str, Any]] = None,
        dimension_config: Optional[Dict[str, Any]] = None,
        flag: int = 2,
    ) -> None:
        """Create a new :class:`TextEncoder`."""
        
        self.logger = logging.getLogger(__name__)
        
        self.vectorizer_config = vectorizer_config
        self.transformer_config = transformer_config
        self.dimension_config = dimension_config
        
        self._vectorizer_model: Optional[Vectorizer] = None
        self._dimension_model: Optional[DimensionModel] = None
        self.flag = flag
    
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
        os.makedirs(save_dir, exist_ok=True)

        state = self.__dict__.copy()
        models = ["vectorizer_model", "dimension_model"]
        models_data = [getattr(self, model_name, None) for model_name in models]

        for idx, model in enumerate(models_data):
            if model is not None:
                model_path = os.path.join(save_dir, models[idx])
                if hasattr(model, "save"):
                    model.save(model_path)
                else:
                    try:
                        joblib.dump(model, f"{model_path}.joblib")
                    except ImportError:
                        with open(f"{model_path}.pkl", "wb") as f:
                            pickle.dump(model, f)
                state.pop(models[idx], None)

        with open(os.path.join(save_dir, "text_encoder.pkl"), "wb") as fout:
            pickle.dump(state, fout)


    @classmethod
    def load(cls, load_dir: str) -> "TextEncoder":
        """Load a previously saved :class:`TextEncoder` instance."""
        text_encoder_path = os.path.join(load_dir, "text_encoder.pkl")
        assert os.path.exists(text_encoder_path), f"Text Encoder path {text_encoder_path} does not exist"

        with open(text_encoder_path, "rb") as fin:
            model_data = pickle.load(fin)

        model = cls()
        model.__dict__.update(model_data)
        
        # Load models
        model_files = {
            "vectorizer_model": Vectorizer if hasattr(Vectorizer, "load") else None,
            "dimension_model": DimensionModel if hasattr(DimensionModel, "load") else None,
        }
        
        for model_name, model_class in model_files.items():
            model_path = os.path.join(load_dir, model_name)
            if os.path.exists(model_path) and model_class is not None:
                setattr(model, model_name, model_class.load(model_path))
            else:
                if model.flag == 3:
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
            print("Running on default config of TruncateSVD")
        
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
            print("Running on default config of TF-IDF")
            
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
        """Encode texts using a Transformer model."""
        if transformer_config is None:
            print("Running on default config of BioBert")
        
        _, transformer_embeddings = Transformer.transform(X_test, transformer_config)
        return transformer_embeddings
    
    @staticmethod
    def _predict_text_using_transformer(
        X_test: Sequence[str], transformer_config: Dict[str, Any]
    ) -> Any:
        """Predict embeddings for ``X_test`` using a Transformer model."""
        if transformer_config is None:
            raise AttributeError("No config found in transformer_config")
        _, transformer_embeddings = Transformer.transform(X_test, transformer_config)
        return transformer_embeddings
        
    def encode(self, X_test: Sequence[str]) -> csr_matrix:
        """
        Encode input texts into normalized feature vectors.
        Supports different encoding pipelines based on `self.flag`:

        flag == 1 → TF-IDF
        flag == 2 → TF-IDF + Transformer
        flag == 3 → Transformer
        flag == 4 → [Transformer | TF-IDF] formula with [SEP] splitting
        flag == 5 → [Transformer | TF-IDF], both on the mention (text before [SEP]), each block
                    L2-normalised before the concat (meant for a char n-gram vectorizer config)
        """

        # Determine encoding mode
        use_tfidf = self.flag in [1, 2]
        use_transformer = self.flag in [2, 3]
        use_formula = self.flag in (4, 5)

        # Logging mode
        if self.flag == 1:
            self.logger.info("Using only TF-IDF as encoder")
        elif self.flag == 2:
            self.logger.info("Using TF-IDF + Transformer to encode")
        elif self.flag == 3:
            self.logger.info("Using only Transformer to encode")
        elif self.flag == 4:
            self.logger.info("Using formula encoder (Transformer + TF-IDF via [SEP])")
        elif self.flag == 5:
            self.logger.info("Using mention encoder (Transformer + TF-IDF, both on the mention)")
        else:
            raise ValueError(f"Invalid flag {self.flag}")

        # --------------------------------------------------------------------------
        # FLAG 4: "Formula mode" using [SEP] to split transformer/tfidf inputs
        # --------------------------------------------------------------------------
        if use_formula:
            self.logger.info("Encoding using [SEP] formula pipeline")

            X_transformer_raw, X_tfidf_raw = [], []
            for item in X_test:
                if "[SEP]" not in item:
                    raise ValueError("Input must contain [SEP] when flag is 4 or 5")
                a, b = item.split("[SEP]", 1)
                X_transformer_raw.append(a)
                X_tfidf_raw.append(b)

            # Transformer embedding
            X_trans = self._encode_text_using_transformer(
                X_transformer_raw, self.transformer_config
            )
            X_trans = csr_matrix(X_trans)

            # TF-IDF + dimension reduction
            if self.flag == 5:
                X_tfidf_raw = X_transformer_raw
            X_tfidf, vec_model = self._encode_text_using_text_vectorizer(
                X_tfidf_raw, self.vectorizer_config
            )
            reduced_x_tfidf, dim_model = self._reduce_dimensions(
                X_tfidf, self.dimension_config
            )

            reduced_x_tfidf = (
                csr_matrix(reduced_x_tfidf) if dim_model is not None else X_tfidf
            )

            # Save models
            self.vectorizer_model = vec_model
            self.dimension_model = dim_model

            # Concatenate: [TRANSFORMER | TF-IDF]
            if self.flag == 5:  # equal block weight; raw transformer norms otherwise take ~all of it
                X_trans, reduced_x_tfidf = normalize(X_trans), normalize(reduced_x_tfidf)
            concat_emb = hstack([X_trans, reduced_x_tfidf])

            # Normalize
            return normalize(concat_emb, norm="l2", axis=1)

        # --------------------------------------------------------------------------
        # OTHER FLAGS (1, 2, 3)
        # --------------------------------------------------------------------------

        concat_emb = None

        # -------- TF-IDF ENCODING --------
        if use_tfidf:
            self.logger.info("Encoding with TF-IDF")

            X_tfidf, vec_model = self._encode_text_using_text_vectorizer(
                X_test, self.vectorizer_config
            )

            reduced_x_tfidf, dim_model = self._reduce_dimensions(
                X_tfidf, self.dimension_config
            )

            if dim_model is None:
                reduced_x_tfidf = X_tfidf
            else:
                reduced_x_tfidf = csr_matrix(reduced_x_tfidf)

            self.vectorizer_model = vec_model
            self.dimension_model = dim_model

            concat_emb = reduced_x_tfidf

        # -------- TRANSFORMER ENCODING --------
        if use_transformer:
            self.logger.info("Encoding with Transformer model")

            X_transformer = self._encode_text_using_transformer(
                X_test, self.transformer_config
            )
            sparse_X_transformer = csr_matrix(X_transformer)

            if concat_emb is None:
                concat_emb = sparse_X_transformer
            else:
                concat_emb = hstack([concat_emb, sparse_X_transformer])

        # Sanity check
        if concat_emb is None:
            raise RuntimeError("No encoder was executed — check flag logic.")

        # Normalize embeddings
        return normalize(concat_emb, norm="l2", axis=1)
    
    def predict(self, X_text_query: Sequence[str]) -> csr_matrix:
        """Encode query data for prediction using the SAME logic as `encode`."""

        use_tfidf = self.flag in [1, 2]
        use_transformer = self.flag in [2, 3]
        use_formula = self.flag in (4, 5)

        # ------------------------------------------------------------
        # FLAG 4: "[SEP]" formula mode
        # ------------------------------------------------------------
        if use_formula:
            self.logger.info("Predicting using [SEP] formula pipeline")

            X_transformer_raw, X_tfidf_raw = [], []
            for item in X_text_query:
                if "[SEP]" not in item:
                    raise ValueError("Input must contain [SEP] when flag is 4 or 5")
                a, b = item.split("[SEP]", 1)
                X_transformer_raw.append(a)
                X_tfidf_raw.append(b)

            # Transformer prediction
            X_trans = self._predict_text_using_transformer(
                X_test=X_transformer_raw,
                transformer_config=self.transformer_config
            )
            X_trans = csr_matrix(X_trans)

            # TF-IDF prediction
            if self.flag == 5:
                X_tfidf_raw = X_transformer_raw
            X_tfidf = self._predict_text_using_text_vectorizer(
                X_test=X_tfidf_raw,
                vec_model=self.vectorizer_model
            )

            # Dimensionality prediction
            if self.dimension_model is None:
                reduced_x_tfidf = X_tfidf
            else:
                reduced_x_tfidf = csr_matrix(
                    self._predict_dimension(X_tfidf, self.dimension_model)
                )

            # Concatenate
            if self.flag == 5:
                X_trans, reduced_x_tfidf = normalize(X_trans), normalize(reduced_x_tfidf)
            concat_emb = hstack([X_trans, reduced_x_tfidf])

            return normalize(concat_emb, norm="l2", axis=1)

        # ------------------------------------------------------------
        # FLAGS 1, 2, 3 (standard modes)
        # ------------------------------------------------------------

        concat_emb = None

        # -------- TF-IDF --------
        if use_tfidf:
            X_tfidf_query = self._predict_text_using_text_vectorizer(
                X_test=X_text_query,
                vec_model=self.vectorizer_model,
            )

            if self.dimension_model is None:
                reduced_x_tfidf = X_tfidf_query
            else:
                reduced_x_tfidf = csr_matrix(
                    self._predict_dimension(X_tfidf_query, self.dimension_model)
                )

            concat_emb = reduced_x_tfidf

        # -------- Transformer --------
        if use_transformer:
            X_transformer = self._predict_text_using_transformer(
                X_test=X_text_query,
                transformer_config=self.transformer_config
            )
            sparse_X_transformer = csr_matrix(X_transformer)

            if concat_emb is None:
                concat_emb = sparse_X_transformer
            else:
                concat_emb = hstack([concat_emb, sparse_X_transformer])

        # Safety check
        if concat_emb is None:
            raise RuntimeError("No encoder was executed — invalid flag or configuration.")

        return normalize(concat_emb, norm="l2", axis=1)
