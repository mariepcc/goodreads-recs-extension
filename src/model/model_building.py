import os

os.environ["TF_USE_LEGACY_KERAS"] = "1"

import os
import numpy as np
from dotenv import load_dotenv
import logging
import tensorflow as tf
import tensorflow_recommenders as tfrs
import mlflow
from src.data.data_ingestion import prepare_datasets

load_dotenv()

logger = logging.getLogger("model_building")
logging.basicConfig(
    level=logging.INFO, format="%(asctime)s - %(name)s - %(levelname)s - %(message)s"
)


class UserTower(tf.keras.Model):
    def __init__(
        self,
        unique_book_ids,
        embedding_matrix,
        emb_dim=32,
        dropout_rate=0.2,
    ):
        super(UserTower, self).__init__()

        self.book_embedding = tf.keras.Sequential(
            [
                tf.keras.layers.IntegerLookup(
                    vocabulary=unique_book_ids,
                    oov_token=0,
                    mask_token=None,
                ),
                tf.keras.layers.Embedding(
                    input_dim=embedding_matrix.shape[0],
                    output_dim=embedding_matrix.shape[1],
                    embeddings_initializer=tf.keras.initializers.Constant(
                        embedding_matrix
                    ),
                    trainable=False,
                ),
            ]
        )

        self.attention = tf.keras.layers.MultiHeadAttention(
            num_heads=8,
            key_dim=embedding_matrix.shape[1] + 1,  
        )
        self.pooling = tf.keras.layers.GlobalAveragePooling1D()

        self.final_layers = tf.keras.Sequential(
            [
                tf.keras.layers.Dense(128, activation="relu"),
                tf.keras.layers.Dropout(dropout_rate),
                tf.keras.layers.Dense(emb_dim),
            ]
        )

    def call(self, inputs):
        user_history = inputs["user_history"]
        history_ratings = inputs["history_ratings"]

        history_emb = self.book_embedding(user_history)  

        mean_rating = tf.reduce_mean(
            history_ratings, axis=1, keepdims=True
        )  
        norm_ratings = history_ratings - mean_rating  
        ratings_expanded = tf.expand_dims(norm_ratings, -1)  
        concat_input = tf.concat(
            [history_emb, ratings_expanded], axis=-1
        ) 

        attn_out = self.attention(concat_input, concat_input)

        pooled = self.pooling(attn_out)

        user_emb = self.final_layers(pooled)

        return user_emb


class ItemTower(tf.keras.Model):
    def __init__(
        self,
        unique_book_ids,
        embedding_matrix,
        emb_dim=32,
        dropout_rate=0.2,
    ):
        super().__init__()
        self.vocabulary = unique_book_ids
        self.embedding_matrix = embedding_matrix

        self.book_embedding = tf.keras.Sequential(
            [
                tf.keras.layers.IntegerLookup(
                    vocabulary=unique_book_ids,
                    oov_token=0,
                    mask_token=None,
                ),
                tf.keras.layers.Embedding(
                    input_dim=embedding_matrix.shape[0],
                    output_dim=embedding_matrix.shape[1],
                    embeddings_initializer=tf.keras.initializers.Constant(
                        embedding_matrix
                    ),
                    trainable=False,
                ),
            ]
        )

        self.final_layers = tf.keras.Sequential(
            [
                tf.keras.layers.Dense(128, activation="relu"),
                tf.keras.layers.Dropout(dropout_rate),
                tf.keras.layers.Dense(emb_dim),
            ]
        )

    def call(self, inputs):
        item_embedding = self.book_embedding(inputs["book_id"])
        rating = tf.expand_dims(inputs["avg_rating"], -1)
        norm_rating = rating - 3.0
        concat_input = tf.concat([item_embedding, norm_rating], axis=-1)
        item_emb = self.final_layers(concat_input)
        return item_emb


class BookRetrievalModel(tfrs.models.Model):
    def __init__(self, user_tower, item_tower, candidates_ds):
        super().__init__()
        self.user_tower = user_tower
        self.item_tower = item_tower

        self.task = tfrs.tasks.Retrieval(
            metrics=tfrs.metrics.FactorizedTopK(
                candidates=candidates_ds.map(self.item_tower)
            )
        )

    def call(self, features):
        user_embeddings = self.user_tower(
            {
                "user_history": features["user_history"],
                "history_ratings": features["history_ratings"],
            }
        )

        book_embeddings = self.item_tower(
            {"book_id": features["book_id"], "avg_rating": features["avg_rating"]}
        )

        return user_embeddings, book_embeddings

    def compute_loss(self, features, training=False):
        user_embeddings, book_embeddings = self(features)
        return self.task(user_embeddings, book_embeddings)


class GoodreadsTrainer:
    def __init__(self):
        pass
        print("GoodreadsTrainer initialized.")
        logger.info("GoodreadsTrainer initialized.")
        t, v, c = prepare_datasets()
        self.trainset = t
        self.testset = v
        self.candidates = c
        logger.info("Dataset preparation complete.")

    def train_model(self):
        logger.info("Starting training...")
        embedding_matrix = np.load("src/data/embeddings_matrix.npy")
        logger.info(f"Loaded embeddings: {embedding_matrix.shape}")

        num_books, embedding_dim = embedding_matrix.shape
        unique_book_ids = np.arange(1, num_books)

        logger.info("Building towers...")
        self.user_tower = UserTower(
            unique_book_ids=unique_book_ids,
            embedding_matrix=embedding_matrix,
        )
        self.item_tower = ItemTower(
            unique_book_ids=unique_book_ids,
            embedding_matrix=embedding_matrix,
        )
        logger.info("Towers built.")

        logger.info("Building retrieval model...")
        self.model = BookRetrievalModel(
            self.user_tower, self.item_tower, self.candidates
        )

        self.model.compile(
            optimizer=tf.keras.optimizers.Adam(0.001),
        )
        logger.info("Model compiled.")

        logger.info("Connecting to MLflow...")
        mlflow.set_tracking_uri(os.getenv("mlflow_tracking_uri"))
        mlflow.set_experiment("TTM Baseline")
        logger.info("MLflow connected.")

        mlflow.keras.autolog()

        logger.info("Starting MLflow run...")
        with mlflow.start_run(run_name="two_tower_model_experiment_2"):
            mlflow.log_param("optimizer", "Adam")
            mlflow.log_param("learning_rate", 0.001)
            mlflow.log_param("epochs", 15)
            mlflow.log_param("batch_size", 256)
            mlflow.log_param("loss", "tfrs.tasks.Retrieval")
            mlflow.log_param("embedding_dim", embedding_dim)
            mlflow.log_param("num_books", num_books - 1)
            mlflow.log_param("num_users", len(self.trainset))
            logger.info("MLflow parameters logged.")

            mlflow_callback = tf.keras.callbacks.LambdaCallback(
                on_epoch_end=lambda epoch, logs: [
                    mlflow.log_metric(name, value, step=epoch)
                    for name, value in logs.items()
                ]
            )

            logger.info("Training model with .fit()...")
            self.history = self.model.fit(
                self.trainset,
                epochs=15,
                callbacks=[mlflow_callback],
                validation_data=self.testset,
                verbose=1,
            )
            logger.info("Training finished.")
            logger.info(self.model.summary())

            logger.info("Saving model to MLflow...")
            mlflow.tensorflow.log_model(self.model, name="two_tower_model")

            self.model.save("book_retrieval_model")

            logger.info(
                "MLflow run completed. Run ID:", mlflow.active_run().info.run_id
            )


if __name__ == "__main__":
    trainer = GoodreadsTrainer()
    trainer.train_model()
