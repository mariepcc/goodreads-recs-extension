import os

os.environ["TF_USE_LEGACY_KERAS"] = "1"

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
        dropout_rate=0.4,
    ):
        super(UserTower, self).__init__()

        self.book_embedding = tf.keras.Sequential(
            [
                tf.keras.layers.IntegerLookup(
                    vocabulary=unique_book_ids.tolist(),
                    oov_token=0,
                    mask_token=None,
                ),
                tf.keras.layers.Embedding(
                    input_dim=embedding_matrix.shape[0],
                    output_dim=embedding_matrix.shape[1],
                    embeddings_initializer=tf.keras.initializers.Constant(
                        embedding_matrix
                    ),
                    trainable=True,
                    mask_zero=True,
                ),
            ]
        )
        self.rating_projector = tf.keras.layers.Dense(64, activation="relu")

        self.attention = tf.keras.layers.MultiHeadAttention(
            num_heads=8,
            key_dim=embedding_matrix.shape[1] + 64,
        )

        self.attn_dropout = tf.keras.layers.Dropout(dropout_rate)

        self.pooling = tf.keras.layers.GlobalMaxPooling1D()

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

        mean_rating = tf.reduce_mean(history_ratings, axis=1, keepdims=True)
        norm_ratings = history_ratings - mean_rating
        ratings_expanded = tf.expand_dims(norm_ratings, -1)
        projected_ratings = self.rating_projector(ratings_expanded)

        concat_input = tf.concat([history_emb, projected_ratings], axis=-1)

        attn_out = self.attention(concat_input, concat_input)

        attn_out = self.attn_dropout(attn_out)

        pooled = self.pooling(attn_out)

        user_emb = self.final_layers(pooled)

        return user_emb


class ItemTower(tf.keras.Model):
    def __init__(
        self,
        unique_book_ids,
        embedding_matrix,
        emb_dim=32,
        dropout_rate=0.4,
        global_mean=4.002,
    ):
        super().__init__()
        self.vocabulary = unique_book_ids
        self.embedding_matrix = embedding_matrix
        self.global_mean = global_mean

        self.book_embedding = tf.keras.Sequential(
            [
                tf.keras.layers.IntegerLookup(
                    vocabulary=unique_book_ids.tolist(),
                    oov_token=0,
                    mask_token=None,
                ),
                tf.keras.layers.Embedding(
                    input_dim=embedding_matrix.shape[0],
                    output_dim=embedding_matrix.shape[1],
                    embeddings_initializer=tf.keras.initializers.Constant(
                        embedding_matrix
                    ),
                    trainable=True,
                ),
            ]
        )
        self.projector = tf.keras.Sequential(
            [
                tf.keras.layers.Dense(256, activation="relu"),
                tf.keras.layers.Dropout(dropout_rate),
                tf.keras.layers.Dense(emb_dim),
            ]
        )

    def call(self, inputs):
        item_embedding = self.book_embedding(inputs["book_id"])
        return self.projector(item_embedding)


class RankingModel(tfrs.models.Model):
    def __init__(self, user_tower, item_tower, droput_rate=0.4):
        super().__init__()
        self.user_tower = user_tower
        self.item_tower = item_tower

        self.ratings = tf.keras.Sequential(
            [
                tf.keras.layers.Dense(
                    128,
                    activation="relu",
                    kernel_regularizer=tf.keras.regularizers.l2(0.001),
                ),
                tf.keras.layers.Dropout(droput_rate),
                tf.keras.layers.Dense(
                    64,
                    activation="relu",
                    kernel_regularizer=tf.keras.regularizers.l2(0.001),
                ),
                tf.keras.layers.Dense(1),
            ]
        )

        self.task = tfrs.tasks.Ranking(
            loss=tf.keras.losses.MeanSquaredError(),
            metrics=[tf.keras.metrics.RootMeanSquaredError()],
        )

    def call(self, features):
        user_emb = self.user_tower(
            {
                "user_history": features["user_history"],
                "history_ratings": features["history_ratings"],
            }
        )

        item_emb = self.item_tower({"book_id": features["book_id"]})

        x = tf.concat([user_emb, item_emb], axis=1)
        # scaled_prediction = 3.0 + tf.keras.activations.tanh(self.ratings(x)) * 2.0

        return self.ratings(x)

    def compute_loss(self, features, training=False):
        labels = features["rating"]
        predictions = self(features)
        return self.task(labels=labels, predictions=predictions)


class GoodreadsTrainer:
    def __init__(self):
        pass
        logger.info("GoodreadsTrainer initialized.")
        t, v, c = prepare_datasets()
        self.trainset = t
        self.testset = v
        self.candidates = c

        self.embedding_matrix = np.load("src/data/embeddings_matrix.npy", mmap_mode="r")

        num_books = self.embedding_matrix.shape[0]
        self.unique_book_ids = np.arange(0, num_books)

        logger.info(f"Dataset & Embeddings ({self.embedding_matrix.shape}) ready.")

    def train_model(self):
        logger.info("Starting training...")

        logger.info("Building towers...")
        self.user_tower = UserTower(
            unique_book_ids=self.unique_book_ids,
            embedding_matrix=self.embedding_matrix,
        )
        self.item_tower = ItemTower(
            unique_book_ids=self.unique_book_ids,
            embedding_matrix=self.embedding_matrix,
        )
        logger.info("Towers built.")

        self.model = RankingModel(self.user_tower, self.item_tower)

        lr = 0.0002
        self.model.compile(
            optimizer=tf.keras.optimizers.Adam(learning_rate=lr),
        )
        logger.info("Model compiled.")

        mlflow.set_tracking_uri(os.getenv("mlflow_tracking_uri"))
        mlflow.set_experiment("TTM Baseline")
        logger.info("MLflow connected.")

        with mlflow.start_run(run_name="two_tower_v2_optimized"):
            mlflow.log_params(
                {
                    "optimizer": "Adam",
                    "learning_rate": lr,
                    "epochs": 10,
                    "batch_size": 128,
                    "embedding_dim_internal": 32,
                }
            )
            logger.info("MLflow parameters logged.")

            mlflow_callback = tf.keras.callbacks.LambdaCallback(
                on_epoch_end=lambda epoch, logs: [
                    mlflow.log_metric(name, value, step=epoch)
                    for name, value in logs.items()
                ]
            )
            early_stopping = tf.keras.callbacks.EarlyStopping(
                monitor="val_loss",
                patience=2,
                restore_best_weights=True,
            )

            train_ds = self.trainset.prefetch(tf.data.AUTOTUNE)
            test_ds = self.testset.prefetch(tf.data.AUTOTUNE)

            logger.info("Training model with .fit()...")
            self.history = self.model.fit(
                train_ds,
                epochs=5,
                callbacks=[mlflow_callback, early_stopping],
                validation_data=test_ds,
                verbose=1,
            )
            results = self.model.evaluate(test_ds, return_dict=True)
            for name, value in results.items():
                mlflow.log_metric(f"final_{name}", value)


if __name__ == "__main__":
    trainer = GoodreadsTrainer()
    trainer.train_model()
    # tf.saved_model.save(trainer.model, "export")

    for batch in trainer.testset.take(1):
        y_true = batch["rating"]
        y_pred = trainer.model(batch)

    print("Real ratings:", y_true.numpy().flatten()[:10])
    print("Predicted:", y_pred.numpy().flatten()[:10])

    """test_input = {
        "user_history": tf.constant([[1, 2, 3, 4, 5]]),
        "history_ratings": tf.constant([[5.0, 4.0, 3.0, 4.5, 5.0]]),
        "book_id": tf.constant([10]),
        "avg_rating": tf.constant([1.0]),
    }

    score = trainer.model.predict(test_input)
    print("Predicted rating:", float(score))
    """
