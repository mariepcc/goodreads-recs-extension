import os

os.environ["TF_USE_LEGACY_KERAS"] = "1"
import os
import numpy as np
import matplotlib.pyplot as plt
from dotenv import load_dotenv
import logging
import tensorflow as tf
import tensorflow_recommenders as tfrs
import mlflow


from src.data.data_ingestion import prepare_datasets

load_dotenv()
logger = logging.getLogger("model_building")
logging.basicConfig(
    level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s"
)


class UserTower(tf.keras.Model):
    def __init__(self, emb_dim=64, dropout_rate=0.3):
        super(UserTower, self).__init__()
        self.rating_projector = tf.keras.layers.Dense(64, activation="relu")
        self.attention = tf.keras.layers.MultiHeadAttention(num_heads=8, key_dim=64)
        self.attn_dropout = tf.keras.layers.Dropout(dropout_rate)
        self.pooling = tf.keras.layers.GlobalAveragePooling1D()
        self.final_layers = tf.keras.Sequential(
            [
                tf.keras.layers.Dense(128, activation="relu"),
                tf.keras.layers.Dropout(dropout_rate),
                tf.keras.layers.Dense(emb_dim),
            ]
        )

    def call(self, inputs):
        history_emb = inputs["user_history"]
        history_ratings = inputs["history_ratings"]
        mask = tf.math.not_equal(history_ratings, 0)
        mean_rating = tf.reduce_mean(history_ratings, axis=1, keepdims=True)
        norm_ratings = history_ratings - mean_rating
        ratings_expanded = tf.expand_dims(norm_ratings, -1)
        projected_ratings = self.rating_projector(ratings_expanded)
        concat_input = tf.concat([history_emb, projected_ratings], axis=-1)
        attn_out = self.attention(
            concat_input,
            concat_input,
            attention_mask=tf.expand_dims(tf.expand_dims(mask, 1), 1),
        )
        attn_out = self.attn_dropout(attn_out)
        attn_out = attn_out + concat_input
        pooled = self.pooling(attn_out, mask=mask)
        return self.final_layers(pooled)


class ItemTower(tf.keras.Model):
    def __init__(self, emb_dim=64, dropout_rate=0.3):
        super().__init__()
        self.projector = tf.keras.layers.Dense(256, activation="relu")
        self.meta_layer = tf.keras.layers.Dense(32, activation="relu")
        self.final_layers = tf.keras.Sequential(
            [
                tf.keras.layers.Dropout(dropout_rate),
                tf.keras.layers.Dense(emb_dim),
            ]
        )

    def call(self, inputs):
        emb_feat = self.projector(inputs["book_emb"])
        avg_norm = tf.expand_dims(inputs["avg_rating"] / 5.0, -1)
        meta_feat = self.meta_layer(avg_norm)
        combined = tf.concat([emb_feat, meta_feat], axis=-1)
        return self.final_layers(combined)


class RankingModel(tfrs.models.Model):
    def __init__(self, user_tower, item_tower, dropout_rate=0.5):
        super().__init__()
        self.user_tower = user_tower
        self.item_tower = item_tower
        self.ratings = tf.keras.Sequential(
            [
                tf.keras.layers.Dense(
                    256,
                    activation="relu",
                    kernel_regularizer=tf.keras.regularizers.l2(0.01),
                ),
                tf.keras.layers.BatchNormalization(),
                tf.keras.layers.Dropout(dropout_rate),
                tf.keras.layers.Dense(
                    128,
                    activation="relu",
                    kernel_regularizer=tf.keras.regularizers.l2(0.01),
                ),
                tf.keras.layers.BatchNormalization(),
                tf.keras.layers.Dropout(dropout_rate),
                tf.keras.layers.Dense(
                    64,
                    activation="relu",
                    kernel_regularizer=tf.keras.regularizers.l2(0.01),
                ),
                tf.keras.layers.Dense(1),
            ]
        )
        self.task = tfrs.tasks.Ranking(
            loss=tf.keras.losses.Huber(),
            metrics=[tf.keras.metrics.RootMeanSquaredError()],
        )

    def call(self, features):
        user_emb = self.user_tower(
            {
                "user_history": features["user_history"],
                "history_ratings": features["history_ratings"],
            }
        )
        item_emb = self.item_tower(
            {"book_emb": features["book_emb"], "avg_rating": features["avg_rating"]}
        )
        x = tf.concat([user_emb, item_emb], axis=1)
        return self.ratings(x)

    def compute_loss(self, features, training=False):
        labels = features["rating"]

        predictions = self(features)

        class_weights_tensor = tf.constant(
            [0.0, 5.0, 3.0, 1.0, 0.8, 1.1], dtype=tf.float32
        )
        # [0.0, 9.62, 3.32, 0.87, 0.55, 0.60]

        sample_weight = tf.gather(class_weights_tensor, tf.cast(labels, tf.int32))
        labels = tf.expand_dims(labels, -1)

        return self.task(
            labels=labels, predictions=predictions, sample_weight=sample_weight
        )


class GoodreadsTrainer:
    def __init__(self):
        logger.info("Loading embedding matrix and preparing datasets...")
        self.embedding_matrix = np.load("src/data/embeddings_matrix.npy")
        zero_padding = np.zeros((1, 1536))
        self.embedding_matrix = np.vstack([zero_padding, self.embedding_matrix]).astype(
            np.float32
        )
        t, v, c = prepare_datasets(self.embedding_matrix)
        self.trainset = t
        self.testset = v
        self.candidates = c

    def train_model(self):
        logger.info("Building model...")
        self.user_tower = UserTower()
        self.item_tower = ItemTower()
        self.model = RankingModel(self.user_tower, self.item_tower)
        self.model.compile(
            optimizer=tf.keras.optimizers.legacy.Adam(learning_rate=0.0008),
            loss="mse",
            metrics=[tf.keras.metrics.RootMeanSquaredError()],
        )

        mlflow.set_tracking_uri(os.getenv("mlflow_tracking_uri"))
        mlflow.set_experiment("TTM Vector Input")

        logger.info("Starting training...")
        with mlflow.start_run(run_name="final_experiment_v2"):
            mlflow.log_param("epochs", 15)
            mlflow.log_param("dropout_rate", 0.5)
            mlflow.log_param("embedding_dim", 64)
            mlflow.log_param("learning_rate", 0.0008)
            mlflow.log_param("batch_size", 512)
            mlflow.log_param("num_train_samples", 49000)
            mlflow.log_param("user_history_length", 50)

            early_stopping = tf.keras.callbacks.EarlyStopping(
                monitor="val_loss",
                patience=5,
                min_delta=0.0005,
                restore_best_weights=True,
                verbose=1,
            )

            reduce_lr = tf.keras.callbacks.ReduceLROnPlateau(
                monitor="val_loss", factor=0.2, patience=1, min_lr=1e-6, verbose=1
            )

            history = self.model.fit(
                self.trainset.prefetch(tf.data.AUTOTUNE),
                epochs=15,
                validation_data=self.testset.prefetch(tf.data.AUTOTUNE),
                callbacks=[early_stopping, reduce_lr],
                verbose=1,
            )

            for epoch, loss in enumerate(history.history["loss"]):
                mlflow.log_metric("loss", loss, step=epoch)
                mlflow.log_metric(
                    "rmse",
                    history.history["root_mean_squared_error"][epoch],
                    step=epoch,
                )
                mlflow.log_metric(
                    "val_rmse",
                    history.history["val_root_mean_squared_error"][epoch],
                    step=epoch,
                )
                mlflow.log_metric(
                    "train_rmse",
                    history.history["root_mean_squared_error"][epoch],
                    step=epoch,
                )


if __name__ == "__main__":
    trainer = GoodreadsTrainer()
    trainer.train_model()
    logger.info("Training completed. Saving model weights and evaluating...")
    trainer.model.save_weights("model_weights.h5")

    for batch in trainer.testset.take(1):
        y_true = batch["rating"]
        y_pred = trainer.model(batch)

        print("Real ratings:", y_true.numpy().flatten())
        print("Predicted:", y_pred.numpy().flatten())

    try:
        plt.figure(figsize=(10, 6))
        plt.scatter(y_true, y_pred, alpha=0.5)
        plt.plot([1, 5], [1, 5], "--r")
        plt.title("Wykres Skuteczności")
        plt.xlabel("Rzeczywiste oceny")
        plt.ylabel("Predykcje modelu")
        plt.savefig("error_scatter.png")
    except Exception as e:
        print(f"Błąd podczas tworzenia wykresu: {e}")
