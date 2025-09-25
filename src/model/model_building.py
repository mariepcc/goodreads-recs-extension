import os

os.environ["TF_USE_LEGACY_KERAS"] = "1"

import os
import numpy as np
import logging
import tensorflow as tf
import tensorflow_recommenders as tfrs
import mlflow
from src.data.data_ingestion import prepare_datasets

logger = logging.getLogger("model_building")
logger.setLevel(logging.DEBUG)


class UserTower(tf.keras.Model):
    def __init__(
        self,
        unique_book_ids,
        num_tokens,
        embedding_matrix,
        emb_dim=32,
        dropout_rate=0.2,
    ):
        super(UserTower, self).__init__()

        self.book_embedding = tf.keras.Sequential(
            [
                tf.keras.layers.IntegerLookup(
                    vocabulary=unique_book_ids,
                    oov_token=None,
                    mask_token=0,
                ),
                tf.keras.layers.Embedding(
                    input_dim=num_tokens + 1,
                    output_dim=embedding_matrix.shape[1],
                    embeddings_initializer=tf.keras.initializers.Constant(
                        embedding_matrix
                    ),
                    mask_zero=True,
                    trainable=False,
                ),
            ]
        )

        self.attention = tf.keras.layers.MultiHeadAttention(
            num_heads=8,
            key_dim=embedding_matrix.shape[1] + 1,  # 1536 embedding + 1 rating
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

        history_emb = self.book_embedding(user_history)  # (batch, seq_len, 1536)

        mean_rating = tf.reduce_mean(
            history_ratings, axis=1, keepdims=True
        )  # (batch, 1)
        norm_ratings = history_ratings - mean_rating  # (batch, seq_len)
        ratings_expanded = tf.expand_dims(norm_ratings, -1)  # (batch, seq_len, 1)
        concat_input = tf.concat(
            [history_emb, ratings_expanded], axis=-1
        )  # (batch, seq_len, 1537)

        attn_out = self.attention(concat_input, concat_input)

        pooled = self.pooling(attn_out)

        user_emb = self.final_layers(pooled)

        return user_emb


class ItemTower(tf.keras.Model):
    def __init__(
        self,
        unique_book_ids,
        num_tokens,
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
                    oov_token=None,
                    mask_token=None,
                ),
                tf.keras.layers.Embedding(
                    input_dim=num_tokens + 1,
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
                candidates=candidates_ds.map(item_tower)
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

        print("Building candidates dataset...")
        self.trainset = prepare_datasets()[0]
        self.valset = prepare_datasets()[1]
        self.candidates = prepare_datasets()[2]
        print("Dataset preparation complete.")

    def train_model(self):
        print("Starting training...")
        embedding_matrix = np.load("../src/data/embeddings_matrix.npy")
        print(f"Loaded embeddings: {embedding_matrix.shape}")

        num_books, embedding_dim = embedding_matrix.shape
        unique_book_ids = np.arange(num_books - 1)

        print("Building towers...")
        self.user_tower = UserTower(
            unique_book_ids=unique_book_ids,
            num_tokens=num_books,
            embedding_matrix=embedding_matrix,
        )
        self.item_tower = ItemTower(
            unique_book_ids=unique_book_ids,
            num_tokens=num_books,
            embedding_matrix=embedding_matrix,
        )
        print("Towers built.")

        print("Building retrieval model...")
        self.model = BookRetrievalModel(
            self.user_tower, self.item_tower, self.candidates
        )

        self.model.compile(
            optimizer=tf.keras.optimizers.Adam(learning_rate=0.001),
            metrics=[
                tf.keras.metrics.Precision(name="precision"),
                tf.keras.metrics.Recall(name="recall"),
                tf.keras.metrics.Accuracy(name="accuracy"),
            ],
        )
        print("Model compiled.")

        print("Connecting to MLflow...")
        mlflow.set_tracking_uri(
            "http://ec2-13-61-4-77.eu-north-1.compute.amazonaws.com:5000/"
        )
        mlflow.set_experiment("TTM Baseline")
        print("MLflow connected.")

        mlflow.keras.autolog()

        print("Starting MLflow run...")
        with mlflow.start_run(run_name="two_tower_model_experiment"):
            mlflow.log_param("optimizer", "Adam")
            mlflow.log_param("learning_rate", 0.1)
            mlflow.log_param("epochs", 5)
            mlflow.log_param("loss", "sparse_categorical_crossentropy")
            mlflow.log_param("embedding_dim", embedding_dim)
            mlflow.log_param("num_books", num_books)
            print("✅ MLflow parameters logged.")

            mlflow_callback = tf.keras.callbacks.LambdaCallback(
                on_epoch_end=lambda epoch, logs: [
                    mlflow.log_metric(name, value, step=epoch)
                    for name, value in logs.items()
                ]
            )

            print("Training model with .fit()...")
            self.model = self.model.fit(
                self.trainset,
                epochs=5,
                callbacks=[mlflow_callback],
                validation_data=self.testset,
            )
            print("Training finished.")

            print("Saving model to MLflow...")
            mlflow.tensorflow.log_model(self.model, artifact_path="two_tower_model")

            print("MLflow run completed. Run ID:", mlflow.active_run().info.run_id)


"""
            def compute_recall_at_k(model, test_users, true_items, k=10):
                user_embs = model.user_tower(test_users)  # shape: (n_users, emb_dim)
                item_embs = model.item_tower(
                    np.arange(num_books)
                )  # shape: (n_items, emb_dim)

                scores = tf.linalg.matmul(
                    user_embs, item_embs, transpose_b=True
                )  # similarity
                top_k = tf.math.top_k(scores, k=k).indices.numpy()  # top-k item IDs

                hits = 0
                for i, true_item in enumerate(true_items):
                    if true_item in top_k[i]:
                        hits += 1
                return hits / len(test_users)

            test_users = []
            test_true_items = []

            for user, true_item in testset.take(-1):
                test_users.append(user.numpy())
                test_true_items.append(true_item.numpy())

            test_users = np.array(test_users)
            test_true_items = np.array(test_true_items)

            ks = [5, 10, 20]
            recalls = []
            for k in ks:
                recall = compute_recall_at_k(model, test_users, test_true_items, k)
                recalls.append(recall)
                mlflow.log_metric(f"recall@{k}", recall)

            plt.figure()
            plt.plot(ks, recalls, marker="o")
            plt.title("Recall@K")
            plt.xlabel("K")
            plt.ylabel("Recall")
            plt.savefig("recall_at_k.png")
            mlflow.log_artifact("recall_at_k.png")
            plt.close()
        """


if __name__ == "__main__":
    trainer = GoodreadsTrainer()
    trainer.get_datasets()
    trainer.train_model()
