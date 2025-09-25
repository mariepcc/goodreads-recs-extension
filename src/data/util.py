import tensorflow as tf
import numpy as np
import pandas as pd
import psutil
import os


policy = tf.keras.mixed_precision.Policy("mixed_float16")
tf.keras.mixed_precision.set_global_policy(policy)

gpus = tf.config.list_physical_devices("GPU")
if gpus:
    print("MPS GPU available for training")
else:
    print("Using CPU, training will be slower")
"""
num_tokens = 10000
embedding_matrix = np.load("embeddings_matrix.npy")
books = pd.read_csv(
    "https://raw.githubusercontent.com/malcolmosh/goodbooks-10k/master/books_enriched.csv",
    index_col=[0],
)
unique_books_ids = books["book_id"].unique().tolist()


class ItemTower(tf.keras.Model):
    def __init__(
        self,
        unique_book_ids,
        num_tokens,
        embedding_matrix,
        user_emb_dim=32,
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
                tf.keras.layers.Dense(user_emb_dim),
            ]
        )

    def call(self, inputs):
        item_embedding = self.book_embedding(inputs["book_id"])
        rating = tf.expand_dims(inputs["avg_rating"], -1)
        norm_rating = rating - 3.0
        concat_input = tf.concat([item_embedding, norm_rating], axis=-1)
        item_emb = self.final_layers(concat_input)
        return item_emb


# Suppose your inputs are single book IDs and average ratings
sample_input = {
    "book_id": tf.constant([1, 2, 3], dtype=tf.int32),
    "avg_rating": tf.constant([4.5, 3.0, 5.0], dtype=tf.float32),
}

item_model = ItemTower(
    unique_book_ids=unique_books_ids,
    num_tokens=num_tokens,
    embedding_matrix=embedding_matrix,
)

# Run the model once to build it
_ = item_model(sample_input)

# Now you can see summary of layers
item_model.summary()


class UserTower(tf.keras.Model):
    def __init__(
        self,
        unique_book_ids,
        num_tokens,
        embedding_matrix,
        user_emb_dim=32,
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
                    output_dim=embedding_matrix.shape[1],  # 1536 for OpenAI embeddings
                    embeddings_initializer=tf.keras.initializers.Constant(
                        embedding_matrix
                    ),
                    mask_zero=True,
                    trainable=True,
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
                tf.keras.layers.Dense(user_emb_dim),
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


sample_input = {
    "user_history": tf.constant(
        [[1, 2, 3, 4, 5], [6, 7, 8, 9, 10], [11, 12, 13, 14, 0]], dtype=tf.int32
    ),
    "history_ratings": tf.constant(
        [
            [5.0, 4.0, 3.0, 4.0, 5.0],
            [3.0, 3.5, 4.0, 2.0, 3.0],
            [4.5, 5.0, 4.0, 5.0, 0.0],
        ],
        dtype=tf.float32,
    ),
}

user_model = UserTower(
    unique_book_ids=unique_books_ids,
    num_tokens=num_tokens,
    embedding_matrix=embedding_matrix,
)

# Call once to build the model
_ = user_model(sample_input)

# Now you can call summary
user_model.summary()


ratings = pd.read_csv(
    "https://raw.githubusercontent.com/malcolmosh/goodbooks-10k/master/ratings.csv"
)

# Count number of books per user
user_counts = ratings.groupby("user_id").size()
users_with_50_plus = user_counts[user_counts >= 50].index.tolist()
print("Total users:", len(user_counts))
print(f"Users with >=50 books: {len(users_with_50_plus)}")

"""
