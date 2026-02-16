import logging
import pandas as pd
import numpy as np
import tensorflow as tf
from ast import literal_eval

logger = logging.getLogger("data_ingestion")
logger.setLevel(logging.DEBUG)


def load_books():
    books = pd.read_csv(
        "https://raw.githubusercontent.com/malcolmosh/goodbooks-10k/master/books_enriched.csv",
        index_col=[0],
        converters={"genres": literal_eval, "authors": literal_eval},
    )
    return books


def load_ratings():
    ratings = pd.read_csv(
        "https://raw.githubusercontent.com/malcolmosh/goodbooks-10k/master/ratings.csv"
    )
    return ratings


def merge_datasets(ratings, books):
    ratings = ratings.merge(
        books[["book_id", "average_rating"]], on="book_id", how="left"
    )
    return ratings


def split_users(ratings, train_ratio=0.8, seed=42):
    users = ratings["user_id"].unique()
    rng = np.random.default_rng(seed)
    rng.shuffle(users)
    # users = users[:15000]
    n_train = int(len(users) * train_ratio)
    train_users = set(users[:n_train])
    test_users = set(users[n_train:])
    train_df = ratings[ratings["user_id"].isin(train_users)]
    test_df = ratings[ratings["user_id"].isin(test_users)]
    return train_df, test_df


def make_leave_one_out(train_df):
    rows = []

    for user, user_ratings in train_df.groupby("user_id"):
        if len(user_ratings) < 51:
            continue

        target = user_ratings.iloc[-1]
        history = user_ratings.iloc[-51:-1]

        rows.append(
            {
                "user_id": user,
                "user_history": history["book_id"].astype(int).tolist(),
                "history_ratings": history["rating"].astype(float).tolist(),
                "book_id": int(target["book_id"]),
                "rating": float(target["rating"]),
                "avg_rating": float(target["average_rating"]),
            }
        )
    return pd.DataFrame(rows)


def df_to_tf_dataset(df, embedding_matrix, batch_size, shuffle=True):
    user_history_ids = tf.keras.preprocessing.sequence.pad_sequences(
        df["user_history"].values, maxlen=50, padding="post"
    )
    history_ratings = tf.keras.preprocessing.sequence.pad_sequences(
        df["history_ratings"].values, maxlen=50, padding="post", dtype="float32"
    )

    user_history_ids = tf.keras.preprocessing.sequence.pad_sequences(
        df["user_history"].values, maxlen=50, padding="post"
    )
    user_history_vectors = tf.nn.embedding_lookup(embedding_matrix, user_history_ids)
    target_book_vectors = tf.nn.embedding_lookup(embedding_matrix, df["book_id"].values)

    ratings = df["rating"].values.astype(np.float32)

    features = {
        "user_history": tf.cast(user_history_vectors, tf.float32),
        "user_history_ids": tf.cast(user_history_ids, tf.int32),
        "history_ratings": tf.cast(history_ratings, tf.float32),
        "book_emb": tf.cast(target_book_vectors, tf.float32),
        "avg_rating": tf.cast(df["avg_rating"].values, tf.float32),
        "rating": tf.cast(ratings, tf.float32),
        "book_id": tf.cast(df["book_id"].values, tf.int32),
    }

    ds = tf.data.Dataset.from_tensor_slices(features)
    if shuffle:
        ds = ds.shuffle(10000, seed=42)
    ds = ds.batch(batch_size)
    return ds


def prepare_datasets(embedding_matrix, batch_size=512):
    books = load_books()
    ratings = load_ratings()
    ratings = merge_datasets(ratings, books)
    train_df, test_df = split_users(ratings)

    loo_train_df = make_leave_one_out(train_df)
    loo_test_df = make_leave_one_out(test_df)

    train_ds = df_to_tf_dataset(loo_train_df, embedding_matrix, batch_size=batch_size)
    test_ds = df_to_tf_dataset(loo_test_df, embedding_matrix, batch_size=batch_size)

    candidates_ds = tf.data.Dataset.from_tensor_slices(
        {"book_emb": tf.cast(embedding_matrix, tf.float32)}
    ).batch(512)

    return train_ds, test_ds, candidates_ds
