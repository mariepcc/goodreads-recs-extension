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
    logger.info(f"Books loaded: {len(books)}")
    return books


def load_ratings():
    ratings = pd.read_csv(
        "https://raw.githubusercontent.com/malcolmosh/goodbooks-10k/master/ratings.csv"
    )
    logger.info(f"Ratings loaded: {len(ratings)}")
    return ratings


def merge_datasets(ratings, books):
    ratings = ratings.merge(
        books[["book_id", "average_rating"]], on="book_id", how="left"
    )
    logger.info("Ratings merged with books (avg_rating added).")
    return ratings


def split_users(ratings, train_ratio=0.8, seed=42):
    users = ratings["user_id"].unique()
    rng = np.random.default_rng(seed)
    rng.shuffle(users)

    users = users[:5000]

    n_train = int(len(users) * train_ratio)
    train_users = set(users[:n_train])
    test_users = set(users[n_train:])

    train_df = ratings[ratings["user_id"].isin(train_users)]
    test_df = ratings[ratings["user_id"].isin(test_users)]

    logger.info(f"Train users: {len(train_users)}, Test users: {len(test_users)}")
    return train_df, test_df


def make_leave_one_out(train_df):
    rows = []
    train_df = train_df.sample(frac=1, random_state=42)

    for user, user_ratings in train_df.groupby("user_id"):
        if len(user_ratings) < 10:
            continue

        target = user_ratings.iloc[0]
        history = user_ratings.iloc[1:21]

        rows.append(
            {
                "user_id": user,
                "user_history": history["book_id"].astype(int).tolist(),
                "history_ratings": history["rating"].astype(float).tolist(),
                "book_id": int(target["book_id"]),
                "rating": float(target["rating"]),
            }
        )

    df = pd.DataFrame(rows)
    return df


def df_to_tf_dataset(df, batch_size=64, shuffle=True):
    user_history = tf.keras.preprocessing.sequence.pad_sequences(
        df["user_history"].values, maxlen=20, padding="post"
    )
    history_ratings = tf.keras.preprocessing.sequence.pad_sequences(
        df["history_ratings"].values, maxlen=20, padding="post", dtype="float32"
    )

    features = {
        "user_history": tf.constant(user_history, dtype=tf.int32),
        "history_ratings": tf.constant(history_ratings, dtype=tf.float32),
        "book_id": tf.constant(df["book_id"].values, dtype=tf.int32),
        "rating": tf.constant(df["rating"].values, dtype=tf.float32),
    }

    ds = tf.data.Dataset.from_tensor_slices(features)
    if shuffle:
        ds = ds.shuffle(10000, seed=42)
    ds = ds.batch(batch_size)
    return ds


def prepare_datasets(batch_size=128):
    books = load_books()
    ratings = load_ratings()
    ratings = merge_datasets(ratings, books)

    train_df, test_df = split_users(ratings)

    loo_train_df = make_leave_one_out(train_df)
    loo_test_df = make_leave_one_out(test_df)

    train_ds = df_to_tf_dataset(loo_train_df, batch_size=batch_size)
    test_ds = df_to_tf_dataset(loo_test_df, batch_size=256)

    candidates_ds = tf.data.Dataset.from_tensor_slices(
        {
            "book_id": books["book_id"].values.astype("int32"),
        }
    ).batch(512)

    return train_ds, test_ds, candidates_ds


if __name__ == "__main__":
    train_ds, test_ds, candidates_ds = prepare_datasets()
    books = load_books()
    ratings = load_ratings()
    ratings = merge_datasets(ratings, books)

    train_df, test_df = split_users(ratings)

    loo_train_df = make_leave_one_out(train_df)
    train_ds = df_to_tf_dataset(loo_train_df, batch_size=256)

    for batch in train_ds.take(1):
        print(batch)

    print("Number of examples for evaluation:", len(loo_train_df))

    for batch in test_ds.take(1):
        print(batch["user_history"].shape)
        print(batch["book_id"].shape)
        print(batch["rating"].shape)
