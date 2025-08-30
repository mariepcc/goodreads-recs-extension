import pandas as pd
import logging

import tensorflow as tf

logger = logging.getLogger("data_ingestion")
logger.setLevel(logging.DEBUG)


def load_data(data_url: str) -> pd.DataFrame:
    """Load data from a CSV file."""
    try:
        df = pd.read_csv(data_url)
        logger.debug("Data loaded from %s", data_url)
        return df
    except pd.errors.ParserError as e:
        logger.error("Failed to parse the CSV file: %s", e)
        raise
    except Exception as e:
        logger.error("Unexpected error occurred while loading the data: %s", e)
        raise


def df_to_tf_dataset(df):
    """Convert a DataFrame to a TensorFlow dataset."""
    df["authors"] = df["authors"].apply(
        lambda x: ",".join(x) if isinstance(x, list) else ""
    )
    df["genres"] = df["genres"].apply(
        lambda x: ",".join(x) if isinstance(x, list) else ""
    )

    return tf.data.Dataset.from_tensor_slices(
        {
            "user_id": df["user_id"].astype(str).values,
            "book_id": df["book_id"].astype(str).values,
            "title": df["title"].astype(str).values,
            "authors": df["authors"].astype(str).values,
            "genres": df["genres"].astype(str).values,
            "average_rating": df["average_rating"].values.astype("float32"),
            "ratings_count": df["ratings_count"].values.astype("float32"),
        }
    )


def preprocess_data(
    books_df: pd.DataFrame, ratings_df: pd.DataFrame
) -> tf.data.Dataset:
    """Preprocess books and ratings datasets for a two-tower recommender."""
    try:
        logger.debug("Starting data preprocessing...")

        books_features = books_df[
            ["book_id", "title", "authors", "genres", "average_rating", "ratings_count"]
        ].copy()

        merged_df = ratings_df.merge(books_features, on="book_id", how="left")

        df = df_to_tf_dataset(merged_df)
    except KeyError as e:
        logger.error("Missing column in the dataframe: %s", e)
        raise
    except Exception as e:
        logger.error("Unexpected error during preprocessing: %s", e)
        raise

    return df
