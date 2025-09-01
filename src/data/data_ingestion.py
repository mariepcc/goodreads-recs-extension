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
    try:
        ratings_tf = tf.data.Dataset.from_tensor_slices(df.to_dict("list"))
        logger.debug("DataFrame converted to TensorFlow dataset")
        return ratings_tf
    except Exception as e:
        logger.error("Error converting DataFrame to TensorFlow dataset: %s", e)
        raise


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

        tf = df_to_tf_dataset(merged_df)
        return tf
    except KeyError as e:
        logger.error("Missing column in the dataframe: %s", e)
        raise
    except Exception as e:
        logger.error("Unexpected error during preprocessing: %s", e)
        raise
