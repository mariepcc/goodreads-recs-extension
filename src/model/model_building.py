import os
import numpy as np
import argparse
import tensorflow as tf
import tensorflow_recommenders as tfrs
import tensorflow_datasets as tfds
import pickle
import plotly.graph_objects as go
from data.data_ingestion import load_data, preprocess_data

BOOK_FEATURES = ["title", "genres", "authors", "average_rating", "ratings_count"]
USER_FEATURES = ["user_id"]


class UserModel(tf.keras.Model):
    def __init__(self, unique_user_ids, embedding_size=32):
        super().__init__()

        self.age_embedding = tf.keras.Sequential(
            [
                tf.keras.layers.IntegerLookup(
                    vocabulary=unique_user_ids, mask_token=None
                ),
                tf.keras.layers.Embedding(len(unique_user_ids) + 1, embedding_size),
            ]
        )

    def call(self, inputs):
        return self.user_embedding(inputs["user_id"])


class BookModel(tf.keras.Model):
    def __init__(
        self,
        unique_book_titles,
        additional_features,
        additional_feature_info,
        embedding_size=32,
    ):
        super().__init__()

        self.additional_embeddings = {}

        self.title_embedding = tf.keras.Sequential(
            [
                tf.keras.layers.StringLookup(
                    vocabulary=unique_book_titles, mask_token=None
                ),
                tf.keras.layers.Embedding(len(unique_book_titles) + 1, 32),
            ]
        )

        if "book_genres" in additional_features:
            self.additional_embeddings["book_genres"] = tf.keras.Sequential(
                [
                    tf.keras.layers.Embedding(
                        max(additional_feature_info["unique_book_genres"]) + 1,
                        embedding_size,
                    ),
                    tf.keras.layers.Lambda(lambda x: tf.math.reduce_mean(x, axis=1)),
                ]
            )
        if "average_rating" in additional_features:
            self.average_rating_normalizer = tf.keras.layers.Normalization(axis=None)
            self.average_rating_normalizer.adapt(
                additional_feature_info["bucketized_average_rating"]
            )
            self.additional_embeddings["bucketized_average_rating"] = (
                tf.keras.Sequential(
                    [self.average_rating_normalizer, tf.keras.layers.Reshape([1])]
                )
            )

    def call(self, inputs):
        return tf.concat(
            [self.title_embedding(inputs["title"])]
            + [
                self.additional_embeddings[k](inputs[k])
                for k in self.additional_embeddings
            ],
            axis=1,
        )


class QueryCandidateModel(tf.keras.Model):
    """Model for encoding user queries."""

    def __init__(self, layer_sizes, embedding_model):
        """Model for encoding user queries.

        Args:
          layer_sizes:
            A list of integers where the i-th entry represents the number of units
            the i-th layer contains.
        """
        super().__init__()

        # We first use the user model for generating embeddings.
        self.embedding_model = embedding_model

        # Then construct the layers.
        self.dense_layers = tf.keras.Sequential()

        # Use the ReLU activation for all but the last layer.
        for layer_size in layer_sizes[:-1]:
            self.dense_layers.add(tf.keras.layers.Dense(layer_size, activation="relu"))

        # No activation for the last layer.
        for layer_size in layer_sizes[-1:]:
            self.dense_layers.add(tf.keras.layers.Dense(layer_size))

    def call(self, inputs):
        feature_embedding = self.embedding_model(inputs)
        return self.dense_layers(feature_embedding)


class TwoTowersModel(tfrs.models.Model):
    def __init__(
        self,
        layer_sizes,
        books,
        unique_book_titles,
        n_unique_user_ids,
        embedding_size,
        additional_features,
        additional_feature_info,
    ):
        super().__init__()
        self.additional_features = additional_features
        self.query_model = QueryCandidateModel(
            layer_sizes,
            UserModel(
                n_unique_user_ids,
                embedding_size=embedding_size,
                additional_features=self.additional_features,
                additional_feature_info=additional_feature_info,
            ),
        )
        self.candidate_model = QueryCandidateModel(
            layer_sizes,
            BookModel(
                unique_book_titles,
                embedding_size=embedding_size,
                additional_features=self.additional_features,
                additional_feature_info=additional_feature_info,
            ),
        )
        self.task = tfrs.tasks.Retrieval(
            metrics=tfrs.metrics.FactorizedTopK(
                candidates=books.batch(128).map(self.candidate_model),
            ),
        )

    def compute_loss(self, features, training=False):
        query_embeddings = self.query_model(
            {
                "user_id": features["user_id"],
                **{
                    k: features[k]
                    for k in self.additional_features
                    if k in USER_FEATURES
                },
            }
        )
        book_embeddings = self.candidate_model(
            {
                "book_title": features["title"],
                **{
                    k: features[k]
                    for k in self.additional_features
                    if k in BOOK_FEATURES
                },
            }
        )
        return self.task(
            query_embeddings, book_embeddings, compute_metrics=not training
        )


class TwoTowersTrainer:
    def __init__(
        self, num_epochs, embedding_size, layer_sizes, additional_feature_sets, retrain
    ):
        self.num_epochs = num_epochs
        self.embedding_size = embedding_size
        self.layer_sizes = tuple(layer_sizes)
        self.additional_feature_sets = additional_feature_sets
        self.retrain = retrain

    def get_datasets(self):
        ratings_url = "https://raw.githubusercontent.com/malcolmosh/goodbooks-10k/master/ratings.csv"
        books_url = "https://raw.githubusercontent.com/malcolmosh/goodbooks-10k/master/books_enriched.csv"

        ratings_df = load_data(ratings_url)
        books_df = load_data(books_url)

        self.ratings = preprocess_data(books_df, ratings_df)

        tf.data.Dataset.save(self.ratings)

    def train_all_models(self):
        models = {}
        for additional_features in self.additional_feature_sets:
            model, history = self.get_two_tower_model(tuple(additional_features))
            models[tuple(additional_features)] = (model, history)
        return models

    def train_two_tower_model(self, additional_features, folder_name):
        trainset = (
            self.ratings.take(80_000)
            .shuffle(100_000)
            .apply(tf.data.experimental.dense_to_ragged_batch(2048))
            .cache()
        )
        testset = (
            self.ratings.skip(80_000)
            .take(20_000)
            .apply(tf.data.experimental.dense_to_ragged_batch(2048))
            .cache()
        )
        model = TwoTowersModel(
            self.layer_sizes,
            self.books,
            self.unique_book_titles,
            self.unique_user_ids,
            self.embedding_size,
            additional_features=additional_features,
            additional_feature_info=self.additional_feature_info,
        )
        model.compile(optimizer=tf.keras.optimizers.Adagrad(0.1), run_eagerly=True)
        model_history = model.fit(
            trainset,
            validation_data=testset,
            validation_freq=5,
            epochs=self.num_epochs,
            verbose=1,
        )
        model.task = tfrs.tasks.Retrieval()
        model.compile()
        tf.saved_model.save(model, f"{folder_name}/model")
        with open(f"{folder_name}/model_history.pkl", "wb") as f:
            pickle.dump(model_history.history, f)
        return tf.saved_model.load(f"{folder_name}/model"), model_history.history


def plot_training_runs(model_histories, datapane_token=None):
    first_key = list(model_histories.keys())[0]
    num_validation_runs = len(
        model_histories[first_key]["val_factorized_top_k/top_100_categorical_accuracy"]
    )
    epochs = [(x + 1) * 5 for x in range(num_validation_runs)]
    fig = go.Figure()
    for k, v in model_histories.items():
        fig.add_trace(
            go.Scatter(
                x=epochs,
                y=v["val_factorized_top_k/top_100_categorical_accuracy"],
                mode="lines",
                name="_".join(k),
            )
        )
    fig.update_layout(
        title="TFRS model comparison on Goodreads Dataset",
        xaxis_title="epoch",
        yaxis_title="validation top-100 accuracy",
        legend_title="Features used",
        font=dict(family="Courier New, monospace", size=12, color="RebeccaPurple"),
    )
    if datapane_token is not None:
        import datapane as dp

        os.system(f"datapane login --token={datapane_token}")
        dp.Report(dp.Plot(fig)).upload(name="Goodreads TFRS")
    return fig


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--num_epochs", type=int, default=10)
    parser.add_argument("--embedding_size", type=int, default=32)
    parser.add_argument("--layer_sizes", nargs="+", default=[32])
    parser.add_argument(
        "--additional_feature_sets",
        nargs="+",
        help="options: timestamp",
        action="append",
    )
    parser.add_argument("--generate_recommendations_for_user", type=int, default=42)
    parser.add_argument("--retrain", action="store_true")
    parser.add_argument("--datapane_token")
    args = parser.parse_args()

    if ["None"] in args.additional_feature_sets:
        args.additional_feature_sets.remove(["None"])
        args.additional_feature_sets.append([])
    else:
        args.additional_feature_sets = [
            x for x in args.additional_feature_sets if len(x) > 0
        ]

    model_trainer = TwoTowersTrainer(
        args.num_epochs,
        args.embedding_size,
        args.layer_sizes,
        args.additional_feature_sets,
        args.retrain,
    )
    models = model_trainer.train_all_models()
    fig = plot_training_runs({k: v[1] for k, v in models.items()}, args.datapane_token)
    fig.show()

    print(f"Run settings: {vars(args)}")
    if args.generate_recommendations_for_user:
        print(
            f"These are the ratings for user {args.generate_recommendations_for_user}:"
        )
        user_ratings_df = tfds.as_dataframe(
            model_trainer.ratings.filter(
                lambda x: x["user_id"] == str(args.generate_recommendations_for_user)
            )
        )
        user_ratings_df.sort_values("rating", ascending=False).to_csv(
            "user_1_ratings.csv", index=False
        )
        for model_name, model_obj in models.items():
            rating_idx = (
                tf.where(
                    model_trainer.all_ratings["user_id"]
                    == str(args.generate_recommendations_for_user).encode()
                )
                .numpy()
                .squeeze()[0]
            )
            user_details_for_query = {
                "user_id": [
                    model_trainer.all_ratings["user_id"][rating_idx].numpy().decode()
                ],
                **{
                    col_name: [model_trainer.all_ratings[col_name][rating_idx].numpy()]
                    for col_name in model_name
                    if col_name in USER_FEATURES
                },
            }
            model = model_obj[0]
            index = tfrs.layers.factorized_top_k.BruteForce()
            index.index_from_dataset(
                model_trainer.books.apply(
                    tf.data.experimental.dense_to_ragged_batch(100)
                ).map(model.candidate_model)
            )
            _, titles = index(model.query_model(user_details_for_query))
            title_names = np.array(
                [model_trainer.unique_book_titles[x] for x in titles.numpy().squeeze()]
            )
            print(
                f"Recommendations for model {model_name}, user {args.generate_recommendations_for_user}:\n"
                f" {title_names[:10]}"
            )
