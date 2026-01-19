import tensorflow as tf
from src.data.data_ingestion import prepare_datasets


loaded = tf.saved_model.load("export")

t, v, c = prepare_datasets()

for batch in v.take(1):
    y_true = batch["rating"]
    y_pred = loaded(batch)

print("Real ratings:", y_true.numpy().flatten()[:10])
print("Predicted:", y_pred.numpy().flatten()[:10])

for book in [10, 100, 500, 999]:
    test = {
        "user_history": tf.ragged.constant([[1, 2, 3, 4, 5]]),
        "history_ratings": tf.ragged.constant([[5.0, 4.0, 3.0, 4.5, 5.0]]),
        "book_id": tf.constant([book]),
        "avg_rating": tf.constant([4.0]),
        "rating": tf.constant([0.0]),
    }
    score = loaded(test)
    print(book, float(score))
