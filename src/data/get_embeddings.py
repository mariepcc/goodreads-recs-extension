import numpy as np
from openai import OpenAI
import pandas as pd
from dotenv import load_dotenv
from ast import literal_eval

load_dotenv()
client = OpenAI()

books_df = pd.read_csv(
    "https://raw.githubusercontent.com/malcolmosh/goodbooks-10k/master/books_enriched.csv",
    index_col=[0],
    converters={"genres": literal_eval, "authors": literal_eval},
)


def format_book_text(row):
    title = f"Title: {row['title']}."
    desc = str(row.get("description", ""))[:1000]
    overview = f"Overview: {desc.lower()}."
    genres = (
        f"Genres: {', '.join(row['genres'])}."
        if isinstance(row["genres"], list)
        else ""
    )
    return " ".join([title, overview, genres]).strip()


def get_embeddings_batch(texts, model="text-embedding-3-small"):
    response = client.embeddings.create(input=texts, model=model)
    return [item.embedding for item in response.data]


all_embeddings = []
batch_size = 100

for i in range(0, len(books_df), batch_size):
    batch_rows = books_df.iloc[i : i + batch_size]
    texts = [format_book_text(row) for _, row in batch_rows.iterrows()]

    print(f"Processing batch {i} to {i + batch_size}...")
    try:
        batch_embs = get_embeddings_batch(texts)
        all_embeddings.extend(batch_embs)
    except Exception as e:
        print(f"Error at batch {i}: {e}")
        break

embeddings_matrix = np.array(all_embeddings, dtype=np.float32)

np.save("embeddings_matrix.npy", embeddings_matrix)
print("Final shape:", embeddings_matrix.shape)
