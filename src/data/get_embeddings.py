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

books_df.head()


def format_book_text(row):
    title = f"Title: {row['title']}."
    overview = (
        f"Overview: {row['description'].lower()}."
        if pd.notna(row.get("description"))
        else ""
    )
    genres = (
        f"Genres: {', '.join(row['genres'])}."
        if isinstance(row["genres"], list)
        else ""
    )
    

    return " ".join([title, overview, genres]).strip()


def get_embedding(text, model="text-embedding-3-small"):
    text = text.replace("\n", " ")
    return client.embeddings.create(input=[text], model=model).data[0].embedding


embeddings_list = []

for idx, row in books_df.iterrows():
    print(f"Embedding row {idx}: {row['title']}")
    book_text = format_book_text(row)
    emb = get_embedding(book_text)
    embeddings_list.append(emb)

embeddings_matrix = np.vstack(embeddings_list)

np.save("embeddings_matrix.npy", embeddings_matrix)
print("Embeddings matrix shape:", embeddings_matrix.shape)
