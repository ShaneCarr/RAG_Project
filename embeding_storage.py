import openai
import psycopg2

def embed_text(text):
    resp = openai.Embedding.create(
        model="text-embedding-3-large",
        input=text
    )
    return resp['data'][0]['embedding']

# Example: store embeddings in pgvector
conn = psycopg2.connect("dbname=mydb user=me")
cur = conn.cursor()

def store_chunk(file, content):
    vector = embed_text(content)
    cur.execute(
        "INSERT INTO code_chunks (file, content, embedding) VALUES (%s, %s, %s)",
        (file, content, vector)
    )
    conn.commit()
