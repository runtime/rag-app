import argparse
import psycopg2
from psycopg2.extras import RealDictCursor
from langchain.prompts import ChatPromptTemplate
from langchain_community.embeddings import OpenAIEmbeddings
#from ollama import generate
from config import DB_HOST, DB_NAME, DB_USER, DB_PASS

PROMPT_TEMPLATE = """
Answer the question based only on the following context:

{context}

---

Answer the question based on the above context: {question}
"""

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("query_text", type=str, help="The query text.")
    args = parser.parse_args()
    query_text = args.query_text
    query_rag(query_text)

def query_rag(query_text: str):
    embedding_function = OpenAIEmbeddings().embed_query
    query_vector = embedding_function(query_text)

    # Query PostgreSQL for relevant chunks
    results = query_postgres(query_vector, top_n=5)
    context_text = "\n\n---\n\n".join([row['content'] for row in results])

    # Format the prompt
    prompt = PROMPT_TEMPLATE.format(context=context_text, question=query_text)
    print(f"Generated prompt:\n{prompt}")

    # Use OpenAI's GPT for the response
    response = query_openai(prompt)
    print(f"Response:\n{response}")

def query_postgres(query_vector, top_n=5):
    connection = psycopg2.connect(
        host=DB_HOST,
        database=DB_NAME,
        user=DB_USER,
        password=DB_PASS
    )
    cursor = connection.cursor(cursor_factory=RealDictCursor)

    query = """
    SELECT chunk_id, content, embedding <-> %s::vector AS distance
    FROM document_chunks
    ORDER BY distance ASC
    LIMIT %s;
    """
    cursor.execute(query, (query_vector, top_n))
    results = cursor.fetchall()
    cursor.close()
    connection.close()
    return results

def query_openai(prompt: str) -> str:
    openai.api_key = "<your-openai-api-key>"

    response = openai.ChatCompletion.create(
        model="gpt-4",  # You can also use gpt-3.5-turbo for faster and cheaper responses
        messages=[
            {"role": "system", "content": "You are an expert assistant."},
            {"role": "user", "content": prompt}
        ],
        temperature=0.7
    )
    return response['choices'][0]['message']['content']

if __name__ == "__main__":
    main()
