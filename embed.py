# vector_db_manager.py
from sentence_transformers import SentenceTransformer
import chromadb
import re
import argparse
from langchain_core.prompts import ChatPromptTemplate
from langchain_groq import ChatGroq
from langchain.globals import set_debug,set_verbose

set_debug(True)  # Shows full prompt + chain logic
set_verbose(True) # Shows just the final prompt

# Configuration
MODEL_NAME = "all-MiniLM-L6-v2"
CHROMA_PATH = "chroma_db"
COLLECTION_NAME = "document_chunks"
DATA_FILE = "output.txt"


class VectorDBManager:
    def __init__(self, llm=None):
        self.model = SentenceTransformer(MODEL_NAME)
        self.client = chromadb.PersistentClient(path=CHROMA_PATH)
        self.collection = self.client.get_or_create_collection(
            name=COLLECTION_NAME,
            metadata={"hnsw:space": "cosine"}
        )
        self.llm = llm
        self.rating_prompt = ChatPromptTemplate.from_template(
            """Rate the relevance of this document chunk to the query on a scale 0-1:

            Query: {query}
            Chunk: {chunk}

            Consider these aspects:
            1. Direct keyword matches
            2. Semantic similarity
            3. Contextual understanding

            Return ONLY a numerical score between 0 (irrelevant) and 1 (perfect match) 
            with 2 decimal places. Example: 0.85"""
        )

    def _load_chunks(self, file_path):
        with open(file_path, 'r') as file:
            content = file.read()

        chunk_pattern = re.compile(r"Chunk \d+:(.*?)(?=\nChunk \d+:|$)", re.DOTALL)
        chunks = [match.group(1).strip() for match in chunk_pattern.finditer(content)]

        if not chunks:
            raise ValueError("No valid chunks found in the input file")

        return chunks

    def write_embeddings(self, file_path):
        chunks = self._load_chunks(file_path)
        embeddings = self.model.encode(chunks, show_progress_bar=True).tolist()

        existing_ids = set(self.collection.get()["ids"])
        new_ids = [f"chunk_{i}" for i in range(len(chunks)) if f"chunk_{i}" not in existing_ids]
        new_chunks = [chunks[i] for i in range(len(chunks)) if f"chunk_{i}" not in existing_ids]
        new_embeddings = [embeddings[i] for i in range(len(chunks)) if f"chunk_{i}" not in existing_ids]

        if new_ids:
            self.collection.add(
                documents=new_chunks,
                embeddings=new_embeddings,
                ids=new_ids
            )
            print(f"Added {len(new_ids)} new chunks to collection")
        else:
            print("No new chunks to add")

        print(f"Total chunks in DB: {self.collection.count()}")

    def _rate_relevance(self, query, chunk_text):
        if not self.llm:
            raise ValueError("LLM not initialized for rating")

        prompt = self.rating_prompt.format(query=query, chunk=chunk_text)
        response = self.llm.invoke(prompt)

        try:
            return float(re.search(r"\d\.\d{2}", response.content).group())
        except (ValueError, AttributeError):
            return 0.0

    def _basic_retrieve(self, query, top_k):
        query_embedding = self.model.encode(query).tolist()

        results = self.collection.query(
            query_embeddings=[query_embedding],
            n_results=top_k,
            include=["documents", "distances"]
        )

        return [
            {
                "score": 1 - distance,
                "text": document,
                "position": int(re.search(r"chunk_(\d+)", id).group(1))
            }
            for id, document, distance in zip(
                results["ids"][0],
                results["documents"][0],
                results["distances"][0]
            )
        ]

    def retrieve(self, query, top_k=10, rating=False):
        results = self._basic_retrieve(query, top_k)

        if rating:
            if not self.llm:
                raise ValueError("LLM required for rating - initialize with llm parameter")

            for result in results:
                result["llm_score"] = self._rate_relevance(query, result["text"])

        return results


def main():
    parser = argparse.ArgumentParser(description="Vector DB Manager")
    subparsers = parser.add_subparsers(dest="command", required=True)

    write_parser = subparsers.add_parser("write", help="Write embeddings to DB")
    write_parser.add_argument("--file", default=DATA_FILE, help="Input file path")

    retrieve_parser = subparsers.add_parser("retrieve", help="Search the DB")
    retrieve_parser.add_argument("query", help="Search query")
    retrieve_parser.add_argument("--top_k", type=int, default=10, help="Number of results")
    retrieve_parser.add_argument("--rating", action="store_true", help="Enable LLM relevance rating")

    args = parser.parse_args()

    llm = ChatGroq(
        model_name="deepseek-r1-distill-llama-70b",
        temperature=0,
        max_tokens=None
    )

    manager = VectorDBManager(llm=llm)

    if args.command == "write":
        manager.write_embeddings(args.file)
    elif args.command == "retrieve":
        results = manager.retrieve(args.query, args.top_k, args.rating)
        print(f"\nTop {args.top_k} results for: '{args.query}'\n")

        for idx, result in enumerate(results, 1):
            score_info = f"Vector score: {result['score']:.4f}"
            if args.rating:
                score_info += f" | LLM relevance: {result.get('llm_score', 0.0):.4f}"

            print(f"Rank {idx} ({score_info})")
            print(f"Chunk Position: {result['position']}")
            print(f"Content: {result['text'][:200]}...\n")


if __name__ == "__main__":
    main()
