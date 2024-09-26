import uuid
import hnswlib
from typing import List, Dict
from unstructured.partition.html import partition_html
from unstructured.chunking.title import chunk_by_title
import cohere  # Assuming 'cohere' is the package name for Cohere API

co = cohere.Client("D14bT4Bm9SoiXE5ioVryf2DGOyIw1yjm1ccR0giQ")

class Vectorstore:
    def __init__(self, raw_documents: List[Dict[str, str]]):
        self.raw_documents = raw_documents
        self.docs = []
        self.docs_embs = []
        self.retrieve_top_k = 10
        self.rerank_top_k = 3
        self.load_and_chunk()
        self.embed()
        self.index()

    def load_and_chunk(self) -> None:
        """
        Loads the text from the sources and chunks the HTML content.
        """
        print("Loading documents...")

        for raw_document in self.raw_documents:
            elements = partition_html(url=raw_document["url"])
            chunks = chunk_by_title(elements)
            for chunk in chunks:
                self.docs.append(
                    {
                        "title": raw_document["title"],
                        "text": str(chunk),
                        "url": raw_document["url"],
                    }
                )

    def embed(self) -> None:
        """
        Embeds the document chunks using the Cohere API.
        """
        print("Embedding document chunks...")

        batch_size = 90
        self.docs_len = len(self.docs)
        for i in range(0, self.docs_len, batch_size):
            batch = self.docs[i : min(i + batch_size, self.docs_len)]
            texts = [item["text"] for item in batch]
            docs_embs_batch = co.embed(
                texts=texts, model="embed-english-v3.0", input_type="search_document"
            ).embeddings
            self.docs_embs.extend(docs_embs_batch)
            
    def index(self) -> None:
        """
        Indexes the documents for efficient retrieval using hnswlib.
        """
        print("Indexing documents...")

        self.idx = hnswlib.Index(space="ip", dim=1024)
        # self.idx = hnswlib.Index(space="cosine", dim=768)  # Adjust space and dim according to your embedding dimensions
        self.idx.init_index(max_elements=self.docs_len, ef_construction=512, M=64)
        self.idx.add_items(self.docs_embs, list(range(len(self.docs_embs))))

        print(f"Indexing complete with {self.idx.get_current_count()} documents.")
        
    # def retrieve(self, query: str) -> List[Dict[str, str]]:
    #     """
    #     Retrieves document chunks based on the given query.

    #     Parameters:
    #     query (str): The query to retrieve document chunks for.

    #     Returns:
    #     List[Dict[str, str]]: A list of dictionaries representing the retrieved document chunks, with 'title', 'text', and 'url' keys.
    #     """
    #     print(f"Retrieving documents for query: {query}")

    #     # Dense retrieval
    #     query_emb = co.embed(
    #         texts=[query], model="embed-english-v3.0", input_type="search_query"
    #     ).embeddings

    #     doc_ids = self.idx.knn_query(query_emb, k=self.retrieve_top_k)[0][0]
        
    #     # Convert doc_ids to a list of integers (if necessary)
    #     # doc_ids = list(doc_ids)

    #     # Reranking
    #     rank_fields = ["title", "text"]  # We'll use the title and text fields for reranking

    #     docs_to_rerank = [self.docs[doc_id] for doc_id in doc_ids]

    #     rerank_results = co.rerank(
    #         query=query,
    #         documents=docs_to_rerank,
    #         top_n=self.rerank_top_k,
    #         model="rerank-english-v3.0",
    #         rank_fields=rank_fields
    #     )
        
    #     # doc_ids_reranked = [result['id'] for result in rerank_results]
    #     doc_ids_reranked = [result.index for result in rerank_results]

    #     docs_retrieved = []
    #     for doc_id in doc_ids_reranked:
    #         docs_retrieved.append(
    #             {
    #                 "title": self.docs[doc_id]["title"],
    #                 "text": self.docs[doc_id]["text"],
    #                 "url": self.docs[doc_id]["url"],
    #             }
    #         )

    #     return docs_retrieved
    
    def retrieve(self, query: str) -> List[Dict[str, str]]:
        """
        Retrieves document chunks based on the given query.

        Parameters:
        query (str): The query to retrieve document chunks for.

        Returns:
        List[Dict[str, str]]: A list of dictionaries representing the retrieved document chunks, with 'title', 'text', and 'url' keys.
        """
        print(f"Retrieving documents for query: {query}")

        # Dense retrieval
        query_emb = co.embed(
            texts=[query], model="embed-english-v3.0", input_type="search_query"
        ).embeddings

        # Ensure that doc_ids are properly extracted as integers
        doc_ids = self.idx.knn_query(query_emb, k=self.retrieve_top_k)[0][0].tolist()
        
        # Reranking
        rank_fields = ["title", "text"]  # We'll use the title and text fields for reranking

        docs_to_rerank = [self.docs[doc_id] for doc_id in doc_ids]

        rerank_results = co.rerank(
            query=query,
            documents=docs_to_rerank,
            top_n=self.rerank_top_k,
            model="rerank-english-v3.0",
            rank_fields=rank_fields
        )

        # Print the rerank results to inspect its structure
        print(f"Rerank results: {rerank_results}")

        # Extracting document indices from rerank results properly
        doc_ids_reranked = [result.index for result in rerank_results.results]
        
        print(f"Reranked doc_ids: {doc_ids_reranked}")

        docs_retrieved = []
        for doc_id in doc_ids_reranked:
            docs_retrieved.append(
                {
                    "title": self.docs[doc_id]["title"],
                    "text": self.docs[doc_id]["text"],
                    "url": self.docs[doc_id]["url"],
                }
            )

        return docs_retrieved



# Example usage:
raw_documents = [
    {
        "title": "Text Embeddings",
        "url": "https://docs.cohere.com/docs/text-embeddings"
    },
    {
        "title": "Similarity Between Words and Sentences",
        "url": "https://docs.cohere.com/docs/similarity-between-words-and-sentences"
    },
    {
        "title": "The Attention Mechanism",
        "url": "https://docs.cohere.com/docs/the-attention-mechanism"
    },
    {
        "title": "Transformer Models",
        "url": "https://docs.cohere.com/docs/transformer-models"
    }
]

# Create an instance of Vectorstore
vectorstore = Vectorstore(raw_documents)

vectorstore.retrieve("multi-head attention definition")

class Chatbot:
    def __init__(self, vectorstore: Vectorstore):
        """
        Initializes an instance of the Chatbot class.

        Parameters:
        vectorstore (Vectorstore): An instance of the Vectorstore class.

        """
        self.vectorstore = vectorstore
        self.conversation_id = str(uuid.uuid4())
        
    def run(self):
        """
        Runs the chatbot application.

        """
        while True:
            # Get the user message
            message = input("User: ")

            # Typing "quit" ends the conversation
            if message.lower() == "quit":
                print("Ending chat.")
                break
            else:
                print(f"User: {message}")
                
            # Generate search queries, if any
            response = co.chat(message=message, search_queries_only=True)
        
        # If there are search queries, retrieve document chunks and respond
            if response.search_queries:
                print("Retrieving information...", end="")
            
                # Retrieve document chunks for each query
                documents = []
                for query in response.search_queries:
                    documents.extend(self.vectorstore.retrieve(query.text))
            
                # Use document chunks to respond
                response = co.chat_stream(
                    message=message,
                    model="command-r",
                    documents=documents,
                    conversation_id=self.conversation_id,
                )
            
            # If there is no search query, directly respond
            else:
                response = co.chat_stream(
                    message=message,
                    model="command-r",
                    conversation_id=self.conversation_id,
                )
                
             # Print the chatbot response, citations, and documents
            print("\nChatbot:")
            citations = []
            cited_documents = []
            
            # Display response
            for event in response:
                if event.event_type == "text-generation":
                    print(event.text, end="")
                elif event.event_type == "citation-generation":
                    citations.extend(event.citations)
                elif event.event_type == "search-results":
                    cited_documents = event.documents
            
            # Display citations and source documents
            if citations:
                print("\n\nCITATIONS:")
                for citation in citations:
                    print(citation)
            
                print("\nDOCUMENTS:")
                for document in cited_documents:
                    print(document)
            
            print(f"\n{'-'*100}\n")
            
    # Instantiate and run the chatbot
if __name__ == "__main__":
    chatbot = Chatbot(vectorstore)
    chatbot.run()
