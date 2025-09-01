from llama_cpp import Llama
from sentence_transformers import SentenceTransformer, CrossEncoder
import chromadb
from chromadb.utils import embedding_functions
import pandas as pd
from typing import List, Dict
from config_loader import config

class RAGMedicalQA:
    def __init__(self):
        # Initialize LLM
        print("Initializing LLM...")
        self._init_llm()
        
        # Initialize embedding model
        print("Initializing embedding model...")
        self._init_embeddings()
        
        # Initialize vector store
        print("Initializing vector store...")
        self._init_vector_store()
        
        # Initialize reranker
        print("Initializing reranker...")
        self.reranker = CrossEncoder(config.rag['rerank_model'])
        
        print("RAG Medical QA system initialized!")

    def _init_llm(self):
        """Initialize the GGUF model using llama.cpp"""
        # Download model if needed
        model_path = config.llm['model_path']
        print('asdfasdf:', model_path)
        
        # Initialize llama.cpp model
        self.model = Llama(
            model_path=model_path,
            n_gpu_layers=config.llm['n_gpu_layers'],
            n_ctx=config.llm['n_ctx'],
            n_batch=config.llm['n_batch'],
        )

    def _init_embeddings(self):
        """Initialize the embedding model"""
        self.embedding_model = SentenceTransformer(config.vector_db['embedding_model'])
        
    def _init_vector_store(self):
        """Initialize ChromaDB vector store"""
        self.chroma_client = chromadb.PersistentClient(path="data/vector_store")
        
        # Create or get collection
        embedding_fn = embedding_functions.SentenceTransformerEmbeddingFunction(
            model_name=config.vector_db['embedding_model']
        )
        
        self.collection = self.chroma_client.get_or_create_collection(
            name=config.vector_db['collection_name'],
            embedding_function=embedding_fn
        )

    def index_documents(self, df: pd.DataFrame):
        """Index documents into the vector store"""
        print("Indexing documents...")
        
        # Combine questions and answers for context
        documents = []
        metadatas = []
        ids = []
        
        for idx, row in df.iterrows():
            doc = f"Question: {row['question']}\nAnswer: {row['answer']}"
            documents.append(doc)
            metadatas.append({"source": "medical_qa", "id": str(idx)})
            ids.append(f"doc_{idx}")
        
        # Add documents to vector store in batches
        batch_size = 100
        for i in range(0, len(documents), batch_size):
            end_idx = min(i + batch_size, len(documents))
            self.collection.add(
                documents=documents[i:end_idx],
                metadatas=metadatas[i:end_idx],
                ids=ids[i:end_idx]
            )
        
        print(f"Indexed {len(documents)} documents")

    def _get_relevant_context(self, question: str) -> str:
        """Retrieve and rerank relevant context"""
        # Get top-k documents from vector store
        results = self.collection.query(
            query_texts=[question],
            n_results=config.vector_db['top_k']
        )
        
        if not results['documents'][0]:
            return ""
        
        # Rerank results
        pairs = [(question, doc) for doc in results['documents'][0]]
        scores = self.reranker.predict(pairs)
        
        # Sort by score and take top k
        ranked_results = sorted(zip(scores, results['documents'][0]), reverse=True)
        top_docs = [doc for _, doc in ranked_results[:config.rag['rerank_top_k']]]
        
        # Combine context
        context = "\n\n".join(top_docs)
        return context

    def generate_response(self, question: str) -> Dict:
        """Generate a response using RAG"""
        # Get relevant context
        context = self._get_relevant_context(question)
        
        # Prepare prompt
        prompt = config.llm['template'].format(
            context=context,
            question=question
        )
        
        # Generate response using llama.cpp
        response = self.model(
            prompt,
            max_tokens=config.llm['max_tokens'],
            temperature=config.llm['temperature'],
            top_p=config.llm['top_p'],
            stop=["Human:", "System:"],  # Stop at next turn
            echo=False  # Don't include prompt in output
        )
        
        # Extract the generated text
        answer = response['choices'][0]['text'].strip()
        
        return {
            "answer": answer,
            "context": context
        }

# main function for testing
if __name__ == '__main__':
    # Load data from configured path
    df = pd.read_csv(config.data['raw_data_path'])

    # Initialize RAGMedicalQA
    rag_qa = RAGMedicalQA()

    # Index the test documents
    rag_qa.index_documents(df)

    # Test a single inference
    test_question = "How do I know if I have diabetes?"
    result = rag_qa.generate_response(test_question)

    print("Generated Answer:", result["answer"])
    print("Retrieved Context:", result["context"])