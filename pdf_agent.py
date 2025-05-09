import re,ast
from langchain_community.vectorstores import FAISS
from transformers import DPRContextEncoder, DPRContextEncoderTokenizer, DPRQuestionEncoder, DPRQuestionEncoderTokenizer
from langchain.schema import Document as LangDocument
from langchain_community.retrievers import BM25Retriever
from langchain.retrievers import EnsembleRetriever
from langchain_core.retrievers import BaseRetriever
from langchain_core.runnables import RunnableParallel, RunnablePassthrough
from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import ChatPromptTemplate
from sentence_transformers.cross_encoder import CrossEncoder
from langchain_openai import ChatOpenAI
import os,faiss
import numpy as np
from cachetools import TTLCache
import hashlib
from dotenv import load_dotenv
load_dotenv()
os.environ["HUGGINGFACE_HUB_TOKEN"] = os.getenv("HUGGINGFACE_HUB_TOKEN")
os.environ["OPENAI_API_KEY"] = os.getenv("OPENAI_API_KEY")


context_tokenizer = DPRContextEncoderTokenizer.from_pretrained("facebook/dpr-ctx_encoder-single-nq-base")
context_model = DPRContextEncoder.from_pretrained("facebook/dpr-ctx_encoder-single-nq-base")
question_tokenizer = DPRQuestionEncoderTokenizer.from_pretrained("facebook/dpr-question_encoder-single-nq-base")
question_model = DPRQuestionEncoder.from_pretrained("facebook/dpr-question_encoder-single-nq-base")
embedding_cache = TTLCache(maxsize=100, ttl=86400)

llm = ChatOpenAI(model_name="gpt-4o", temperature=0)

cross_encoder = CrossEncoder('cross-encoder/ms-marco-MiniLM-L-6-v2')
class DPRRetriever(BaseRetriever):
    index: faiss.Index
    documents: dict
    q_model: DPRQuestionEncoder
    q_tokenizer: DPRQuestionEncoderTokenizer
    k: int = 2

    def __init__(self, index, documents, k=2, **kwargs):
        super().__init__(
            index=index,
            documents=documents,
            q_model=question_model,  # Use global model
            q_tokenizer=question_tokenizer,  # Use global tokenizer
            k=k,
            **kwargs
        )

    def _get_relevant_documents(self, query: str) -> list[LangDocument]:
        q_inputs = self.q_tokenizer(query, return_tensors="pt", truncation=True, padding=True, max_length=512)
        q_embedding = self.q_model(**q_inputs).pooler_output.detach().numpy()
        try:
            distances, indices = self.index.search(q_embedding, self.k)
            valid_indices = [i for i in indices[0] if i >= 0 and i in self.documents]
            if not valid_indices:
                print("No valid documents found in FAISS search.")
                return [LangDocument(page_content="No relevant documents found", metadata={})]
            return [self.documents[i] for i in valid_indices]
        except Exception as e:
            print(f"Error in FAISS search: {str(e)}")
            return [LangDocument(page_content=f"Search error: {str(e)}", metadata={})]
    
global_chunked_documents = []
global_documents = ""
global_embeddings = None
global_faiss_index = None


def embed_with_dpr(documents):
    embeddings = []
    for doc in documents:
        inputs = context_tokenizer(doc.page_content, return_tensors="pt", truncation=True, padding=True, max_length=512)
        embedding = context_model(**inputs).pooler_output.detach().numpy()
        embeddings.append(embedding[0])
    return np.array(embeddings)


class PdfProcessing:
    def __init__(self, query: str, user_id: str, session_id: str,parsed_path:str, context:str):
        self.query = query
        self.user_id = user_id
        self.session_id = session_id
        self.parsed_path = parsed_path
        self.context = context
        
        
    def initialize_documents(self, file_content: str):
        """Initialize documents, embeddings, FAISS index, and retriever with caching."""
        # Create a unique cache key based on file content
        content_hash = hashlib.sha256(file_content.encode()).hexdigest()
        cache_key = f"{self.user_id}:{self.session_id}:{content_hash}"

        # Check cache for embeddings and documents
        if cache_key in embedding_cache:
            print("Using cached embeddings and FAISS index.")
            self.chunked_documents, self.embeddings, self.faiss_index = embedding_cache[cache_key]
            self._initialize_retrievers()
            return

        self.chunked_documents = []
        pattern = r"====CHUNK START====\s*Content:\s*(.*?)\s*Metadata:\s*(.*?)\s*====CHUNK END===="
        matches = re.findall(pattern, file_content, re.DOTALL)
        
        for content_text, metadata_str in matches:
            content_text = content_text.strip()
            
            metadata_str = metadata_str.strip()
            try:
                metadata = ast.literal_eval(metadata_str)
            except Exception as e:
                print(f"Error parsing metadata: {e}")
                metadata = {}
            doc = LangDocument(page_content=content_text, metadata=metadata)
            self.chunked_documents.append(doc)
        
        # Compute embeddings
        self.embeddings = embed_with_dpr(self.chunked_documents)
        
        # Initialize FAISS index
        dimension = self.embeddings.shape[1]
        index = faiss.IndexFlatL2(dimension)
        index.add(self.embeddings)
        self.faiss_index = FAISS.from_embeddings(
            [(doc.page_content, emb) for doc, emb in zip(self.chunked_documents, self.embeddings)],
            embedding=None
        )

        # Cache the results
        embedding_cache[cache_key] = (self.chunked_documents, self.embeddings, self.faiss_index)
        
        # Initialize retrievers
        self._initialize_retrievers()

    def _initialize_retrievers(self):
        """Initialize retrievers for this instance."""
        vector_retriever = DPRRetriever(
            index=self.faiss_index.index,
            documents={i: doc for i, doc in enumerate(self.chunked_documents)},
            k=2
        )
        bm25_retriever = BM25Retriever.from_texts(
            [doc.page_content for doc in self.chunked_documents]
        )
        bm25_retriever.k = 1
        self.retriever = EnsembleRetriever(
            retrievers=[bm25_retriever, vector_retriever],
            weights=[0.3, 0.7]
        )

    def process(self) -> tuple[str, str]:
        print(self.query)
        print(self.parsed_path)
        file_content = self.xyz(self.parsed_path)
        
        # Update global documents if file_content differs
        if file_content != global_documents:
            self.initialize_documents(file_content)

        custom_prompt = """You are an assistant tasked with answering the user's question based on the provided context and chat history. Follow these instructions:
        1. Retrieve the full content of relevant chunks, including all tabular data in markdown format like '|', without omission.
        2. Avoid speculation or adding information not present in the context.

        Context:
        {context}
        Question:
        {question}

        Response:
        """
        prompt = ChatPromptTemplate.from_template(custom_prompt)
        
        rag_chain_from_docs = (
            RunnablePassthrough.assign(context=(lambda x: self.format_docs(x["context"])))
            | prompt
            | llm
            | StrOutputParser()
        )

        rag_chain_with_source = RunnableParallel(
            {"context": self.retriever, "question": RunnablePassthrough()}
        ).assign(answer=rag_chain_from_docs)

        final_response = self.cot_rag_with_initial_context(
            self.query, rag_chain_with_source,self.context
        )
        return final_response

    def format_docs(self, docs):
        return "\n\n".join(doc.page_content for doc in docs)
    
    def xyz(self, parsed_path) -> str:
        print(parsed_path)
        try:
            with open(parsed_path, 'r', encoding='utf-8') as file:
                file_content = file.read()
            return file_content
        except FileNotFoundError:
            raise FileNotFoundError(f"File not found at {parsed_path}")
        except IOError as e:
            raise IOError(f"Error reading file at {parsed_path}: {str(e)}")
    
    def cot_rag_with_initial_context(self, question, custom_query_engine, initial_context, max_iterations=1):
        # Ensure question is a string
        print(question)
        question = question.tool_calls[0]["args"].get("query")
        print(question)
        try:
            print("Invoking custom_query_engine...")
            retrieved_content = custom_query_engine.invoke(question)
            print(f"Retrieved content: {retrieved_content}")
        except Exception as e:
            print(f"Error in custom_query_engine.invoke: {str(e)}")
            import traceback
            traceback.print_exc()
            return {
                "response": f"Error: {str(e)}",
                "refined_context": "No context available due to processing error"
            }

        reranked_docs = self.rerank_documents(question, retrieved_content["context"])
        context = self.format_docs(reranked_docs)

        refined_context = llm.invoke(f"""
        You are an expert at extracting relevant information from text and providing guidance to find it within the source.
        Source: {context}
        Question: {question}
        Task: Identify the single, most relevant section of the Source that directly answers the Question. Return that section verbatim (without any modification) enclosed in a markdown quote. Also, provide a clear, concise directional cue to help the reader quickly locate this section within the document.
        Output Format:
        Relevant Section:
        > "Exact text from the source, verbatim, in markdown quote format"
        Location: [Directional cue, e.g., "Near the beginning"]
        If no section of the Source directly answers the Question, state "No relevant information found."
        """).content

        output = llm.invoke(f"""
        Instruction for CoT-based response generation:
        1. Thoroughly read the provided context.
        2. Use the context to answer the given question explicitly and accurately without missing any information covering all relevant aspects. Prioritize detailed information in responses.
        3. If the context is partially relevant to the question, return the partially retrieved response.
        4. If the context is completely irrelevant to the question only then, return "I tried reviewing the context provided, but it does not appear to contain information directly relevant to the question."
        Context:
        {context}
        Query:
        {question}
        """).content

        initial_response = output
        reasoning_chain = [initial_response]

        for iteration in range(1, max_iterations + 1):
            print(f"\nStep {iteration}: Refining the response...")
            output = llm.invoke(f"""\
            Response:
            {reasoning_chain[-1]}
            Context:
            {context}
            1. Check whether the response is verifiable from the context. If it is not, modify the response to make it completely accurate based on the context.
            2. Revise the response to fill in any gaps or missing information using the context provided.
            3. Strictly adhere to the information available in the provided Context and remove any information not present in the Context.
            4. Provide the revised response quoting from the context only without any explanations or additional instructions.
            5. Do not add unnecessary formatting or code blocks.
            """).content
            response = output
            reasoning_chain.append(response)
            if response == reasoning_chain[-2]:
                print("\nResponse has converged.")
                break

        final_response = response
        return final_response
 # Extract content from AIMessage
    
    def rerank_documents(self, query: str, candidate_docs: list):
        pairs = [(query, doc.page_content) for doc in candidate_docs]
        scores = cross_encoder.predict(pairs)
        for doc, score in zip(candidate_docs, scores):
            if doc.metadata is None:
                doc.metadata = {}
            doc.metadata["rerank_score"] = score
        ranked_docs = sorted(candidate_docs, key=lambda d: d.metadata.get("rerank_score", 0), reverse=True)
        return ranked_docs