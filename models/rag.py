"""
Multi-PDF RAG System with Keyword Enhancement
This module creates a Retrieval-Augmented Generation system for multiple PDFs with keyword extraction
"""

import os
import glob
import json
import pickle
import numpy as np
import faiss
import re
import torch
from tqdm import tqdm
from typing import List, Tuple, Dict, Any, Optional, Union
import re 

# Try to import all required packages, with fallbacks
try:
    import PyPDF2
except ImportError:
    print("Warning: PyPDF2 not installed. PDF text extraction may be limited.")

try:
    from sentence_transformers import SentenceTransformer
except ImportError:
    print("Warning: sentence-transformers not installed. Please install it with: pip install sentence-transformers")

try:
    from langchain.text_splitter import RecursiveCharacterTextSplitter
except ImportError:
    print("Warning: langchain not installed. Please install with: pip install langchain")

try:
    from langchain_community.llms import HuggingFacePipeline
except ImportError:
    print("Warning: langchain_community not installed. Please install with: pip install langchain_community")

try:
    from transformers import AutoTokenizer, AutoModelForCausalLM, pipeline
except ImportError:
    print("Warning: transformers not installed. Please install with: pip install transformers")

try:
    from pdfminer.high_level import extract_text as pdfminer_extract_text
except ImportError:
    print("Warning: pdfminer.six not available, will use PyPDF2 only")
    pdfminer_extract_text = None

try:
    import nltk
    from nltk.corpus import stopwords
    # Check if necessary resources are downloaded
    try:
        nltk.data.find('tokenizers/punkt')
    except LookupError:
        nltk.download('punkt', quiet=True)
    try:
        nltk.data.find('corpora/stopwords')
    except LookupError:
        nltk.download('stopwords', quiet=True)
except ImportError:
    print("Warning: nltk not installed. Please install with: pip install nltk")

try:
    from keybert import KeyBERT
except ImportError:
    print("Warning: keybert not installed. Please install with: pip install keybert")


class RAGSystem:
    def __init__(self, pdf_folder='../backend/uploads', use_gpu=False):
        """Initialize the RAG System.
        
        Args:
            pdf_folder (str): Directory to store PDFs and processed data
            use_gpu (bool): Whether to use GPU for embedding and LLM
        """
        self.pdf_folder = pdf_folder
        self.indices_folder = os.path.join(pdf_folder, 'indices')
        self.keywords_folder = os.path.join(pdf_folder, 'keywords')
        self.embedding_model = None
        self.llm = None
        self.metadata_paths = []
        self.conversation_history = []
        self.use_gpu = use_gpu
        self.device = torch.device('cuda' if torch.cuda.is_available() and use_gpu else 'cpu')
        
        # Create necessary folders
        os.makedirs(pdf_folder, exist_ok=True)
        os.makedirs(self.indices_folder, exist_ok=True)
        os.makedirs(self.keywords_folder, exist_ok=True)
        
        print(f"Initialized RAG System using folder: {os.path.abspath(pdf_folder)}")
        print(f"Using device: {self.device}")
    
    def extract_text_from_pdf(self, pdf_path: str) -> str:
        """Extract text from a PDF file.
        
        Args:
            pdf_path (str): Path to the PDF file
            
        Returns:
            str: Extracted text from the PDF
        """
        filename = os.path.basename(pdf_path)
        print(f"Extracting text from {filename}...")
        
        # Method 1: PyPDF2
        if 'PyPDF2' in globals():
            try:
                pdf_reader = PyPDF2.PdfReader(pdf_path)
                pdf_text = ""
                for page_num in range(len(pdf_reader.pages)):
                    page = pdf_reader.pages[page_num]
                    pdf_text += page.extract_text() + "\n"
                    
                if len(pdf_text.strip()) > 100:  # Check if we got meaningful text
                    print(f"Successfully extracted {len(pdf_text)} characters using PyPDF2")
                    return pdf_text
            except Exception as e:
                print(f"PyPDF2 extraction failed: {e}")
        
        # Method 2: pdfminer.six
        if pdfminer_extract_text is not None:
            try:
                pdf_text = pdfminer_extract_text(pdf_path)
                if len(pdf_text.strip()) > 100:
                    print(f"Successfully extracted {len(pdf_text)} characters using pdfminer.six")
                    return pdf_text
            except Exception as e:
                print(f"pdfminer.six extraction failed: {e}")
        
        # Fallback: Simple file read
        try:
            with open(pdf_path, 'rb') as file:
                content = file.read()
                # Very crude text extraction
                pdf_text = str(content)
                print(f"Using basic binary read as fallback: {len(pdf_text)} characters")
                return pdf_text
        except Exception as e:
            print(f"Basic file read failed: {e}")
            
        print(f"WARNING: Could not extract text from {filename}")
        return ""
    
    def split_text(self, text: str, chunk_size: int = 500, chunk_overlap: int = 50) -> List[str]:
        """Split text into chunks using LangChain's splitter.
        
        Args:
            text (str): Text to split
            chunk_size (int): Size of each chunk
            chunk_overlap (int): Overlap between chunks
        
        Returns:
            List[str]: List of text chunks
        """
        text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=chunk_size,
            chunk_overlap=chunk_overlap,
            separators=["\n\n", "\n", ".", " ", ""]
        )
        chunks = text_splitter.split_text(text)
        return chunks
    
    def extract_keywords(self, chunks: List[str], top_n: int = 5) -> List[List[str]]:
        """Extract keywords from text chunks.
        
        Args:
            chunks (List[str]): List of text chunks
            top_n (int): Number of keywords to extract per chunk
        
        Returns:
            List[List[str]]: List of keywords for each chunk
        """
        if 'KeyBERT' not in globals():
            # Fallback using basic word frequency if KeyBERT is not available
            print("Using basic keyword extraction (KeyBERT not available)")
            all_keywords = []
            for chunk in tqdm(chunks, desc="Extracting keywords"):
                # Simple keyword extraction
                words = re.findall(r'\b[a-zA-Z]{3,15}\b', chunk.lower())
                if 'stopwords' in globals():
                    words = [w for w in words if w not in stopwords.words('english')]
                # Get most frequent words
                word_freq = {}
                for word in words:
                    word_freq[word] = word_freq.get(word, 0) + 1
                keywords = sorted(word_freq.items(), key=lambda x: x[1], reverse=True)
                keywords = [k for k, _ in keywords[:top_n]]
                all_keywords.append(keywords)
            return all_keywords
        
        # Initialize KeyBERT
        keyword_model = KeyBERT()
        
        # Extract keywords for each chunk
        all_keywords = []
        for chunk in tqdm(chunks, desc="Extracting keywords"):
            # Get keywords with scores
            keywords = keyword_model.extract_keywords(
                chunk, 
                keyphrase_ngram_range=(1, 2), 
                stop_words='english', 
                use_mmr=True, 
                diversity=0.7, 
                top_n=top_n
            )
            
            # Keep only the keywords (not scores)
            keywords_only = [keyword for keyword, _ in keywords]
            all_keywords.append(keywords_only)
        
        return all_keywords
    
    def create_embeddings(self, chunks: List[str], model_name: str = 'all-MiniLM-L6-v2') -> Tuple[Any, Any]:
        """Create embeddings and FAISS index for text chunks.
        
        Args:
            chunks (List[str]): List of text chunks
            model_name (str): Name of the SentenceTransformer model to use
        
        Returns:
            Tuple[Any, Any]: FAISS index and embedding model
        """
        # Initialize embedding model if not already done
        if self.embedding_model is None:
            self.embedding_model = SentenceTransformer(model_name, device=self.device)
        
        # Generate embeddings
        print(f"Creating embeddings for {len(chunks)} chunks...")
        embeddings = self.embedding_model.encode(chunks, show_progress_bar=True)
        
        # Normalize the embeddings for cosine similarity search
        faiss.normalize_L2(embeddings)
        
        # Create the FAISS index
        index = faiss.IndexFlatIP(embeddings.shape[1])  # Inner product for cosine similarity
        index.add(embeddings)
        
        return index, self.embedding_model
    
    def save_processed_data(self, pdf_path: str, chunks: List[str], keywords: List[List[str]], 
                          index: Any) -> str:
        """Save processed PDF data to disk.
        
        Args:
            pdf_path (str): Path to the PDF file
            chunks (List[str]): List of text chunks
            keywords (List[List[str]]): List of keywords for each chunk
            index (Any): FAISS index
            
        Returns:
            str: Path to the saved metadata file
        """
        # Get the base filename (without extension)
        base_filename = os.path.splitext(os.path.basename(pdf_path))[0]
        
        # Save chunks
        chunks_path = os.path.join(self.indices_folder, f"{base_filename}_chunks.pkl")
        with open(chunks_path, 'wb') as f:
            pickle.dump(chunks, f)
        
        # Save keywords
        keywords_path = os.path.join(self.keywords_folder, f"{base_filename}_keywords.json")
        with open(keywords_path, 'w') as f:
            json.dump(keywords, f)
        
        # Save FAISS index
        index_path = os.path.join(self.indices_folder, f"{base_filename}_index.faiss")
        faiss.write_index(index, index_path)
        
        # Create a metadata file with mapping info
        metadata = {
            "pdf_filename": os.path.basename(pdf_path),
            "num_chunks": len(chunks),
            "chunks_path": chunks_path,
            "keywords_path": keywords_path,
            "index_path": index_path
        }
        
        metadata_path = os.path.join(self.indices_folder, f"{base_filename}_metadata.json")
        with open(metadata_path, 'w') as f:
            json.dump(metadata, f)
        
        print(f"Saved processed data for {os.path.basename(pdf_path)}")
        return metadata_path
    
    def load_processed_data(self, metadata_path: str) -> Tuple[List[str], List[List[str]], Any, Dict[str, Any]]:
        """Load processed PDF data from disk.
        
        Args:
            metadata_path (str): Path to the metadata file
        
        Returns:
            Tuple[List[str], List[List[str]], Any, Dict[str, Any]]: Chunks, keywords, FAISS index, and metadata
        """
        # Load metadata
        with open(metadata_path, 'r') as f:
            metadata = json.load(f)
        
        # Load chunks
        with open(metadata["chunks_path"], 'rb') as f:
            chunks = pickle.load(f)
        
        # Load keywords
        with open(metadata["keywords_path"], 'r') as f:
            keywords = json.load(f)
        
        # Load FAISS index
        index = faiss.read_index(metadata["index_path"])
        
        return chunks, keywords, index, metadata
    
    def process_pdf(self, pdf_path: str) -> Optional[str]:
        """Process a single PDF file.
        
        Args:
            pdf_path (str): Path to the PDF file
        
        Returns:
            Optional[str]: Path to the metadata file or None if processing failed
        """
        # Check if already processed
        base_filename = os.path.splitext(os.path.basename(pdf_path))[0]
        metadata_path = os.path.join(self.indices_folder, f"{base_filename}_metadata.json")
        
        if os.path.exists(metadata_path):
            print(f"{os.path.basename(pdf_path)} already processed. Skipping.")
            return metadata_path
        
        # Extract text
        text = self.extract_text_from_pdf(pdf_path)
        if not text:
            return None
        
        # Split into chunks
        chunks = self.split_text(text)
        if not chunks:
            print(f"No chunks created for {os.path.basename(pdf_path)}. Skipping.")
            return None
        
        # Extract keywords
        keywords = self.extract_keywords(chunks)
        
        # Create embeddings and index
        index, _ = self.create_embeddings(chunks)
        
        # Save processed data
        metadata_path = self.save_processed_data(pdf_path, chunks, keywords, index)
        return metadata_path
    
    def process_all_pdfs(self) -> List[str]:
        """Process all PDFs in the folder.
        
        Returns:
            List[str]: List of paths to metadata files
        """
        # Get all PDF files in the folder
        pdf_pattern = os.path.join(self.pdf_folder, '*.pdf')
        pdf_files = glob.glob(pdf_pattern)
        
        if not pdf_files:
            print(f"No PDF files found in {self.pdf_folder}")
            return []
        
        print(f"Found {len(pdf_files)} PDF files:")
        for i, pdf in enumerate(pdf_files):
            print(f"{i+1}. {os.path.basename(pdf)}")
        
        # Initialize embedding model if not already done
        if self.embedding_model is None:
            self.embedding_model = SentenceTransformer('all-MiniLM-L6-v2', device=self.device)
        
        # Process each PDF
        metadata_paths = []
        for pdf_path in pdf_files:
            metadata_path = self.process_pdf(pdf_path)
            if metadata_path:
                metadata_paths.append(metadata_path)
        
        self.metadata_paths = metadata_paths
        return metadata_paths
    
    def load_llm(self, model_id: str = "TinyLlama/TinyLlama-1.1B-Chat-v1.0") -> Any:
        """Load a language model for generation.
        
        Args:
            model_id (str): Hugging Face model ID
        
        Returns:
            Any: Language model
        """
        print(f"Loading language model: {model_id}")
        
        try:
            # First try loading with 8-bit quantization
            try:
                from transformers import BitsAndBytesConfig
                
                # Configure quantization
                quantization_config = BitsAndBytesConfig(
                    load_in_8bit=True,
                    bnb_4bit_compute_dtype=torch.float16
                )
                
                # Load model with quantization config
                tokenizer = AutoTokenizer.from_pretrained(model_id)
                model = AutoModelForCausalLM.from_pretrained(
                    model_id, 
                    quantization_config=quantization_config,
                    device_map="auto"
                )
                print("Loaded model with 8-bit quantization")
            except Exception as e:
                print(f"8-bit quantization failed with error: {e}")
                print("Falling back to 16-bit precision...")
                
                # Fall back to fp16 without quantization
                tokenizer = AutoTokenizer.from_pretrained(model_id)
                model = AutoModelForCausalLM.from_pretrained(
                    model_id, 
                    torch_dtype=torch.float16,
                    device_map="auto"
                )
                print("Loaded model with fp16 precision")
            
            # Create a text generation pipeline
            pipe = pipeline(
                "text-generation",
                model=model,
                tokenizer=tokenizer,
                max_new_tokens=512,
                do_sample=True,
                temperature=0.7,
                top_p=0.9,
            )
            
            # Create a HuggingFacePipeline for use with LangChain
            llm = HuggingFacePipeline(pipeline=pipe)
            
            print(f"Successfully loaded {model_id} for text generation.")
            return llm
        
        except Exception as e:
            print(f"Error loading LLM: {e}")
            print("Using a simple function-based LLM implementation")
            
            # If we get here, use a simple function-based approach as last resort
            def simple_llm(prompt):
                # Extract question and context
                question_start = prompt.find("Question:")
                context_start = prompt.find("Context:")
                
                if question_start != -1 and context_start != -1:
                    context = prompt[context_start+8:question_start].strip()
                    question = prompt[question_start+9:].strip().split("\n")[0].strip()
                    
                    # Very simple logic to find relevant sentences
                    sentences = context.split(". ")
                    relevant_sentences = []
                    
                    for sentence in sentences:
                        lower_s = sentence.lower()
                        lower_q = question.lower()
                        # Find sentences containing keywords from the question
                        if any(word in lower_s for word in lower_q.split() if len(word) > 3):
                            relevant_sentences.append(sentence)
                    
                    # If we found relevant sentences, combine them
                    if relevant_sentences:
                        answer = "Based on the provided document, " + ". ".join(relevant_sentences) + "."
                    else:
                        answer = "I couldn't find specific information about that in the provided context."
                        
                    return answer
                return "I couldn't process that question properly."
            
            # Wrap the function to match the expected interface
            class SimpleLLM:
                def _call_(self, prompt):
                    return simple_llm(prompt)
            
            return SimpleLLM()
    
    def enhanced_search(self, question: str, chunks: List[str], keywords: List[List[str]], 
                      index: Any, top_k: int = 3) -> List[str]:
        """Search for relevant chunks using both embeddings and keywords.
        
        Args:
            question (str): The question to search for
            chunks (List[str]): List of text chunks
            keywords (List[List[str]]): List of keywords for each chunk
            index (Any): FAISS index
            top_k (int): Number of top chunks to return
            
        Returns:
            List[str]: List of relevant text chunks
        """
        # Check if the embedding model is initialized
        if self.embedding_model is None:
            self.embedding_model = SentenceTransformer('all-MiniLM-L6-v2', device=self.device)
        
        # 1. Embed the question
        question_embedding = self.embedding_model.encode([question])
        faiss.normalize_L2(question_embedding)
        
        # 2. Vector search
        scores, indices = index.search(question_embedding, top_k * 2)  # Get more candidates
        
        # 3. Extract keywords from the question
        if 'KeyBERT' in globals():
            keyword_model = KeyBERT()
            question_keywords = [kw for kw, _ in keyword_model.extract_keywords(
                question, keyphrase_ngram_range=(1, 2), top_n=3
            )]
        else:
            # Simple keyword extraction fallback
            words = re.findall(r'\b[a-zA-Z]{3,15}\b', question.lower())
            if 'stopwords' in globals():
                words = [w for w in words if w not in stopwords.words('english')]
            question_keywords = words[:3]
        
        # 4. Combine vector search with keyword matching
        candidates = []
        
        # Process vector search results
        for i, idx in enumerate(indices[0]):
            if idx >= len(keywords):
                continue  # Skip if index is out of bounds
                
            # Calculate keyword overlap
            chunk_kws = keywords[idx]
            overlap = len(set(question_keywords) & set(chunk_kws))
            
            # Combine vector similarity with keyword overlap
            # Adjust the weights as needed (0.7 for vector, 0.3 for keywords)
            combined_score = 0.7 * scores[0][i] + 0.3 * (overlap / max(1, len(question_keywords)))
            
            candidates.append((idx, combined_score, chunks[idx]))
        
        # Sort by combined score
        candidates.sort(key=lambda x: x[1], reverse=True)
        
        # Take top_k
        top_chunks = [chunk for _, _, chunk in candidates[:top_k]]
        
        return top_chunks
    
    def find_relevant_pdf(self, question: str) -> Tuple[str, float, str, str]:
        """Find the most relevant PDF for a given question.
        
        Args:
            question (str): The question to find a relevant PDF for
            
        Returns:
            Tuple[str, float, str, str]: Metadata path, similarity score, filename, and PDF path
        """
        # Make sure we have processed PDFs
        if not self.metadata_paths:
            self.process_all_pdfs()
        
        if not self.metadata_paths:
            raise ValueError("No processed PDFs available. Please add PDFs to the folder and process them.")
        
        # Check if the embedding model is initialized
        if self.embedding_model is None:
            self.embedding_model = SentenceTransformer('all-MiniLM-L6-v2', device=self.device)
        
        # Find the most relevant PDF
        pdf_relevance = []
        
        for metadata_path in self.metadata_paths:
            # Load metadata
            with open(metadata_path, 'r') as f:
                metadata = json.load(f)
            
            # Load sample of chunks (first few) to check relevance
            with open(metadata["chunks_path"], 'rb') as f:
                chunks_sample = pickle.load(f)[:5]  # Just load a few chunks for quick testing
            
            # Create sample embedding for the PDF
            sample_text = " ".join(chunks_sample)
            sample_embedding = self.embedding_model.encode([sample_text])
            
            # Embed the question
            question_embedding = self.embedding_model.encode([question])
            
            # Calculate similarity
            similarity = np.dot(sample_embedding, question_embedding.T)[0][0]
            
            # Store metadata path, similarity, filename, and original PDF path
            original_pdf_path = os.path.join(os.path.dirname(os.path.dirname(metadata["chunks_path"])), metadata["pdf_filename"])
            pdf_relevance.append((metadata_path, similarity, metadata["pdf_filename"], original_pdf_path))
        
        # Sort by relevance
        pdf_relevance.sort(key=lambda x: x[1], reverse=True)
        
        # Return the most relevant PDF
        return pdf_relevance[0]
    
    def query(self, question: str, max_history: int = 3) -> Tuple[str, str]:
        """Query the RAG system with a question.
        
        Args:
            question (str): The question to ask
            max_history (int): Maximum number of conversation history items to include
            
        Returns:
            Tuple[str, str]: Response and source document path
        """
        # Make sure we have processed PDFs
        if not self.metadata_paths:
            self.process_all_pdfs()
        
        # Make sure we have an LLM
        if self.llm is None:
            self.llm = self.load_llm()
        
        # Find the most relevant PDF
        metadata_path, similarity, filename, pdf_path = self.find_relevant_pdf(question)
        
        print(f"Most relevant document: {filename} (relevance: {similarity:.2f})")
        print(f"Document path: {pdf_path}")
        
        # Load PDF data
        chunks, keywords, index, metadata = self.load_processed_data(metadata_path)
        
        # Search using enhanced search
        relevant_chunks = self.enhanced_search(
            question, chunks, keywords, index
        )
        
        # Add source info to chunks
        source_chunks = [f"[From {metadata['pdf_filename']}] {chunk}" for chunk in relevant_chunks]
        
        # Combine all results
        context = "\n\n".join(source_chunks)
        
        # Format the conversation history if provided
        conversation_context = ""
        if self.conversation_history and len(self.conversation_history) > 0:
            conversation_context = "Previous conversation:\n"
            history_items = self.conversation_history[-max_history:]
            for i, (q, a) in enumerate(history_items):
                conversation_context += f"Question {i+1}: {q}\nAnswer {i+1}: {a}\n\n"
        
        # Create prompt for the LLM
        prompt = f"""
You are an AI assistant specializing in providing accurate information from PDF documents.
Answer the following question based on the provided context from the document.
Only use information from the context. If you don't know the answer, say so.

{conversation_context}
Document: {metadata['pdf_filename']} (at {pdf_path})

Context:
{context}

Question: {question}

Answer:
"""
        
        # Generate response
        response = self.llm(prompt)
        
        # Add to conversation history
        self.conversation_history.append((question, response))
        if len(self.conversation_history) > max_history:
            self.conversation_history = self.conversation_history[-max_history:]
        
        return response, pdf_path,context
    
    def add_pdf(self, pdf_path: str) -> Optional[str]:
        """Add a new PDF to the system.
        
        Args:
            pdf_path (str): Path to the PDF file
            
        Returns:
            Optional[str]: Path to the metadata file or None if processing failed
        """
        # Check if the file exists
        if not os.path.exists(pdf_path):
            print(f"Error: File {pdf_path} does not exist")
            return None
        
        # Copy the file to the PDF folder if it's not already there
        if not pdf_path.startswith(self.pdf_folder):
            import shutil
            dest_path = os.path.join(self.pdf_folder, os.path.basename(pdf_path))
            shutil.copy2(pdf_path, dest_path)
            pdf_path = dest_path
            print(f"Copied PDF to {dest_path}")
        
        # Process the PDF
        metadata_path = self.process_pdf(pdf_path)
        
        # Add to metadata paths if successful
        if metadata_path and metadata_path not in self.metadata_paths:
            self.metadata_paths.append(metadata_path)
        
        return metadata_path

    def clear_conversation_history(self):
        """Clear the conversation history."""
        self.conversation_history = []
        print("Conversation history cleared.")



def get_rag_response(query, pdf_folder='../backend/uploads', use_gpu=False, model_id=None):
    """Get answer and context from the RAG system for a given query."""
    global _rag_instance, _last_pdf_files

    # Get current list of PDFs
    current_pdf_files = sorted(glob.glob(os.path.join(pdf_folder, '*.pdf')))

    # Only re-initialize if PDFs changed
    if _rag_instance is None or current_pdf_files != _last_pdf_files:
        print("Initializing or refreshing RAG system...")
        _rag_instance = RAGSystem(pdf_folder=pdf_folder, use_gpu=use_gpu)
        if model_id:
            _rag_instance.llm = _rag_instance.load_llm(model_id=model_id)
        _rag_instance.process_all_pdfs()
        _last_pdf_files = current_pdf_files

    rag = _rag_instance

    try:
        response,_,context = rag.query(query)

        answer_match = re.search(r"Answer:\s*(.*)", response, re.DOTALL | re.IGNORECASE)
        answer = answer_match.group(1).strip() if answer_match else "Answer not found"

        return {
            "answer": answer,
            "context": context
        }

    except Exception as e:
        return {
            "error": f"Error processing query: {str(e)}"
        }

_rag_instance = None
_last_pdf_files = []

# Main function for running as a script
def main():
   
    rag = RAGSystem()
    
    # Process all PDFs
    rag.process_all_pdfs()
    
    # Query loop
    print("\nRAG system is ready! Type 'quit' to exit.")
    while True:
        query = input("\nEnter your question: ")
        if query.lower() == 'quit':
            break
        
        try:
            response, source_doc = rag.query(query)
            print("\nAnswer:")
            print(response)
            print(f"\nSource document: {source_doc}")
        except Exception as e:
            print(f"Error: {e}")

if __name__ == "__main__":
    main()
