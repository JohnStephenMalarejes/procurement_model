import torch
import chromadb
from sentence_transformers import SentenceTransformer
from transformers import AutoTokenizer, AutoModelForCausalLM, BitsAndBytesConfig
from peft import PeftModel

from app.config import (
    BASE_MODEL_NAME,
    ADAPTER_DIR,
    CHROMA_DB_DIR,
    COLLECTION_NAME,
    EMBEDDING_MODEL_NAME,
    SYSTEM_PROMPT_TEMPLATE,
)


class ProcurementRAGEngine:
    def __init__(self):
        print("1. Initializing Embedding Model & ChromaDB...")
        self.embedder = SentenceTransformer(EMBEDDING_MODEL_NAME)
        self.chroma_client = chromadb.PersistentClient(path=CHROMA_DB_DIR)
        self.collection = self.chroma_client.get_or_create_collection(
            name=COLLECTION_NAME
        )

        print(f"2. Loading Tokenizer from {ADAPTER_DIR}...")
        self.tokenizer = AutoTokenizer.from_pretrained(ADAPTER_DIR)
        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token

        print(f"3. Loading Base Model ({BASE_MODEL_NAME}) & LoRA Adapter...")
        if torch.cuda.is_available():
            quant_config = BitsAndBytesConfig(
                load_in_4bit=True,
                bnb_4bit_quant_type="nf4",
                bnb_4bit_use_double_quant=True,
                bnb_4bit_compute_dtype=torch.float16,
            )
            base_model = AutoModelForCausalLM.from_pretrained(
                BASE_MODEL_NAME,
                quantization_config=quant_config,
                device_map="auto",
                torch_dtype=torch.float16,
            )
        else:
            # Fallback if running on a server without a dedicated NVIDIA GPU
            print("CUDA GPU not detected. Loading model in float32 on CPU...")
            base_model = AutoModelForCausalLM.from_pretrained(
                BASE_MODEL_NAME,
                device_map="cpu",
                torch_dtype=torch.float32,
            )

        self.model = PeftModel.from_pretrained(base_model, ADAPTER_DIR)
        self.model.eval()
        self.model.config.use_cache = True
        print("Fine-tuned procurement RAG engine successfully loaded.")

    def add_documents(self, documents: list[str], ids: list[str], metadatas: list[dict] = None):
        """Encodes and stores documents in ChromaDB."""
        embeddings = self.embedder.encode(documents).tolist()
        self.collection.upsert(
            documents=documents,
            embeddings=embeddings,
            ids=ids,
            metadatas=metadatas,
        )

    def retrieve_context(self, query: str, top_k: int = 3) -> list[str]:
        """Retrieves the top_k most relevant policy chunks for a user query."""
        if self.collection.count() == 0:
            return []

        query_embedding = self.embedder.encode(query).tolist()
        results = self.collection.query(
            query_embeddings=[query_embedding],
            n_results=min(top_k, self.collection.count()),
        )
        return results.get("documents", [[]])[0]

    def ask(self, question: str, top_k: int = 3, max_new_tokens: int = 250) -> dict:
        """Runs full RAG pipeline: Retrieve -> Augment Prompt -> Generate."""
        retrieved_chunks = self.retrieve_context(question, top_k=top_k)

        if retrieved_chunks:
            context_block = "\n\n".join(f"- {chunk}" for chunk in retrieved_chunks)
        else:
            context_block = "No specific organization policy documents found."

        system_prompt = SYSTEM_PROMPT_TEMPLATE.format(context=context_block)

        messages = [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": question},
        ]

        prompt = self.tokenizer.apply_chat_template(
            messages,
            tokenize=False,
            add_generation_prompt=True,
            enable_thinking=False,
        )

        inputs = self.tokenizer(prompt, return_tensors="pt").to(self.model.device)

        with torch.no_grad():
            outputs = self.model.generate(
                **inputs,
                max_new_tokens=max_new_tokens,
                do_sample=False,
                repetition_penalty=1.05,
                pad_token_id=self.tokenizer.pad_token_id,
                eos_token_id=self.tokenizer.eos_token_id,
            )

        generated_tokens = outputs[0][inputs["input_ids"].shape[1]:]
        answer = self.tokenizer.decode(generated_tokens, skip_special_tokens=True).strip()

        return {
            "question": question,
            "answer": answer,
            "retrieved_context": retrieved_chunks,
        }