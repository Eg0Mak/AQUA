import os
import csv
from PyPDF2 import PdfReader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.schema import Document
from huggingface_hub import hf_hub_download
import onnxruntime as ort
from transformers import AutoTokenizer
import numpy as np

path_to_pdf = "RAG/data/pdf"
path_to_csv = "RAG/data/csv"

def read_pdf(pdf_folder):
    docs = []
    for file in os.listdir(pdf_folder):
        if file.endswith(".pdf"):
            path = os.path.join(pdf_folder, file)
            reader = PdfReader(path)
            for i, page in enumerate(reader.pages):
                text = page.extract_text()
                if text:
                    docs.append(
                        Document(
                            page_content=text,
                            metadata={"source": file, "page": i + 1}
                        )
                    )
    return docs


def read_csv(csv_folder):
    docs = []
    for file in os.listdir(csv_folder):
        if file.endswith(".csv"):
            path = os.path.join(csv_folder, file)
            with open(path, "r", encoding="utf-8", errors="ignore") as f:
                reader = csv.reader(f)
                header = next(reader, None)  # первая строка (заголовки)
                for i, row in enumerate(reader):
                    # собираем строку вида "col1: val1, col2: val2, ..."
                    if header:
                        text = ", ".join([f"{col}: {val}" for col, val in zip(header, row)])
                    else:
                        text = ", ".join(row)
                    docs.append(
                        Document(
                            page_content=text,
                            metadata={"source": file, "row": i + 1}
                        )
                    )
    return docs


# Load all documents
all_docs = read_pdf(path_to_pdf) + read_csv(path_to_csv)
print(f"Загружено документов: {len(all_docs)}")

# Split on chunks with saving the metadata
splitter = RecursiveCharacterTextSplitter(chunk_size=500, chunk_overlap=50)
chunked_docs = splitter.split_documents(all_docs)

# print(f"After splitting: {len(chunked_docs)} чанков")
# print("Example:")
# print(chunked_docs[1].page_content[:300])
# print("Metadata:", chunked_docs[1].metadata)


# Model Loading
# model_id = "onnx-community/embeddinggemma-300m-ONNX"
# model_path = hf_hub_download(model_id, subfolder="onnx", filename="model.onnx")
# hf_hub_download(model_id, subfolder="onnx", filename="model.onnx_data")
# session = ort.InferenceSession(model_path)
# tokenizer = AutoTokenizer.from_pretrained(model_id)

model_id = "sentence-transformers/all-MiniLM-L6-v2-onnx"
model_path = hf_hub_download(model_id, filename="model.onnx")
session = ort.InferenceSession(model_path)
tokenizer = AutoTokenizer.from_pretrained(model_id)

def get_embeddings(documents, tokenizer, session):
    embeddings = []
    for doc in documents:
        # Text tokenization
        inputs = tokenizer(doc.page_content, return_tensors="np", padding=True, truncation=True)
        input_ids = inputs["input_ids"]
        attention_mask = inputs["attention_mask"]
        

        ort_inputs = {
            "input_ids": input_ids,
            "attention_mask": attention_mask
        }
        outputs = session.run(None, ort_inputs)
        embedding = outputs[0].mean(axis=1)
        embeddings.append({
            "embedding": embedding[0],
            "metadata": doc.metadata,
            "content": doc.page_content
        })
    return embeddings

# Получение эмбеддингов
embeddings = get_embeddings(chunked_docs, tokenizer, session)
print(f"Создано {len(embeddings)} эмбеддингов")
