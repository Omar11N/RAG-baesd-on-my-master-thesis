# Intelligent PDF Q&A System (RAG)

*An end-to-end system for extracting, processing, and querying PDF documents using state-of-the-art NLP and deep learning techniques.*
*I have built an api for this RAG SYS, I'll privide it upon request*

## Overview

This project demonstrates an intelligent, scalable pipeline that transforms raw PDF content into structured, semantically meaningful text chunks and leverages deep learning to generate detailed, context-aware answers to user queries. Built in Python, the system integrates multiple advanced libraries—including [PyMuPDF](https://pymupdf.readthedocs.io), [Sentence Transformers](https://www.sbert.net), [spaCy](https://spacy.io), and [Hugging Face Transformers](https://huggingface.co)—to deliver high-quality insights directly from PDF documents.

The workflow consists of:

- **PDF Parsing & Preprocessing:** Efficiently extracts text from PDFs, cleans it, and splits it into sentences using spaCy.
- **Text Chunking & Embedding:** Organizes text into coherent chunks and generates dense vector embeddings with a powerful transformer model.
- **Query Retrieval & Response Generation:** Retrieves the most relevant text passages for a given query and generates a well-formulated answer using a cutting-edge language model.

## Key Features

- **Robust PDF Processing:** Reads and cleans PDF text with precision, making it ideal for academic papers, reports, and technical documents.
- **Semantic Text Chunking:** Groups sentences into meaningful chunks, ensuring each piece has sufficient context.
- **State-of-the-Art Embeddings:** Utilizes the "all-mpnet-base-v2" model for creating rich, high-dimensional representations of text.
- **Dynamic Query Answering:** Combines retrieval with prompt engineering to generate detailed, human-like answers.
- **Scalable & Efficient:** Employs batching and GPU-acceleration (if available) for fast processing, suitable for large document collections.

## Technical Highlights

- **Deep Learning Integration:** Leverages PyTorch for tensor operations and model inference on either GPU or CPU.
- **Advanced NLP Pipeline:** Uses spaCy’s sentencizer to accurately segment text and the Sentence Transformer for embedding generation.
- **End-to-End Automation:** Automatically saves and reloads embeddings to avoid redundant computation.
- **Custom Prompt Engineering:** Formats context-rich prompts for a causal language model, enabling nuanced and contextually aware responses.

