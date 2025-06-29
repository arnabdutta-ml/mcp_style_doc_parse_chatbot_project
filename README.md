# MCP Style Doc Parse Chatbot

A CLI tool to chat locally with PDF and DOCX files using sentence-transformers for embeddings, FAISS for retrieval, and Ollama for local LLaMA models. Answers questions with MCP-style JSON structure.

## Features
- Local embeddings (no API needed)
- FAISS-based retrieval
- Ollama-powered local LLM
- MCP-style JSON answers for document queries

## Installation

Clone and install requirements.

## Usage
ollama pull llama2
Run mcp-style-doc-parse-chatbot

## Example

Your input: doc: What are the payment terms?

Answer: Payment is due within 30 days.

Citations:
- Payment must be made within 30 days after invoice.
- Late fees apply for delayed payment.

Your input: chat: Tell me a joke.

Assistant: Why did the AI cross the road? To optimize the chicken!