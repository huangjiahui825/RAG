
# Dynamic Multi-Modal Retrieved Augmented Generation (RAG) Tracking System：Based on Model Context Protocol (MCP)

[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](LICENSE)

## Project Overview

This project presents a dynamic **Retrieved Augmented Generation (RAG)** system based on the **Model Context Protocol (MCP)** framework. 
It integrates local PDF and Excel documents, external PDF sources, and real-time web search data to enable multi-source knowledge retrieval, intelligent document analysis, and automatic evaluation.

### Project Objectives
- Develop a flexible **Model Context Protocol (MCP)** for dynamic prompt construction and multi-turn dialogue management.
- Benchmark and integrate mainstream vector databases for efficient semantic retrieval.
- Enable **multi-modal retrieval capabilities** across text, tables, and images.
- Establish a multi-dimensional evaluation framework for rigorous answer quality assessment.
- Implement a dynamic knowledge update mechanism with integrated **Web Search** for real-time context enrichment.

### Application Scenarios
This system is designed to power intelligent AI agents for:
- Enterprises: Real-time knowledge management and decision support.
- Government Agencies: Regulatory document tracking and compliance monitoring.
- Academic Institutions: Research repositories with continuous knowledge updates.

---

## Key Features
- Multi-source data fusion: Integrates local PDFs, Excel files, external PDFs, and web data.
- Automated document parsing and semantic vectorization via MinerU and Sentence Transformers.
- Vector storage and retrieval using Qdrant vector database.
- Automatic file monitoring for seamless synchronization between files and vector storage.
- Intelligent multi-turn dialogue with dynamic topic switching and history summarization.
- Integrated LLM-based Q&A with auto-evaluation on answer accuracy, latency, and robustness.
- One-click generation of comprehensive evaluation reports in PDF format.

---

### External Tools
- MinerU PDF Parser (included in `Coding.zip`)
- Qdrant Vector Database (local instance required)
- Brave Web Search API (for real-time web search)
- OpenAI API Key or Nuwa API Proxy for LLM interactions

---

## Quick Start
1. Prepare PDF and Excel datasets and place them in the designated directories.
2. Start the Qdrant vector database service.
3. Configure file paths and API keys in `RAG.py`.
4. Run the main script and follow the interactive instructions.

---

## Configuration Details
- PDF/Excel directory paths
- MinerU batch script path
- API keys and endpoints
- Qdrant database URL
- Processed file log path (`processed_files.json`)

---

## Usage Guide
1. Choose whether to enable Web Search (type `yes` or `no`).
2. Enter query text for web search; type `stop` to end.
3. Enter PDF URL to download and process (or type `skip`).
4. Enter local queries for document retrieval and answer generation; type `stop` to end.
5. The system will auto-evaluate answers and generate an `Answer.pdf` report.

---

## Files & Modules Description
(Omitted here for brevity, included in final README generation.)

---

## Evaluation & Testing
This system features automatic LLM-based evaluation:
- Answer Quality: relevance, faithfulness, completeness, clarity, conciseness.
- Performance: response latency, evaluation latency.
- Robustness: automated fake-answer testing.

---

## License
This project is licensed under the **MIT License**.

---

## Acknowledgements
This project is inspired by MinerU, Qdrant, and various open-source tools in the AI community.
