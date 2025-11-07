# CA AI Buddy Backend  
### *Your Expert Accounting & CA Assistant Powered by Groq + MongoDB + FastAPI*

[![FastAPI](https://img.shields.io/badge/FastAPI-005571?style=for-the-badge&logo=fastapi)](https://fastapi.tiangolo.com/)
[![Groq](https://img.shields.io/badge/Groq-LLM%20Inference-00D4AA?style=for-the-badge&logo=data:image/svg+xml;base64,PHN2ZyB4bWxucz0iaHR0cDovL3d3dy53My5vcmcvMjAwMC9zdmciIHdpZHRoPSIyNCIgaGVpZ2h0PSIyNCIgdmlld0JveD0iMCAwIDI0IDI0IiBmaWsbD0ibm9uZSIgc3Ryb2tlPSIjMDBENEFBIiBzdHJva2Utd2lkdGg9IjIiIHN0cm9rZS1saW5lY2FwPSJyb3VuZCIgc3Ryb2tlLWxpbmVqb2luPSJyb3VuZCI+PHBhdGggZD0iTTEyIDJMMiAxMm0xMCAwVjEybC0xMC0xMCIvPjwvc3ZnPg==)](https://groq.com)
[![MongoDB](https://img.shields.io/badge/MongoDB-4EA94B?style=for-the-badge&logo=mongodb&logoColor=white)](https://www.mongodb.com/)
[![Python](https://img.shields.io/badge/Python-3.10%2B-3776AB?style=for-the-badge&logo=python&logoColor=white)](https://python.org)

---

## Overview

**CA AI Buddy** is a **high-performance, production-ready backend** for an AI-powered assistant specialized in **Chartered Accountancy (CA), Accounting, Taxation, and Tally ERP**.

Built with **FastAPI**, powered by **Groq's ultra-fast LLM inference**, and backed by **MongoDB Atlas**, this service delivers **smart, context-aware, and concise responses** — perfect for professionals, students, and firms.

---

## Key Features

| Feature | Description |
|-------|-----------|
| **Smart Context Management** | Auto-summarizes long conversations to stay under token limits |
| **Session-Based Chats** | Full chat history per user/session stored in MongoDB |
| **Dynamic Truncation + Summarization** | Keeps only the most relevant context |
| **Production-Ready** | Logging, health checks, CORS, error handling |
| **Secure & Scalable** | Uses `certifi`, TLS, environment variables |
| **Fast Inference** | Powered by **Groq** (`gpt-oss-120b`) |
| **Clean API** | RESTful endpoints with Pydantic validation |

---

## Tech Stack

```bash
FastAPI + Uvicorn
Groq (via LangChain)
MongoDB Atlas + PyMongo
LangChain (Runnables, History, Prompts)
Pydantic v2
certifi + TLS
dotenv for secrets
