# EquityAI RAG Production Runbook

## 1. Purpose

The EquityAI Retrieval-Augmented Generation (RAG) subsystem is the
production knowledge and document-retrieval layer for financial research.

It provides grounded retrieval from uploaded financial documents and is
designed to serve both the EquityAI API and downstream agentic AI services.

## 2. Production Architecture

Document ingestion follows an asynchronous durable pipeline:

Client
→ FastAPI
→ document validation and registration
→ Amazon S3
→ Amazon SQS
→ ECS ingestion worker
→ PDF parsing and chunking
→ OpenAI embeddings
→ PostgreSQL / pgvector

Question answering follows:

User question
→ retrieval-scope resolution
→ query embedding
→ hybrid retrieval
→ reranking
→ balanced evidence selection
→ grounded generation
→ answer with document/page/chunk sources

## 3. Asynchronous Ingestion

The API does not perform embedding synchronously.

Upload flow:

1. Validate uploaded document.
2. Calculate SHA-256 hash.
3. Register document in PostgreSQL.
4. Upload source PDF to S3.
5. Send ingestion message to SQS.
6. Return HTTP 202.
7. ECS worker claims the document atomically.
8. Worker downloads the PDF from S3.
9. PDF is parsed and chunked.
10. Chunks are embedded.
11. Embeddings and metadata are stored in PostgreSQL/pgvector.
12. Document status becomes completed.
13. Successful SQS message is deleted.

Processing and dispatch leases protect against duplicate or abandoned work.

A reconciliation process redispatches queued documents that were uploaded
to S3 but were not successfully dispatched to SQS.

## 4. Retrieval Pipeline

Production retrieval uses hybrid semantic and keyword search.

The RAG service retrieves:

RERANK_CANDIDATE_LIMIT = 20

These candidates are passed to the reranker.

The final evidence set remains:

limit = 3

This separates candidate recall from evidence precision.

## 5. Financial Evidence Reranking

Financial questions receive additional evidence-aware reranking.

The reranker reuses the canonical financial metric vocabulary from the
financial analysis subsystem.

For questions containing:

- a recognised financial metric; and
- explicit fiscal/calendar years,

a candidate receives an evidence-completeness signal when it contains:

- the requested financial metric or recognised alias; and
- all explicitly requested years.

Metric aliases use boundary-safe matching.

This allows direct financial evidence to outrank generic annual-report
content with superficially higher semantic similarity.

## 6. Comparative Report Retrieval

When a comparison requests multiple years:

- use the requested year's filing when available;
- if an older filing is unavailable, comparative information contained
  in a newer annual filing may be used;
- never fabricate unavailable evidence.

Repeated year mentions are deduplicated while preserving first-occurrence
order.

## 7. Grounding and Abstention

The system must not manufacture financial evidence.

If retrieved evidence is insufficient, the API returns:

"I could not find sufficient evidence in the provided documents."

Answers expose source metadata including:

- document
- page
- chunk
- reranking score

## 8. Production Acceptance Baseline

Production acceptance was completed using:

HSBC Holdings plc Annual Report and Accounts 2025.

Acceptance question:

"According to HSBC Holdings plc Annual Report and Accounts 2025,
what was HSBC's reported profit before tax in 2025, and how did it
compare with 2024?"

Production result:

- HTTP 200
- 2025 profit before tax: $29,907 million
- 2024 profit before tax: $32,309 million
- decrease: $2,402 million
- approximately 7% decline
- primary evidence: page 16, chunk 1

The relevant evidence was previously hybrid rank 18 and was excluded by
the old five-candidate retrieval pool.

After widening candidate retrieval and introducing financial
evidence-aware reranking, page 16 became the highest-ranked evidence.

## 9. Production AWS Components

Primary production services:

- Amazon ECS / Fargate — API and ingestion worker
- Amazon S3 — durable source-document storage
- Amazon SQS — ingestion queue and retry/DLQ mechanism
- PostgreSQL / pgvector — metadata, chunks and vector storage
- AWS Secrets Manager — production secrets
- CloudWatch — application/container logging
- Application Load Balancer — API ingress

## 10. Testing Baseline

RAG closure baseline:

383 tests passed.

The permanent regression suite covers:

- document ingestion
- S3 document storage
- SQS dispatch
- worker processing
- processing leases
- dispatch leases
- orphan reconciliation
- retrieval scope
- report selection
- hybrid retrieval
- reranking
- financial evidence ranking
- balanced evidence selection
- RAG observability
- API behaviour

## 11. Operational Rules

Do not lower grounding or abstention controls to force an answer.

Do not increase retrieval or reranking parameters based on a single
issuer without a regression test and production evidence.

Do not perform expensive embedding work in the API request lifecycle.

Do not delete SQS messages before successful or terminal-safe processing.

Do not duplicate the canonical financial metric vocabulary inside the
reranker.

All material retrieval changes require:

1. regression test;
2. full test-suite pass;
3. deployment;
4. production acceptance test.

## 12. Agentic AI Integration Contract

The RAG subsystem is the Financial Analysis Agent's document knowledge
layer.

Future agents should consume the RAG service rather than implement their
own document ingestion, embeddings, vector database or financial filing
retrieval stack.

The planned agentic research architecture is:

Research Orchestrator
→ Financial Analysis Agent
→ Financial Market Data Agent
→ Macroeconomic & Political Environment Agent
→ Market & Competitor Intelligence Agent
→ Synthesis / Risk
→ Validation & Guardrails
→ Investment Research Output

The Financial Analysis Agent should use this RAG subsystem for grounded
company-document evidence and PostgreSQL/SQL for structured financial
analysis.

The RAG subsystem should therefore be treated as a stable production
data plane while agent orchestration is developed independently.
