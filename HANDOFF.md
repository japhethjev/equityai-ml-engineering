# EquityAI ML Engineering — Development Handoff

**Repository:** `equityai-ml-engineering`
**Branch:** `main`
**Status:** Stable
**Last validated:** 24 September 2026

## 1. Project Purpose

EquityAI is being developed as a production-oriented financial research and investment intelligence platform.

The current backend implements a Retrieval-Augmented Generation (RAG) pipeline that answers questions from uploaded financial PDF documents while enforcing evidence grounding, financial metric distinctions, temporal accuracy, citations, abstention behaviour, evaluation, regression protection and operational observability.

The immediate engineering objective has been to build the RAG foundation using practical ML/LLMOps engineering principles rather than simply creating an LLM chatbot.

---

## 2. Current RAG Architecture

The current request flow is:

```text
User Question
      │
      ▼
FastAPI /ask
      │
      ├── Request ID
      │
      ▼
Embedding Service
      │
      ▼
Hybrid Retrieval
      │
      ▼
PostgreSQL + pgvector
      │
      ▼
Reranker
      │
      ▼
Top Evidence Chunks
      │
      ▼
Evidence Sufficiency Controls
      │
      ├── Direct evidence
      ├── Temporal matching
      ├── Financial metric matching
      └── Calculation permission
      │
      ▼
OpenAI Generation
      │
      ├── ANSWERED
      ├── ABSTAINED
      └── ERROR
      │
      ▼
Answer + Sources + Request ID
```

---

## 3. Current Core Components

### API

Main API:

```text
app/api/main.py
```

Current endpoints include:

```text
GET    /health
POST   /ask
POST   /documents/upload
GET    /documents
DELETE /documents/{document_name}
```

The API includes HTTP-level request observability and correlation IDs.

### RAG Service

Main orchestration:

```text
app/rag/rag_service.py
```

Responsibilities include:

* query validation;
* calculation-intent detection;
* query embedding;
* hybrid retrieval;
* reranking;
* evidence context construction;
* evidence-constrained generation;
* abstention;
* source metadata;
* request ID propagation;
* stage-level timing;
* structured success/error telemetry.

### Database

The application uses:

```text
PostgreSQL
pgvector
```

PostgreSQL runs through Docker Compose.

The vector database stores document chunks and their embeddings.

### Document Ingestion

Main ingestion module:

```text
app/rag/ingest.py
```

Current ingestion pipeline:

```text
PDF
 ↓
load_pdf()
 ↓
chunk_pages()
 ↓
duplicate detection
 ↓
embed new chunks only
 ↓
store_chunks()
 ↓
PostgreSQL / pgvector
```

Duplicate chunks are detected before embedding to avoid unnecessary embedding API calls.

---

## 4. Financial Evidence Controls

The RAG system is intentionally conservative about financial facts.

It distinguishes between related but non-equivalent concepts including:

```text
IPO offer price ≠ current market price

Revenue ≠ profit

Gross profit ≠ operating profit

Operating profit ≠ profit after tax

Profit ≠ profit margin

EPS ≠ dividend per share

Dividend per share ≠ dividend yield

Annual figures ≠ interim figures

Forecast figures ≠ actual figures
```

Temporal qualifiers such as:

```text
2024
2025
2026
H1
H2
current
latest
interim
full year
```

are treated as material components of a question.

---

## 5. Calculation Permission Control

A deterministic calculation-intent control has been added to `rag_service.py`.

The system distinguishes between a factual retrieval request and an explicit calculation request.

Example:

```text
What was the refinery's 2025 operating profit margin?
```

If the margin is not explicitly reported in the evidence:

```text
→ ABSTAIN
```

The system must not silently derive the margin from operating profit and revenue.

However:

```text
Calculate the refinery's 2025 operating profit margin.
```

provides explicit calculation permission.

If the necessary inputs are supported by retrieved evidence:

```text
→ calculation permitted
```

This distinction is important because EquityAI must not present a model-derived financial metric as though it were a directly reported company figure.

A defensive post-generation guard also detects explicit unrequested calculation behaviour and converts it to abstention.

---

## 6. Abstention Behaviour

Canonical abstention response:

```text
I could not find sufficient evidence in the provided documents.
```

An abstention is considered a successful RAG outcome rather than an application failure.

Operational states are therefore:

```text
ANSWERED
    ↓
HTTP 200
Grounded answer returned

ABSTAINED
    ↓
HTTP 200
Evidence insufficient

ERROR
    ↓
HTTP 503
Technical/application failure
```

---

## 7. Observability

Observability utilities are located in:

```text
app/observability.py
```

The RAG pipeline currently records:

* request ID;
* outcome;
* retrieved chunk count;
* used chunk count;
* embedding latency;
* retrieval latency;
* reranking latency;
* generation latency;
* total latency;
* failed pipeline stage;
* exception type for failures.

Typical successful log:

```text
event=rag_request
request_id=<id>
outcome=ANSWERED
retrieved_chunks=5
used_chunks=3
embedding_ms=<value>
retrieval_ms=<value>
reranking_ms=<value>
generation_ms=<value>
total_ms=<value>
```

Abstentions are logged as:

```text
outcome=ABSTAINED
```

Technical failures are logged as:

```text
outcome=ERROR
failed_stage=<stage>
error_type=<exception>
```

Error instrumentation has been tested across:

```text
embedding
retrieval
reranking
generation
```

---

## 8. End-to-End Correlation IDs

The API middleware generates a request ID for each HTTP request.

Example:

```text
X-Request-ID: 06a2a1da5c7f
```

The same ID is propagated through:

```text
HTTP request
      ↓
FastAPI
      ↓
RAG service
      ↓
RAG telemetry
      ↓
JSON response
```

The response contains:

```json
{
  "request_id": "06a2a1da5c7f"
}
```

This allows a production request to be traced through application and RAG logs.

---

## 9. Automated Testing

Latest complete test run:

```bash
pytest -v
```

Result:

```text
29 passed
```

The test suite covers areas including:

* API health;
* empty questions;
* valid questions;
* document listing;
* document deletion;
* missing-document handling;
* PDF loading;
* duplicate ingestion;
* financial number equivalence;
* financial rounding tolerance;
* materially incorrect financial values;
* percentage matching;
* abstention detection;
* financial value extraction;
* regression-gate behaviour;
* RAG failure handling;
* HTTP 503 behaviour;
* text chunking.

All tests were passing at handoff.

---

## 10. RAG Evaluation Framework

Evaluation command:

```bash
python -m evaluation.evaluate_rag
```

The current benchmark contains:

```text
30 RAG evaluation cases
```

It tests categories including:

* direct factual questions;
* paraphrases;
* temporal questions;
* metric distinctions;
* abstention;
* semantic traps.

Latest validated result:

```text
Cases completed: 30/30
Evaluation failures: 0
Overall pass rate: 100.0%

Answer accuracy: 100.0%
Retrieval hit rate: 100.0%
Top-1 retrieval accuracy: 100.0%
Citation accuracy: 100.0%
Abstention accuracy: 100.0%
```

---

## 11. Regression Testing

Baseline:

```text
evaluation/baseline.json
```

Current evaluation outputs:

```text
evaluation/results/latest.json
evaluation/results/latest.csv
```

Regression comparison:

```bash
python -m evaluation.compare_regression
```

Latest validated result:

```text
BENCHMARK STATUS: VALID

overall_pass_rate:
Baseline: 100.0%
Current:  100.0%

answer_accuracy:
Baseline: 100.0%
Current:  100.0%

retrieval_hit_rate:
Baseline: 100.0%
Current:  100.0%

top1_retrieval_accuracy:
Baseline: 100.0%
Current:  100.0%

citation_accuracy:
Baseline: 100.0%
Current:  100.0%

abstention_accuracy:
Baseline: 100.0%
Current:  100.0%

Case regressions: 0
Category regressions: 0

MODEL REGRESSION: NO
REGRESSION STATUS: PASSED
```

**Do not update `baseline.json` merely because a future evaluation performs worse.**

A failing regression test should first be investigated as a possible deterioration in RAG behaviour.

---

## 12. Important Regression Incident Resolved

The evaluation framework previously detected one real behavioural regression:

```text
Case: dangote_028

Question:
What was the refinery's 2025 operating profit margin?
```

The underlying document contained operating profit and revenue but did not directly report the requested operating profit margin.

The model incorrectly calculated approximately:

```text
1.19%
```

instead of abstaining.

This caused:

```text
Overall pass rate: 96.7%
Abstention accuracy: 87.5%

MODEL REGRESSION: YES
REGRESSION STATUS: FAILED
```

The problem was fixed by adding deterministic calculation-permission logic and a defensive post-generation calculation guard.

After the fix:

```text
30/30 evaluation cases passed

Overall pass rate: 100%
Abstention accuracy: 100%

MODEL REGRESSION: NO
REGRESSION STATUS: PASSED
```

This incident demonstrates why the regression gate must remain part of the development workflow.

---

## 13. Docker

The application runs using Docker Compose.

Typical startup:

```bash
docker compose up -d
```

Check services:

```bash
docker compose ps
```

Expected services:

```text
equityai-api
equityai-postgres
```

PostgreSQL should report healthy.

After application-code changes, rebuild the API when required:

```bash
docker compose build api
docker compose up -d
```

Logs:

```bash
docker logs equityai-api --tail 50
```

---

## 14. Production Validation Already Completed

A live Docker request was tested:

```bash
curl -i -X POST \
  http://localhost:8000/ask \
  -H "Content-Type: application/json" \
  -d '{"question":"What is the IPO offer price?"}'
```

Successful behaviour:

```text
HTTP/1.1 200 OK
```

Answer:

```text
The IPO offer price is ₦525.00.
```

The response included sources and a request ID.

A semantic-trap request was also tested:

```text
What is the current market price of Dangote Refinery shares?
```

The system correctly returned:

```text
I could not find sufficient evidence in the provided documents.
```

with:

```text
HTTP 200
outcome=ABSTAINED
```

rather than incorrectly returning the IPO offer price.

---

## 15. CI

GitHub Actions CI exists at:

```text
.github/workflows/ci.yml
```

The workflow runs Python tests for pushes and pull requests against `main`.

Any future production change should preserve automated test execution before merging.

---

## 16. Current Quality Gate

Before considering a significant RAG change stable, run:

```bash
pytest -v
```

Then:

```bash
python -m evaluation.evaluate_rag
```

Then:

```bash
python -m evaluation.compare_regression
```

Expected release condition:

```text
Software tests
    PASS
      ↓
30-case RAG evaluation
    PASS
      ↓
Regression comparison
    PASS
      ↓
Candidate change is safe to proceed
```

Do not rely only on `pytest`.

Traditional tests verify software behaviour; the RAG benchmark verifies AI behaviour.

---

## 17. Recommended Next Development Phase

The next phase is:

# Aggregate Production Observability

Individual request telemetry is already working.

The next objective is to turn those individual events into operational metrics.

Recommended metrics:

```text
Request count

ANSWERED rate

ABSTAINED rate

ERROR rate

P50 total latency

P95 total latency

P99 total latency

Embedding P50/P95 latency

Retrieval P50/P95 latency

Reranking P50/P95 latency

Generation P50/P95 latency
```

This should make it possible to answer operational questions such as:

```text
How many RAG requests are being processed?

What percentage are successfully answered?

How often does EquityAI abstain?

What is the production error rate?

How long does a typical request take?

How slow are the worst 5% of requests?

Which RAG stage is responsible for most latency?

Has latency deteriorated after a deployment?
```

The existing logs already capture much of the raw information required.

---

## 18. Recommended Future Roadmap

After aggregate observability, suggested engineering sequence:

```text
1. Aggregate metrics
2. Metrics endpoint / monitoring integration
3. Production dashboard
4. Alerting
5. Structured JSON logging
6. Retrieval-quality monitoring
7. LLM token and API-cost monitoring
8. Larger RAG evaluation dataset
9. CI regression gate
10. Authentication / authorization
11. Rate limiting
12. Multi-user / multi-tenant document isolation
13. Production deployment
```

Do not implement all of these simultaneously.

Continue incrementally and run the complete quality gate after material RAG changes.

---

## 19. Engineering Principle

The current EquityAI backend should preserve the following principle:

```text
Retrieve evidence
      ↓
Verify evidence sufficiency
      ↓
Answer only what the evidence supports
      ↓
Cite the evidence
      ↓
Abstain when evidence is insufficient
      ↓
Observe production behaviour
      ↓
Evaluate continuously
      ↓
Block regressions
```

For a financial AI system, a confident unsupported answer is generally more dangerous than an appropriate abstention.

---

## 20. Handoff State

At handoff:

```text
Automated tests:          29 PASS
RAG benchmark:            30/30 PASS
Overall RAG pass rate:    100%
Answer accuracy:          100%
Retrieval hit rate:       100%
Top-1 retrieval accuracy: 100%
Citation accuracy:        100%
Abstention accuracy:      100%
Case regressions:         0
Category regressions:     0
Regression status:        PASSED
Docker API:               VALIDATED
Request correlation:      VALIDATED
ANSWERED telemetry:       VALIDATED
ABSTAINED telemetry:      VALIDATED
ERROR handling/tests:     VALIDATED
```

**Resume development from aggregate production observability.**
