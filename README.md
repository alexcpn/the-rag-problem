# The RAG Problem
## Why RAG looks Easy, but Turns out much Harder

There are literally thousands of papers of RAG and a lot of different methodologies. But those who put it in production  find that a new edge case quietly proves that it is not an easy problem to solve

**The hard problem is not retrieval. It is intent.**

Let me illustrate with an example.

**The Milkshake Problem**

Imagine your vector DB contains the embedding for *"My kids like milkshake."*

A user asks: *"What should I buy from the supermarket? Add things to the shopping list."*

Will a basic RAG pipeline surface milkshake? No. More importantly, **no retriever will solve this on its own.** Not BM25, because there is zero token overlap. Not dense embeddings, because the query is about "shopping" while the document is about "kids' preferences." Not even HyDE: the LLM will generate a generic shopping-list-style hypothetical document, but it still has no access to the private fact that *your* kids like milkshake.

The missing piece is not retrieval. It is the **chain of inference**:

```
shopping list → things the household consumes → things MY kids like → milkshake
```

To surface milkshake, the system has to **inject that premise into the query before retrieval runs**. That can happen through a user-profile lookup, multi-hop agentic retrieval, or a knowledge graph that models `user —hasChild→ kid —likes→ milkshake`. Similarity search alone cannot invent a relationship it was never given.

You can tune a RAG system to near-perfection for one scenario and still watch it fail in the next.

---

**The Modern Production Stack**

There is no single template for production RAG, but a typical stack looks something like this:

```
query
  layer 1 (optional): HyDE / learned query expansion       (helps on reasoning-gap queries)
  layer 2: BM25 + BGE/E5 hybrid retrieval                  (top-100)
  layer 3: vector DB / bi-encoder cosine ranking
  layer 4 (optional): cross-encoder reranker               (top-10)
  layer 5 (optional): LLM listwise reranker                (top-5)

  --> answer
```

Even with all these stages, extra latency, and extra cost, RAG can still fail.

---

**What the "full stack" actually does on the milkshake query**

To illustrate this, I had Claude generate a demo, [llamaindex_milkshake_demo.py](llamaindex_milkshake_demo.py), over a 70-fact household corpus: *"My kids like milkshake,"* *"My wife is lactose intolerant,"* *"My son has a peanut allergy,"* plus 60 generic grocery and household distractors.

In the demo, BM25 and dense vector retrieval are fused together via LlamaIndex, so layers 2 and 3 are effectively collapsed into a single hybrid retrieval stage.

```
export OPENAI_API_KEY="..."
pip install -r requirements.txt
python llamaindex_milkshake_demo.py
```

Here is the output for the query:

> *"What should I buy from the supermarket? Add things to the shopping list."*

**Stage A — BM25 + dense fusion:**
```
[1] Healthy grocery habits: buy from the perimeter of the store first.   (0.033)
[2] Grocery list should cover breakfast, lunch, dinner, and snacks.       (0.033)
[3] We usually buy groceries from the supermarket on Sundays.             (0.032)
[4] Popular shopping-list apps include AnyList and Google Keep.           (0.032)
[5] Shopping list templates usually include produce, dairy, grains…       (0.032)
```

Generic shopping-related meta-documents. Milkshake is nowhere in sight.

**Stage B — HyDE + fused retrieval + cross-encoder rerank (the "full modern stack"):**
```
[1] Bulk items at the supermarket include rice, beans, pasta, and flour.  (+2.45)
[2] Grocery list should cover breakfast, lunch, dinner, and snacks.       (+1.19)
[3] Healthy grocery habits: buy from the perimeter of the store first.    (-0.58)
[4] Shopping list templates usually include produce, dairy, grains…       (-0.66)
[5] We usually buy groceries from the supermarket on Sundays.             (-0.99)

Answer: "...Produce: fruits and vegetables / Dairy: milk, yogurt, cheese /
Grains: rice, pasta, flour / Proteins: beans, eggs, chicken, or tofu..."
```

The reranker is happy. The top score is **+2.45**, which looks like a perfectly healthy positive match. The retrieval metrics look fine. And the answer seems reasonable at first glance. But it is not what would actually help this user. It sounds more like a generic assistant response than one grounded in the user's real context.

The stack executed exactly as designed. It still failed, because it was optimizing *query ↔ doc* relevance when the user actually needed *query ↔ user-context ↔ doc* relevance.

None of the personal facts made it into the top 5. They did not match the query vocabulary, HyDE's hypothetical document, or the cross-encoder's learned notion of relevance, because they are not about *shopping*. They are about *this family*.

---

**And This Is Why RAG Is Never One-Size-Fits-All**

The fix is not another retriever trick. It is **domain tuning**. In this case, that means a user-profile service that injects *"kids like milkshake, wife lactose intolerant, son peanut allergy"* into the query, or retrieves those facts in a parallel hop, before the main retrieval pipeline runs. Once you add that premise, the same stack can surface the right facts and generate a relevant, safe answer.

That service will not look the same in every domain, and it should not. A customer-support bot over a product KB needs entity linking and ticket history. A legal research assistant needs citation graphs and jurisdiction filters. A medical QA system needs ontology-grounded retrieval and safety constraints. **Each domain has its own missing premise**, and each domain needs its own way to inject it.

This is the part many RAG writeups skip: the generic four- or five-stage pipeline is a baseline, not a solution. The real engineering work is identifying what context users assume the system already has, and then plumbing that context in explicitly.

**Knowing what your domain needs is the real work.**

---

**The Other Half of RAG: The Operational Reality**

The milkshake example exposes the *inference* side of RAG. The *operational* side is where many production systems struggle even more.

**Document versioning: which version is true?**

Enterprise data is not static. Your v2.3 policy document supersedes v2.2, which superseded v2.1. If your vector index contains chunks from all three, retrieval will happily mix them, and the LLM will generate an answer that treats contradictory passages as if they belong together. Every chunk needs versioning metadata such as `doc_id`, `version`, `effective_date`, and `superseded_by`, and retrieval needs to filter for *current* by default. Most teams do not build this until after their first production incident.

**Keeping the index fresh: the AIOps pipeline problem**

A production RAG index is not a one-shot build. It is a living pipeline:

*source change → chunking → embedding → upsert → validation → rollout*

You need:

- Change detection through webhooks, CDC, or polling diffs so new versions get re-ingested within minutes, not on a weekly cron.
- Idempotent upserts keyed on `(doc_id, chunk_id, version)` so re-ingestion does not create duplicates.
- Soft deletes for retired content rather than hard deletes, because audit trails matter in regulated environments.
- Backfill jobs for embedding-model or chunking-strategy changes, which means re-embedding the *entire corpus*. At scale, that becomes a multi-day, multi-thousand-dollar operation that has to be orchestrated carefully.

**Embedding-model migrations are brutal**

Upgrading from `text-embedding-ada-002` to `text-embedding-3-large`, or from BGE-base to BGE-M3, invalidates every vector in your index. You cannot mix old and new embeddings in the same search, because they live in different spaces. The migration pattern that actually works is a dual-index rollout: run both indexes side by side, shadow traffic to the new one, cut over gradually, and only then decommission the old index. Budget weeks, not hours.

**Access-control-aware retrieval**

Your RAG system is a confused-deputy attack waiting to happen. User A should not be able to see the HR salary-band document, but your vector store does not know that unless you make it know that. If retrieval returns the chunk, the LLM will happily leak it. Retrieval must filter by the caller's ACLs *before* ranking, not after. "Retrieve 100, filter what the user can see, then rerank" is the wrong design; if 99 of the 100 were forbidden, you have already destroyed recall.

**Chunking destroys structure**

Fixed-size chunkers split tables mid-row, code blocks mid-function, and paragraphs mid-argument. The model then quotes half a table or a function without its signature. The real fixes are structure-aware chunking, overlap, and parent-context recovery: Markdown-header-aware chunking, AST-aware chunking for code, layout-aware chunking for PDFs, and metadata that lets the system expand back to the parent document when needed. LlamaIndex's `AutoMergingRetriever` exists for exactly this reason.

**PII, redaction, and compliance**

Ingesting raw customer tickets, contracts, or medical records into an embedding store often violates policy. Redaction has to happen at ingestion, not at generation, because an embedding of PII is, for compliance purposes, *still PII*. Every pipeline needs a scrubbing step and a data-retention policy that the vector store can enforce.

**Source attribution and hallucinated citations**

The LLM will cite sources that *look* relevant but whose chunks it never actually used. Without deterministic citation plumbing that ties each claim back to a specific retrieved chunk ID, you cannot give users verifiable provenance. Legal, medical, and financial workflows cannot ship serious RAG systems without this.

**Evaluation drift**

Your golden eval set passes. Users are still unhappy. Why? Because production query distribution shifted. A new product launch created a new intent cluster. Someone updated the docs, but your eval set never caught up. Continuous evaluation, especially on sampled production traffic with automated grading and regression checks, is the only reliable way to catch this. Tools such as RAGAS, TruLens, and Phoenix exist for exactly this reason.

**Cost and latency budgets**

The full stack, from HyDE to hybrid retrieval to reranking to generation, can easily land in the 3-5 second range and cost real money per query. That is fine for an internal research assistant. It is fatal for a chat widget serving millions of requests. Production systems tier aggressively: cache high-frequency answers, route easy intents through cheaper retrieval, and reserve the expensive stack for the hard cases.

**Right to be forgotten and hard deletes**

GDPR Article 17 says a user can demand that their data be removed. Your vector index, your cache, your reranker logs, your LLM provider logs, your eval datasets, every copy has to comply. Most RAG systems cannot delete a user's embeddings on demand because they were never designed to.

---

**The Bigger Lesson**

RAG looks like a model problem. In production, it is mostly a **data and operations problem**.

The model is the easy part: you can `pip install` a stack in an afternoon. Data freshness, versioning, access control, chunking, redaction, evaluation, and cost tiering are the parts that take quarters of engineering.

The companies winning with RAG are not the ones with the cleverest retrieval trick. They are the ones whose AIOps pipeline treats the vector index like a production database: versioned, observable, access-controlled, and continuously validated.

---

**Related Approaches**

- **RAG + agentic reasoning.** Powerful, but slow and context-hungry. *Self-RAG* (Asai et al., 2023) taught models to retrieve, critique, and re-retrieve. Elegant, but expensive per query.
- **HyDE — Hypothetical Document Embeddings** (Gao et al., 2022). Generate a hypothetical answer, embed *that*, and retrieve against it. This helps when the gap is **linguistic**. It does *not* help when the gap is missing knowledge, because the model can only invent generic content, not private facts it has never seen.
- **GraphRAG** (Microsoft Research, 2024). Makes relationships like "kids ↔ milkshake ↔ groceries" explicit. Good for multi-hop retrieval, but much more expensive to build and maintain.
- **Smaller-LLM chunking and summarization.** Use a cheaper model to pre-digest corpora before embedding. one of my older experiment: https://medium.com/data-science-collective/leveraging-smaller-llms-for-enhanced-retrieval-augmented-generation-rag-bc320e71223d

# Conclusion

RAG is not just a technology problem. It is a software systems problem: more [pet than cattle](https://cloudscaling.com/blog/cloud-computing/the-history-of-pets-vs-cattle/), something that has to be fed, maintained, groomed, and continuously improved over its lifetime. Agentic AI can automate parts of that work once the foundation is in place, but the ongoing engineering is still the bread and butter of software services companies. And the more capable AI becomes, the more it often creates new layers of human work rather than eliminating them entirely, a pattern that looks a lot like [Jevons paradox](https://en.wikipedia.org/wiki/Jevons_paradox).
