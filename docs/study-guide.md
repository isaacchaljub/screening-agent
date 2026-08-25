# Study guide — explaining the trickier design decisions

Short Q&A entries for design decisions that are easy to get wrong or hand-wave past in a live
walkthrough. Each one: what it is, why it exists, and the concrete evidence that motivated it.

---

## The FAQ interrogative tie-break (`rag/retrieve.py`)

**What is it?**

A ~15-line, pure-Python re-ranking step inside `retrieve()`. FAQ retrieval normally works by
embedding similarity alone: embed the candidate's question, embed all 40 FAQ chunks once (at
index time), and return whichever chunks are closest by cosine distance. The tie-break sits
*after* that similarity search and *before* the top-k cut: it looks at the query and each
candidate's FAQ heading for a recognizable interrogative word — "cuánto"/"how much",
"cuándo"/"when", "cómo"/"how", "dónde"/"where", "qué"/"cuál"/"what"/"which", "por qué"/"why" — and
if the query has one:

- a candidate whose own heading starts with the **same** interrogative word gets a small score
  bonus (+0.05)
- a candidate whose heading starts with a **conflicting** one gets a small penalty (−0.03)
- a candidate with no recognizable interrogative at all is untouched

Candidates are then re-sorted by this adjusted score, not the raw cosine similarity, before the
top-k is taken. The `relevance` field callers see is still the *raw* cosine value — the bonus is
only ever used for ordering, never reported as the hit's actual similarity.

**Why does it exist?**

Not hypothetical — it was found live, in this exact system, while testing the FAQ path:

> Candidate asks **"cuánto pagan por domicilio?"** (how much do you pay per delivery). The agent
> answers **"El pago es semanal, por transferencia bancaria..."** (payment is weekly, by bank
> transfer) — that's the answer to a *different* question, "¿cuándo me pagan?" (when do you get
> paid).

Measured directly: for the query `"cuanto pagan"`, the embedding model
(`intfloat/multilingual-e5-small`, chosen for cost/latency — see README decision 4) scores:

| Candidate FAQ heading | Cosine relevance |
|---|---|
| ¿Cuándo me pagan? (when — **wrong answer**) | 0.8798 |
| ¿Cuánto pagan por entrega? (how much — **right answer**) | 0.8746 |

Both clear the 0.84 relevance floor, so both look "confident" — the model just ranks the wrong one
first, by a margin of 0.0052. "Cuánto" and "cuándo" differ by a single letter, and this specific
embedding model doesn't weight that distinction heavily enough. It's a narrow, specific failure
mode of a small multilingual embedding model on near-homophone interrogatives, not a general
retrieval-quality problem — the same query correctly finds the right chunk once it's slightly more
specific ("cuanto pagan **por entrega**" scores 0.901 vs 0.880, correct order already).

**Why fix it this way, instead of something bigger?**

Three alternatives were on the table and rejected:

1. **Swap embedding models.** Would require re-measuring and recalibrating `DEFAULT_RELEVANCE_FLOOR`
   against the *entire* 40-entry FAQ (see README decision 4's model comparison table) for a fix
   targeted at one word pair. Disproportionate.
2. **Re-weight the embedded text** (e.g. repeat the FAQ heading 2–3× before the answer, so the
   interrogative word dominates more of the embedding). Tested empirically — it helps some queries
   but not all (`"cuanto me pagan"` stayed wrong at every weight tried), because generic
   reweighting doesn't target the actual pivot word, it just dilutes the answer text around it.
3. **A full cross-encoder reranker.** The right call at a few thousand FAQ chunks (see README "What
   I'd do differently"), but it's a new model in the hot path for a defect that's really about two
   specific words. Disproportionate the other way.

The interrogative tie-break is the narrowest fix that actually targets the linguistic thing the
embedding model is getting wrong (the wh-word), costs no extra model call (same "pure Python, no
vendor call" philosophy as `guardrails.classify()`'s off-script detection), and doesn't touch
anything for the ~38 other FAQ chunks that were never confused in the first place.

**How was the fix validated?**

Empirically, against the local embedding model (free — it runs in-process, no API cost) before
touching the real module: computed cosine similarity for the confusable pair across several query
phrasings, confirmed the interrogative-category bonus/penalty flips every case tested (`"cuanto
pagan"`, `"¿Cuánto pagan?"`, `"cuanto pagan por entrega"`, `"cuanto me pagan"`, `"cuanto es la
paga"`, and the reverse `"cuando pagan"`/`"cuando me pagan"` queries), then re-ran it against the
*exact* phrasings from the production conversation that surfaced the bug. Also re-ran the full test
suite including the live `tests/test_retrieval.py` suite (real embeddings) — all passing, including
the tests that specifically assert cross-lingual ranking and the relevance-floor cutoff, which the
tie-break doesn't touch (it only reorders candidates that already cleared the floor).

**What it doesn't fix.** It's a same-category-word tie-break, not a semantic reranker — a
near-miss that isn't a wh-word collision (the README notes `"¿qué vehículo necesito?"` sometimes
ranking a related-but-wrong vehicle FAQ) is out of scope for this fix and would need the general
reranker described in "What I'd do differently."
