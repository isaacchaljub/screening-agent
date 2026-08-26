# Deployment, monitoring, and scaling to 10K candidates/week

What is in the repository today runs as a single container and is honestly sized for a demo and a
pilot. This document says what that is, exactly where it breaks, and what each break costs to fix.
The ordering matters more than the destination: the first bottleneck is not throughput.

---

## 1. What ships today

```
┌──────────────┐   POST /api/chat    ┌──────────────────────────────┐
│  Browser     │────────────────────▶│  uvicorn · api.py (1 worker) │
│  web/chat.js │◀────────────────────│  ├─ engine.Conversation      │
└──────────────┘                     │  ├─ llm/ (retry + fallback)  │
                                     │  └─ APScheduler sweep        │
                                     └───────────┬──────────────────┘
                                                 │
                          ┌──────────────────────┼──────────────────────┐
                          ▼                      ▼                      ▼
                   data/screening.db      data/chroma/           vendor APIs
                   (SQLite)               (Chroma, on disk)      (Anthropic, Google)
```

One process, three local dependencies, no network state. `docker run` serves it.

**The honest limits of this shape**, in the order they bite:

1. **In-flight conversations live in a module-level dict** (`api._conversations`). `Conversation`
   carries turn-scoped state — `attempts`, `history`, detected language — that is not fully
   reconstructible from SQLite. A restart drops every in-flight conversation, and a second worker
   would serve some requests from a process that has never seen the conversation. **This is why
   the Dockerfile pins `--workers 1`**, and it is the first thing to fix — not because of load,
   but because it makes any deploy a small outage.
2. **SQLite** is fine for a pilot and wrong the moment there are two writers.
3. **APScheduler runs in-process**, so N replicas means N duplicate nudge sweeps — the candidate
   gets N copies of "still there?".
4. **Chroma is on local disk**, so each replica embeds and stores its own copy of the same 40
   chunks.

Note that (1), (3) and (4) are all the same underlying problem: *state that is implicitly
single-process*. Scaling is mostly the work of naming that state and moving it somewhere shared.

---

## 2. Target shape

```
                    ┌──────────────┐
   WhatsApp / SMS ─▶│  API gateway │─▶ rate limit, auth, TLS
   Browser ────────▶└──────┬───────┘
                           ▼
                 ┌────────────────────┐     ┌──────────────────────┐
                 │ api (N replicas)   │────▶│ Redis                │
                 │ stateless          │◀────│ conversation state   │
                 └─────────┬──────────┘     │ + idempotency keys   │
                           │                └──────────────────────┘
                           ▼
                 ┌────────────────────┐     ┌──────────────────────┐
                 │ Postgres           │     │ Celery beat + workers│
                 │ conversations,     │◀────│ re-engagement sweep  │
                 │ turns, profiles    │     │ ATS webhook delivery │
                 └────────────────────┘     └──────────────────────┘
                           │
                           ▼
                 ┌────────────────────┐
                 │ pgvector           │  (Chroma's replacement — see §4)
                 └────────────────────┘
```

### 2.1 Make the API stateless

Move `api._conversations` into Redis: serialize `profile`, `attempts`, `history`, `language`,
`stage` under `conv:{id}` with a TTL matching the re-engagement window (~4 days, one day past the
last nudge). `Conversation` already keeps all of it in plain attributes, and `CandidateProfile` is
a pydantic model, so this is a serializer and two calls in `api.py` — not a redesign. Nothing in
`engine.py`, `stages.py`, or `validators.py` changes.

Once conversations are in Redis, replicas are interchangeable, deploys stop dropping conversations,
and `--workers 1` can go.

### 2.2 Postgres instead of SQLite

`store.py` is SQLAlchemy already, so the engine URL is the change. Two things to fix in the same
pass, both currently masked by SQLite:

- `store._as_utc()` exists because **SQLite has no native datetime type** and round-trips an
  aware `datetime` back as naive. Postgres has `timestamptz` and the shim becomes dead code.
- Add the indices the pilot never needed: `conversations(outcome, last_candidate_activity)` for the
  re-engagement sweep, `turns(conversation_id, turn_index)` for transcript reads.

### 2.3 Celery instead of APScheduler

Already anticipated — `reengage/scheduler.py`'s docstring specifies the swap, and `run_once()`,
`Store.list_active()` and `Store.record_nudge()` need no changes. Celery beat dispatches
`reengage.sweep` on the interval; workers pull it. One dispatch, N workers, no duplicate nudges.

The sweep query must become "conversations due for a nudge" rather than `list_active()` +
filter-in-Python once the table is large — `next_nudge()` stays pure, it just gets a smaller
candidate set.

---

## 3. Does this reach 10K candidates/week?

10,000/week is ~60/hour averaged, but recruitment traffic is not flat — assume a 5× evening peak,
so **~300 concurrent-ish conversations at peak** and ~1,400 model calls/hour.

The measured per-conversation figures (see the README's bake-off) are ~6–9 turns, 2 model calls per
turn, and a cost in the low cents. Two observations:

- **The bottleneck is vendor rate limits and latency, not our compute.** Each turn is two
  sequential API calls; the process is almost entirely waiting on I/O. Three or four API replicas
  handle this comfortably. What needs headroom is the vendor account's requests-per-minute.
- **Cost scales linearly and predictably**, because the token count per turn is bounded by design
  — `config.TONE.max_words` caps the reply, and the extraction schema is fixed. There is no
  unbounded-context growth path here; a conversation is 6–9 short turns and ends.

Concretely, at 10K/week and the measured blended cost, the model spend is roughly a few hundred
dollars a week — set against three recruiters' time, ~80% of which currently goes to candidates
who were never eligible. That comparison, not the absolute number, is the one to put in front of
the client.

### The three things that actually need work at that volume

1. **Per-vendor rate limiting and a request queue.** Today `llm/retry.py` reacts to a 429 after the
   fact (it does parse the vendor's own suggested delay, which matters). At 1,400 calls/hour a
   token-bucket limiter *in front* of the client, shared across replicas via Redis, converts a
   thundering herd into a queue.
2. **Prompt caching.** Extract's system prompt is fixed and compose's is nearly fixed; the growing
   part is the transcript, which is appended at the end. That is the ideal shape for a cached
   prefix, and it is the single largest cost lever available — worth more than any model swap.
3. **Batching the re-engagement sweep.** At 10K/week there are tens of thousands of open
   conversations. The sweep must be a bounded query with pagination, not a full scan.

### What does *not* need to change

The stage machine, the validators, and the guardrail classifier are pure functions with no I/O.
They cost microseconds and are horizontally scalable by construction. The part of this system that
makes the decisions is the part that will never be the bottleneck — which is a direct consequence
of R1 (see the README), not a coincidence.

---

## 4. Retrieval at scale

40 chunks in Chroma on local disk is right for a pilot and wrong for replicas. Two options:

- **pgvector**, if Postgres is already there. One less system to operate, and the FAQ is small
  enough that an exact scan beats an approximate index. This is the recommendation.
- **A managed vector store**, if the knowledge base grows past a few thousand chunks or gets
  per-client scoping.

Either way, add a **cache on the embedding call keyed by the query text**. Candidates ask the same
handful of questions ("¿cuánto pagan?" dominates), so a cache with a long TTL removes most
retrieval embedding calls outright.

The FAQ index is built by `python -m screening_agent.rag.index --rebuild`. In a real deployment
that is a **job that runs on FAQ change**, not on container start — otherwise every replica
re-embeds the same content on every deploy, and a vendor outage during a rollout becomes a failed
rollout.

---

## 5. Monitoring

The failure mode to design for is not the loud one. A crash pages someone; **an agent that has
quietly started mis-extracting cities does not**, and it corrupts handoffs for days before a
recruiter notices they are calling people about the wrong city.

### Alert on these

| Signal | Why | Suggested trigger |
|---|---|---|
| `needs_human` rate by field | The single best proxy for extraction quality. A jump in one field means a prompt or validator regression, and it names the culprit. | >2× the 7-day baseline for any field |
| Fallback rate by vendor | `fallback.py` already logs every primary→backup swap with both model names and the exception type. A rising rate is a vendor degrading before it fully fails. | any sustained non-zero rate |
| Terminal-outcome mix | `qualified` / `disqualified` / `needs_human` / `abandoned` proportions are stable for a stable funnel. A shift means something upstream changed. | ±10pp week over week |
| p95 turn latency | Two sequential model calls per turn; this is what the candidate feels in a messaging UI. | >8s |
| Truncation errors | `TruncatedResponseError` means a model spent its output budget on reasoning. Silent before it was given its own type — see the README. | any |
| Schema-retry rate | `extract.py` retries a bad parse against the same model. A rising rate is a model or schema drift signal. | >1% of extractions |
| Cost per completed conversation | Catches a prompt that quietly grew, or a model swap that was more expensive than expected. | >1.5× baseline |

### Instrument these

- **Structured logs**, already PII-safe by construction: `guardrails.redact_for_log()` reduces a
  message to a length and a short hash. Keep it that way — candidate text belongs in the database
  the candidate was told about, not in a log stream that gets shipped to an aggregator.
- **A trace per turn**, spanning extract → validate → compose, with the model that actually served
  each call. `TextResult.model` and `StructuredResult.model` already carry the vendor-qualified
  name specifically so a trace can record who served a call after a fallback.
- **The eval suite as a canary.** `tests/evals/` is 12 scenarios with expected outcomes. Run it
  against production models on a schedule, not just in CI — it is the only thing that detects a
  *vendor-side* model change, which no amount of pinning on our side prevents.

### Drop-off reporting

`metrics.drop_off_stage` is recorded on every abandoned conversation. It answers the client's real
question — *which question loses people* — and it is the input to the highest-leverage change
available: reordering or rewording a single stage.

---

## 6. Rollout

1. **Shadow.** Run against real applications, hand every outcome to a recruiter to check without
   acting on it. Measure agreement per field. This is where a wrong-field bug surfaces cheaply.
2. **One zone, one language.** Madrid, Spanish. Small enough that a recruiter can review every
   conversation.
3. **Widen by zone**, keeping `needs_human` routed to a human the whole time.
4. **Add channels.** The browser UI is the demo surface; WhatsApp is where a candidate who ignores
   a phone call will actually answer. `api.py` is already the seam — a channel adapter maps
   inbound webhooks onto `POST /api/chat`.

Keep two things through all of it: the free-tier guard (`APP_ENV != dev` refuses free-tier models,
because those tiers permit training on prompts), and the eval suite as a release gate.

---

## 7. Configuration and secrets

| Variable | Purpose |
|---|---|
| `APP_ENV` | `dev` · `demo` · `prod`. Anything but `dev` refuses free-tier models at startup. |
| `ANTHROPIC_API_KEY` | Primary vendor (extract + compose) |
| `GEMINI_API_KEY` | Dev-only backup for extract/compose. Not used for embeddings — those run in-process (§4). |
| `GROQ_API_KEY` | Eval sweeps only — never a live primary or backup |
| `OPENAI_API_KEY` | Present but unused; see the README |

Keys come from the environment via `config.py` and nowhere else — no module calls `os.getenv` for
a vendor key on its own. In a real deployment they come from a secret manager injected at runtime;
`.env` is a development convenience and is both git- and docker-ignored.

**The R7 guard is a deployment control, not a lint.** `config.assert_model_allowed()` raises at
process start — not mid-conversation — if a free-tier model is selected outside `dev`. Free tiers
permit the vendor to train on submitted prompts, and candidate PII is not eligible for that. It is
enforced in code because a deployment convention would eventually be violated by a hurried rollback.
