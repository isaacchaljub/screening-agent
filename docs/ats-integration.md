# ATS integration — design

**Status: design only.** Nothing in this document is implemented. It specifies the contract the
screening agent would expose to Grupo Sazón's applicant tracking system, and the assumptions
behind it. `store.export_json()` already produces the payload described in §3 — what is missing
is the transport, the auth, and the retry semantics.

The design goal is narrow and worth stating up front: **the ATS should never need to read a
transcript to know what happened.** Every decision the agent makes is already a typed field by
the time a conversation ends, so the integration ships structured data and treats the transcript
as an audit artifact, not as a source of truth.

---

## 1. Direction of integration

Two flows, and they are not symmetric.

| Flow | Direction | Trigger | Why this direction |
|---|---|---|---|
| **Applicant intake** | ATS → agent | A new application lands in the ATS | The ATS owns the candidate record and the requisition. The agent is a step in its pipeline, not a parallel system of record. |
| **Screening outcome** | Agent → ATS | A conversation reaches a terminal state | Push, not poll. ~200 applications/week is far too little volume to justify the ATS polling, and a push carries the outcome at the moment a recruiter could act on it. |

The agent deliberately does **not** own candidate identity. It receives an `external_candidate_id`
at intake and echoes it back on every callback. If the ATS and the agent ever disagree about who a
candidate is, the ATS wins — it is the system a recruiter actually works in.

---

## 2. Intake — `POST /api/v1/screenings`

Creates a screening and returns the link the candidate is sent (WhatsApp, SMS, or email —
whichever channel the client already uses; the agent only cares that it terminates in the chat UI).

```http
POST /api/v1/screenings
Authorization: Bearer <token>
Idempotency-Key: 8f0d1c9e-7b2a-4f1e-9c33-2a5b6d4e8f10
Content-Type: application/json
```

```json
{
  "external_candidate_id": "ats-CAND-99213",
  "requisition_id": "REQ-MAD-DRIVER-2026-08",
  "locale_hint": "es",
  "channel": "whatsapp",
  "contact": { "phone_e164": "+34600111222" },
  "prefill": { "full_name": "Laura Fernández", "city_raw": "Barcelona" }
}
```

| Field | Required | Notes |
|---|---|---|
| `external_candidate_id` | yes | Opaque to the agent. Echoed on every callback. The correlation key. |
| `requisition_id` | yes | Determines which service zones are acceptable for *this* role, rather than the global zone list. |
| `locale_hint` | no | A starting language only. Detection per message still overrides it — the candidate's actual language wins over what the ATS believed. |
| `contact` | yes | Never returned in any callback; see §6. |
| `prefill` | no | Fields already known from the application form. Treated as **candidate-asserted, not validated** — they go through `validators.py` exactly as a typed answer would, and a prefilled city that fails to resolve is re-asked rather than trusted. |

**Response `201`:**

```json
{
  "screening_id": "6be07cab-3f21-4a90-b0c2-8d1e5f7a4c33",
  "status": "pending",
  "candidate_url": "https://screening.gruposazon.example/c/6be07cab…",
  "expires_at": "2026-09-01T09:00:00Z"
}
```

`Idempotency-Key` is required, not optional. An ATS that retries a timed-out create must not
produce a second screening link for the same person — the candidate would receive two messages and
the recruiter two partial profiles. Replaying a key returns the original `201` body unchanged.

---

## 3. Outcome callback — `POST <client_webhook_url>`

Fired once per terminal state. This is the payload that matters, and it is deliberately the same
shape `store.export_json()` already writes to disk today.

```json
{
  "event": "screening.completed",
  "event_id": "evt_01J9X2K4M8",
  "occurred_at": "2026-08-25T11:42:07Z",
  "screening_id": "6be07cab-3f21-4a90-b0c2-8d1e5f7a4c33",
  "external_candidate_id": "ats-CAND-99213",
  "outcome": "qualified",
  "disqualify_reason": null,
  "language": "es",
  "profile": {
    "full_name": "Laura Fernández",
    "has_license": true,
    "city_raw": "Barcelona",
    "zone_id": "barcelona",
    "availability": "full_time",
    "preferred_schedule": "afternoon",
    "experience_years": 3.0,
    "experience_platforms": ["Glovo", "Rappi"],
    "start_date": "2026-09-15",
    "starts_immediately": false
  },
  "summary": "Laura Fernández — Barcelona. Licence: yes. Availability: full_time, afternoon shift. Experience: 3.0 years (Glovo, Rappi). Start: 2026-09-15.",
  "metrics": {
    "turns": 9,
    "duration_seconds": 412,
    "drop_off_stage": null,
    "nudges_sent": 0
  },
  "transcript_url": "https://screening.gruposazon.example/api/v1/screenings/6be07cab…/transcript"
}
```

Four properties of this payload are load-bearing:

- **`outcome` is an enum, not prose** — `qualified` · `disqualified` · `needs_human` ·
  `abandoned`. The ATS routes on it. Nothing about routing requires reading `summary`.
- **`summary` is generated from `profile`, never from the transcript.** It therefore cannot
  contradict the structured fields. This is the same guarantee `engine.generate_summary()` makes
  today, and it is the reason a recruiter can trust the one-line version.
- **`zone_id` is resolved, `city_raw` is preserved.** The ATS gets both the normalized zone it can
  filter on and the exact string the candidate typed, which is what a recruiter needs if the
  resolution ever looks wrong.
- **The transcript is a URL, not a body.** It is much larger than the rest of the payload, needed
  in a minority of cases, and subject to a different retention rule (§6).

### Outcome → recommended ATS action

| `outcome` | `disqualify_reason` | What the ATS should do |
|---|---|---|
| `qualified` | — | Advance to recruiter review. This is the only path that should consume recruiter time. |
| `disqualified` | `no_license` | Reject, **flagged as temporary.** The candidate is likely eligible later — this is a re-marketing list, not a blacklist. |
| `disqualified` | `outside_service_area` | Reject. Re-queue automatically if a zone is later added in their country. |
| `needs_human` | field name | Route to a recruiter **with the partial profile attached.** The failing field is named so the call starts where the conversation stopped. |
| `abandoned` | `off_script` or null | No recruiter action. `metrics.drop_off_stage` is the reporting signal — it is what tells the client which question actually loses people. |

### Delivery semantics

At-least-once, with an exponential backoff ladder (1m, 5m, 30m, 2h, 12h) and a dead-letter queue
after that. `event_id` is stable across retries, so **the consumer must deduplicate on it.** Every
callback is signed: `X-Signature: sha256=<hmac(body, shared_secret)>` plus a timestamp header,
rejected outside a 5-minute window to bound replay.

A second event type, `screening.needs_human`, fires on the same channel for the escalation path,
so a recruiter can be paged without waiting for a terminal state.

---

## 4. What the agent does *not* accept from the ATS

Worth stating explicitly, because it is a design decision rather than an omission:

- **No stage overrides.** The ATS cannot ask the agent to skip the licence question. The stage
  machine's guarantees (see the README's R1) hold only because nothing outside `stages.py` can
  reorder it.
- **No outcome overrides.** A recruiter can of course reject a `qualified` candidate — in the
  ATS, on their own record. The agent's outcome is what the *screening* concluded and stays
  immutable, so the client's funnel reporting stays honest.
- **No free-text prompt injection point.** There is no `custom_instructions` field. Per-requisition
  variation belongs in configuration (zones, tone constants) rather than in text a model reads, for
  the same reason `guardrails.py` documents: the flow must not be reachable from message content.

---

## 5. Zone and requisition mapping

`requisition_id` maps to a subset of `config.ZONES`. A Madrid requisition should disqualify a
Guadalajara candidate for *that role* while leaving them eligible for a Mexican one. The mapping
lives on the agent's side, synced from the ATS via `GET /api/v1/requisitions` at intake time and
cached — a stale zone list is a worse failure than a slow one, because it silently rejects
eligible people.

---

## 6. Data protection

The candidate is told at the `GREETING` stage that their answers are stored for this application.
That sentence is the basis for everything below, and it constrains the integration:

- **Contact details are write-only to the agent.** `contact` is accepted at intake and never
  appears in any callback, log line, or export. The ATS already has it.
- **Transcripts have a shorter retention than profiles.** The structured profile is the record of
  the application; the transcript is evidence for a disputed decision. Default: profile retained
  per the client's ATS policy, transcript 90 days, then dropped while the profile stays.
- **Logs never contain candidate text.** `guardrails.redact_for_log()` reduces a message to a
  length and a short hash — enough to correlate log lines across a retry, never enough to
  reconstruct what someone wrote. Application logs are longer-lived and more widely readable than
  the database, and are sometimes shipped to third-party aggregators.
- **Deletion is a supported operation.** `DELETE /api/v1/screenings/{id}` hard-deletes the
  transcript and profile and returns a tombstone, so an ATS-side erasure request propagates rather
  than leaving a copy here.
- **Free-tier models are refused outside dev.** `config.assert_model_allowed()` raises at startup
  if a free-tier vendor is selected while `APP_ENV != dev`, because those tiers permit the vendor
  to train on prompts. Candidate data is not eligible for that, and the control is enforced in
  code rather than left to a deployment convention.

---

## 7. Failure modes, and what the ATS sees

| Failure | Candidate experience | ATS experience |
|---|---|---|
| Model vendor outage | Falls back to the backup vendor mid-conversation (R5); the candidate notices nothing | Nothing — the conversation completes |
| Both vendors down | `503` with a plain retry message; state is persisted, so resuming loses nothing | No callback yet; the screening stays `in_progress` |
| Candidate never replies | Up to three nudges, then a warm close | `screening.completed` with `outcome: "abandoned"` and `drop_off_stage` set |
| Agent cannot parse an answer twice | Told a recruiter will follow up | `outcome: "needs_human"` with the failing field named |
| Webhook endpoint down | Nothing — the conversation is already over | Retried on the backoff ladder, then dead-lettered; the screening remains readable via `GET /api/v1/screenings/{id}` |
