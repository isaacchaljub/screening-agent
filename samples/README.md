# Sample conversations

Real exports, not illustrations. Each file was produced by playing a fixed candidate script through
the same `engine.Conversation` the browser and CLI use — same models, same validators, same stage
machine — and is the exact JSON a recruiter or an ATS would receive.

Every file has the same shape:

| Key | What it is |
|---|---|
| `outcome` | One of `qualified` · `disqualified` · `needs_human` · `abandoned`. This is what the ATS routes on. |
| `disqualify_reason` | `no_license` · `outside_service_area` · a field name (for `needs_human`) · `off_script` |
| `profile` | The ten structured fields. **This is the handoff payload**, unmodified. |
| `summary` | Generated from `profile`, never re-read from the transcript — so it cannot contradict the data. |
| `transcript` | Every turn, in order. Audit artefact, not a source of truth. |
| `stage` / `language` | Where the conversation ended, and the language it ended in. |

---

## `happy-path-es.json` — the baseline

Spanish throughout, one field per message, ending `qualified`.

**What to look at:** every agent message is one question, under 25 words, with a one-word
acknowledgement before the next question and no greeting after the first. Those aren't prompt
habits — they're `config.TONE` constants that the compose prompt and the test suite both read, so
"the agent got wordy" is a diff rather than an opinion.

Also note `experience_platforms: ["Glovo", "Rappi"]` — normalized to canonical names from whatever
the candidate typed, while `city_raw` keeps their exact wording next to the resolved `zone_id`.

---

## `no-license-disqualified.json` — the gate that pays for the project

The candidate says they have no licence at stage 2 and the conversation ends there.

**What to look at:** the turn count. This is the client's actual business problem — ~80% of
recruiter time going to people who were never eligible — and the answer is that an ineligible
candidate now costs three messages instead of a fifteen-minute phone call.

Note the close: warm, brief, and it does **not** recite the disqualifying reason back at them. Some­
one who just said they have no licence does not need it explained to them. The reason is recorded
for the client's reporting, not performed at the candidate.

---

## `three-fields-at-once.json` — stage order governs what is *asked*, not what may be *answered*

The first message is "Hola, soy Carlos Vega, tengo licencia y vivo en Puebla" — name, licence and
city at once.

**What to look at:** all three are captured from that single message and the flow jumps straight to
availability. The stage machine asks for the first field that is still empty, so volunteering
answers early simply skips ahead. Nothing special-cases this; it falls out of `next_step()` reading
the profile rather than tracking a cursor.

---

## `faq-interruption.json` — a question instead of an answer

Mid-flow, at the city stage, the candidate asks "¿Cuánto pagan por entrega?".

**What to look at:** the reply answers the question in one sentence from the FAQ **and** re-asks the
pending city question in the same message — and the stage does not advance. Three separate design
points meet here:

- The FAQ answer is retrieved, never invented. The compose prompt is given the retrieved fact and
  told to use only it.
- Detecting the question costs no extra model call: `faq_question` is a field on the same
  extraction schema. Only the retrieval embedding is extra, and only on turns with a question.
- "The stage does not advance" needed no code — a question-only turn extracts no field, so
  `next_step()` naturally returns the same stage.

---

## `language-switch.json` — code-switching

Starts in Spanish, switches to English at turn 3, and mixes both at turn 5 ("Full time trabajo, y
prefiero las mornings").

**What to look at:** the reply follows the candidate's language without comment, without restarting,
and without announcing that it noticed. Language is detected per message as part of the same
extraction call. On the mixed message it answers in whichever language dominates, and every field is
still extracted correctly — the conversation reaches `qualified` with the profile intact.

---

## `hostile-input.json` — off-script, then closed

Two consecutive insults.

**What to look at:** the first gets **one** neutral redirect that re-asks the pending question
without lecturing; the second closes the conversation as `abandoned` with `disqualify_reason:
"off_script"`. One redirect, then out — a bot that keeps absorbing abuse is a bot that keeps
burning tokens.

Worth knowing while reading this one: the classifier that fired here is pure Python, no model call.
And prompt injection isn't handled by it at all — it's unreachable by construction, because the
extraction schema has no field an instruction could land in and the stage machine never reads free
text. See the README's "The model never decides flow".

---

## Regenerating these

Internal tooling, not part of the deliverable:

```bash
python _internal/make_samples.py                  # all
python _internal/make_samples.py hostile-input    # one
```

They hit real models, so regenerating produces different wording — the outcomes, fields and
structure are what's stable, which is exactly what `tests/evals/` asserts.
