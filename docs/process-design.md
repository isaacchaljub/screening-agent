# Screening Process Design — Grupo Sazón delivery drivers

**Purpose.** Replace the first recruiter phone call with a messaging conversation that collects
seven facts, rejects candidates who cannot qualify, and hands the rest to a recruiter with a
structured summary.

**The problem this targets.** Three recruiters run ~15 manual phone screens a day against ~200
applications a week. 60% of candidates never answer the phone, and 80% of recruiter time goes to
people who were never eligible. Both numbers are addressable by moving first contact to a channel
candidates actually answer, and by disqualifying early instead of at the end.

---

## 1. Conversation stages

Stages run in a fixed order. The order is owned by application code, not by the language model —
the model reads answers and writes messages, but never decides what is asked next or who is
rejected.

| # | Stage | Asks for | Branching |
|---|-------|----------|-----------|
| 0 | `GREETING` | — | Introduces the role and the company, states that answers are stored for this application, invites the candidate to start. Always advances. |
| 1 | `NAME` | Full name | Always advances. |
| 2 | `LICENSE` | Valid driver's licence, yes/no | **No → `DISQUALIFIED`.** Hedged or unclear → re-ask once, then flag for a human. |
| 3 | `CITY` | City or working zone | **Outside the service areas → `DISQUALIFIED`.** Unrecognised → re-ask once with nearby zones named. |
| 4 | `AVAILABILITY` | Full-time / part-time / weekends | Always advances. |
| 5 | `SCHEDULE` | Morning / afternoon / evening / flexible | Always advances. |
| 6 | `EXPERIENCE` | Years, and which platforms | Zero experience is valid and does **not** disqualify. |
| 7 | `START_DATE` | When they can begin | Always advances. |
| 8 | `WRAP_UP` | — | Confirms what was captured, states next steps → `QUALIFIED`. |

Terminal states: `QUALIFIED`, `DISQUALIFIED`, `NEEDS_HUMAN`, `ABANDONED`.

**Why the two gates sit at positions 2 and 3.** A candidate who cannot qualify learns so within
four messages instead of twelve. That respects their time, and it is the single largest lever on
the client's wasted-recruiter-hours problem — an ineligible candidate now costs three messages of
compute instead of a fifteen-minute call.

**Answers arriving early are kept.** If a candidate volunteers "hi, I'm Ana, I have a licence and
I'm in Sevilla", all three fields are captured on that turn and the flow resumes at the first
still-empty stage. The stage order governs what is *asked*, not what may be *answered*.

---

## 2. Fields and validation

| Field | Type | Valid | Invalid → |
|-------|------|-------|-----------|
| `full_name` | string | Two or more words, letters and common name punctuation | Re-ask; refusals and placeholders rejected |
| `has_license` | boolean | An unambiguous yes or no | Hedges ("I'm taking the test") count as **no**, confirmed once before disqualifying |
| `city` | string → zone id | Resolves to a listed service zone in Spain or Mexico | Unknown → re-ask once naming the nearest zones; still unknown → disqualify |
| `availability` | enum | `full_time` · `part_time` · `weekends` | Re-ask offering the three options |
| `preferred_schedule` | enum | `morning` · `afternoon` · `evening` · `flexible` | Re-ask offering the four options |
| `experience_years` | number ≥ 0 | Any non-negative number; "none" resolves to 0 | Ranges resolve to their lower bound |
| `experience_platforms` | list | Normalised against a known list; unknown names kept verbatim | Empty is valid when years is 0 |
| `start_date` | date or `immediately` | An ISO date, or a relative expression resolved against today | Past dates re-asked; "not sure" accepted once and flagged |

**Two rules that matter more than the table.**

*Never guess.* If a message is ambiguous, the field is left empty rather than filled with a best
effort. An empty field costs one more message; a wrong field silently corrupts the handoff and is
not visible to anyone until the recruiter phones a candidate about the wrong city.

*Two attempts, then a human.* Each field may be re-asked once. A second failure moves the
conversation to `NEEDS_HUMAN` rather than looping. Loops are the most common way a screening bot
becomes the thing candidates complain about.

---

## 3. Edge cases

**The candidate stops responding.** State is persisted after every turn, so any conversation
resumes exactly where it stopped. A scheduler watches for conversations that are quiet and not
terminal, and sends up to three follow-ups: roughly 45 minutes ("still there?"), one day
(value-led — pay and shift length, the two things candidates actually want to know), and three days
(a final note that the application will close). Any reply cancels the remainder. After the third,
the conversation closes as `ABANDONED`. Follow-ups only send inside waking hours in the candidate's
own country, and never before the city is known. Drop-off stage is recorded, which tells the client
exactly which question loses people.

**Invalid or ambiguous answers.** Validation returns a *reason*, not a boolean, and the reason
becomes the next message — "I didn't catch a city there, do you mean Sevilla or Seville in the
US?" reads as a person paying attention; "Invalid input, please try again" reads as a form. Two
attempts per field, then a human.

**The candidate switches language.** Language is detected per message and the reply follows the
candidate, mid-conversation, without comment or restart. A message mixing Spanish and English gets
a reply in whichever dominates, preserving loanwords the candidate used ("¿tienes coche o moto?"
stays "moto", not "motorcycle"). Both languages are supported end to end, including the FAQ, which
is retrieved by meaning rather than keyword — the knowledge base is embedded as one bilingual
collection with no language filter, so a question finds the closest answer regardless of which
language it was written in. In practice the same-language entry wins, which is the better outcome:
the retrieved fact is then already in the candidate's language and does not have to be translated
on the way into the reply.

**The candidate asks a question instead of answering.** Common enough to design for. The question
is answered from the FAQ in one sentence, then the outstanding stage question is re-asked in the
same message. The stage does not advance.

**Off-script or inappropriate input.** Insults and nonsense get one neutral redirect, then the
conversation closes politely. Attempts to change the agent's instructions are ignored — the stage
machine is not reachable from message text, so the worst case is a wasted turn, not a bypassed
disqualification.

---

## 4. Outcome paths

**Qualified.** A summary is generated from the stored structured fields — never re-read from the
transcript, so it cannot disagree with the data. The candidate is told what happens next and
within what timeframe. A handoff payload (profile, transcript reference, timestamps, outcome) is
written for the recruiter or an ATS.

**Disqualified.** Closed warmly and briefly. The reason is recorded for the client's reporting but
not recited at the candidate beyond what they already know — someone who just said they have no
licence does not need it explained back to them. No recruiter time is spent. Where the block is
temporary (no licence yet), the close leaves the door open to reapply.

**Needs human.** Everything captured so far is preserved and flagged with the field that failed,
so a recruiter picks up with context rather than starting over.

**Abandoned.** Partial data retained, drop-off stage recorded.

---

## 5. Tone and length

This is messaging. The reference is how a good recruiter texts, not how a company emails.

- **One question per message.** Two questions get one answer and lose the other.
- **Under 25 words.** Longer messages get skimmed, and skimmed questions get half-answered.
- **No greeting blocks, no sign-offs, no bullet lists.** After the first message, no "Hi again".
- **Contractions and plain words.** "¿Cuándo puedes empezar?" not "¿Cuál sería su fecha de
  incorporación?"
- **Spanish uses *tú*,** in both markets — this is a driver role, not a bank.
- **Acknowledge, then ask.** "Perfecto" before the next question makes the exchange feel heard.
  One word, not a sentence.
- **Never invent policy.** Pay, contract type and shift patterns come from the FAQ or not at all.
  "I'll check with the team" is an acceptable answer; a plausible-sounding invented number is not.
  The message-writing step is given one retrieved fact and told it may use only that text, so the
  boundary between "answered from the knowledge base" and "made up" is enforced by what the model
  is given, not by asking it to be careful.

These rules live in a configuration file that both the message-writing prompt and the test suite
read, so "the agent got wordy" is a diff rather than an opinion. Every message the system sends
reads them — including the follow-ups sent when a candidate goes quiet, which are composed by a
separate, smaller prompt and are still the same brand speaking.

---

## 6. Voice input

A candidate may reply by voice instead of typing, from a mic button in the browser UI. The
recording is transcribed (ElevenLabs Scribe, auto-detecting language, `voice/elevenlabs.py`) and
the transcript becomes that turn's message — nothing downstream of that point knows or cares
whether the text came from a keyboard or a microphone. Every rule above the fold applies exactly
as written: the two-attempt cap, "never guess" on ambiguous input, and mid-conversation language
switching (§3) all fire on a voice turn precisely as they would on a typed one, without special
casing. Silence or an unintelligible recording transcribes to an empty string, which reads as an
unanswered turn — "didn't catch that as an answer" — the same graceful path already defined for a
silent typed reply.

The agent's replies stay text-only. Speaking the reply back (TTS) is not implemented: it needs a
paid ElevenLabs plan for API access to a usable voice, which the account behind the current key
does not have (a live-verified `402 payment_required` on every voice tried, not a shape or
integration problem). See `voice/elevenlabs.py` and README "Bonus features" for the detail.

---

## 7. Data handling

The candidate is told at the greeting that their answers are stored for this application. That
sentence is the commitment the rest of the design is held to.

- **Answers go to the database; they never go to the application log.** A logged message is reduced
  to a length and a short hash — enough to correlate log lines across a retry, never enough to
  reconstruct what someone wrote. Logs are longer-lived, more widely read, and more often shipped
  to third parties than the database the candidate was told about.
- **No candidate text reaches a model tier that may train on it.** Free vendor tiers generally
  reserve the right to use submitted prompts to improve their products, so the system refuses to
  start if one is selected outside local development. This is checked at startup rather than left
  to a deployment convention.
- **Answering questions from the knowledge base requires no third party at all.** The retrieval
  step runs a small model in-process rather than sending the candidate's question to a vendor. This
  matters because a question is the one thing a candidate writes that is unpredictable — a name or
  a city is a field, but "¿me pagan si me lesiono?" is whatever they chose to say.
- **Voice input is the one honest exception to "no candidate data reaches a free tier."** A
  recording is sent to ElevenLabs for transcription, and `config.assert_model_allowed`'s R7 gate —
  which blocks exactly this for the LLM roles — was never extended to cover it, because voice was
  added after that policy was written, against whatever plan the supplied key happens to be on.
  Before turning voice on for real candidate traffic, this needs the same treatment the first two
  bullets above already got: confirm the plan behind `ELEVENLABS_API_KEY` doesn't reserve training
  rights over submitted audio, or gate it the way free-tier LLM vendors are gated.
