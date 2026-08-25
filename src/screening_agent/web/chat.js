(() => {
  "use strict";

  const thread = document.getElementById("thread");
  const typing = document.getElementById("typing");
  const composer = document.getElementById("composer");
  const input = document.getElementById("input");
  const sendButton = document.getElementById("send");
  const statusPill = document.getElementById("status-pill");
  const micButton = document.getElementById("mic");
  const micIcon = micButton.querySelector(".mic-icon");

  let conversationId = null;
  let finished = false;
  let mediaRecorder = null; // set only while a recording is in progress

  function addMessage(role, text) {
    const el = document.createElement("div");
    el.className = `msg ${role}`;
    el.textContent = text;
    thread.appendChild(el);
    thread.scrollTop = thread.scrollHeight;
    return el;
  }

  function setTyping(on) {
    typing.classList.toggle("hidden", !on);
    if (on) thread.scrollTop = thread.scrollHeight;
  }

  function setBusy(busy) {
    input.disabled = busy || finished;
    sendButton.disabled = busy || finished;
    // Recording itself isn't "busy" (it doesn't block on the network) — only disable the mic
    // while a request from a *previous* turn is in flight, same as the text input.
    if (!mediaRecorder) micButton.disabled = busy || finished;
  }

  function setOutcome(outcome) {
    if (!outcome) return;
    statusPill.textContent = outcome.replace("_", " ");
    statusPill.className = `status-pill ${outcome}`;
  }

  async function postJson(url, body) {
    const response = await fetch(url, {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify(body),
    });
    return readJsonOrThrow(response);
  }

  async function postForm(url, formData) {
    const response = await fetch(url, { method: "POST", body: formData });
    return readJsonOrThrow(response);
  }

  async function readJsonOrThrow(response) {
    if (!response.ok) {
      const detail = await response.json().catch(() => ({}));
      throw new Error(detail.detail || `request failed (${response.status})`);
    }
    return response.json();
  }

  function callChat(body) {
    return postJson("/api/chat", body);
  }

  // Applies a turn's outcome/finished state the same way regardless of whether it came from
  // typed text or a transcribed voice reply — one input path onto one conversation (api.py).
  function applyTurnResult(data) {
    addMessage("agent", data.reply);
    finished = data.finished;
    setOutcome(data.outcome);
    if (finished) {
      addMessage("system", "This conversation has ended.");
    }
  }

  async function start() {
    setTyping(true);
    try {
      const data = await callChat({});
      conversationId = data.conversation_id;
      statusPill.textContent = "in progress";
      addMessage("agent", data.reply);
      finished = data.finished;
      setOutcome(data.outcome);
    } catch (err) {
      addMessage("system", `Couldn't reach the agent: ${err.message}`);
      statusPill.textContent = "offline";
    } finally {
      setTyping(false);
      setBusy(false);
      input.focus();
    }
  }

  composer.addEventListener("submit", async (event) => {
    event.preventDefault();
    const text = input.value.trim();
    if (!text || finished || !conversationId) return;

    addMessage("candidate", text);
    input.value = "";
    setBusy(true);
    setTyping(true);

    try {
      const data = await callChat({ conversation_id: conversationId, message: text });
      applyTurnResult(data);
    } catch (err) {
      addMessage("system", `Something went wrong: ${err.message}`);
    } finally {
      setTyping(false);
      setBusy(false);
      if (!finished) input.focus();
    }
  });

  // --- voice input (speech-to-text via ElevenLabs Scribe, api.py's /api/chat/voice) ---
  // Hidden entirely unless both the browser can record audio and the server has a key
  // configured (GET /api/health's voice_input) — see checkVoiceAvailability below.

  const _RECORDER_MIME_TYPES = ["audio/webm;codecs=opus", "audio/webm", "audio/mp4"];

  function pickRecorderMimeType() {
    if (typeof MediaRecorder === "undefined") return null;
    return _RECORDER_MIME_TYPES.find((type) => MediaRecorder.isTypeSupported(type)) || "";
  }

  async function checkVoiceAvailability() {
    if (pickRecorderMimeType() === null || !navigator.mediaDevices?.getUserMedia) return;
    try {
      const health = await fetch("/api/health").then((r) => r.json());
      if (health.voice_input === "ready") micButton.classList.remove("hidden");
    } catch {
      // Health check failed — leave the mic button hidden rather than offering a feature that
      // may not work; the text composer already reports the outage via start()'s own catch.
    }
  }

  async function sendVoiceTurn(blob, mimeType) {
    if (finished || !conversationId) return;
    setBusy(true);
    setTyping(true);
    try {
      const extension = mimeType.includes("mp4") ? "mp4" : "webm";
      const formData = new FormData();
      formData.append("conversation_id", conversationId);
      formData.append("audio", blob, `recording.${extension}`);
      const data = await postForm("/api/chat/voice", formData);
      addMessage("candidate", data.transcript || "(didn't catch that)");
      applyTurnResult(data);
    } catch (err) {
      addMessage("system", `Something went wrong: ${err.message}`);
    } finally {
      setTyping(false);
      setBusy(false);
    }
  }

  async function startRecording() {
    let stream;
    try {
      stream = await navigator.mediaDevices.getUserMedia({ audio: true });
    } catch {
      addMessage("system", "Couldn't access the microphone — check your browser permissions.");
      return;
    }
    const mimeType = pickRecorderMimeType();
    const recorder = new MediaRecorder(stream, mimeType ? { mimeType } : undefined);
    const chunks = [];
    recorder.addEventListener("dataavailable", (event) => {
      if (event.data.size > 0) chunks.push(event.data);
    });
    recorder.addEventListener("stop", () => {
      stream.getTracks().forEach((track) => track.stop());
      mediaRecorder = null;
      micButton.classList.remove("recording");
      micButton.setAttribute("aria-label", "Record a voice reply");
      micIcon.textContent = "🎤";
      const blob = new Blob(chunks, { type: recorder.mimeType });
      if (blob.size > 0) sendVoiceTurn(blob, recorder.mimeType);
    });
    recorder.start();
    mediaRecorder = recorder;
    micButton.classList.add("recording");
    micButton.setAttribute("aria-label", "Stop recording");
    micIcon.textContent = "⏹";
  }

  micButton.addEventListener("click", () => {
    if (mediaRecorder) {
      mediaRecorder.stop(); // response handling happens in the recorder's own "stop" listener
    } else {
      startRecording();
    }
  });

  checkVoiceAvailability();
  start();
})();
