(() => {
  "use strict";

  const thread = document.getElementById("thread");
  const typing = document.getElementById("typing");
  const composer = document.getElementById("composer");
  const input = document.getElementById("input");
  const sendButton = document.getElementById("send");
  const statusPill = document.getElementById("status-pill");

  let conversationId = null;
  let finished = false;

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
  }

  function setOutcome(outcome) {
    if (!outcome) return;
    statusPill.textContent = outcome.replace("_", " ");
    statusPill.className = `status-pill ${outcome}`;
  }

  async function callChat(body) {
    const response = await fetch("/api/chat", {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify(body),
    });
    if (!response.ok) {
      const detail = await response.json().catch(() => ({}));
      throw new Error(detail.detail || `request failed (${response.status})`);
    }
    return response.json();
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
      addMessage("agent", data.reply);
      finished = data.finished;
      setOutcome(data.outcome);
      if (finished) {
        addMessage("system", "This conversation has ended.");
      }
    } catch (err) {
      addMessage("system", `Something went wrong: ${err.message}`);
    } finally {
      setTyping(false);
      setBusy(false);
      if (!finished) input.focus();
    }
  });

  start();
})();
