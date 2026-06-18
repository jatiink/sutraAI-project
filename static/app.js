const $ = (id) => document.getElementById(id);

let staged = []; // files waiting to be indexed on next send

// user_id is stable per browser (long-term memory); session_id is per thread
// and resets on "New chat" (short-term memory).
const userId = (() => {
  let id = localStorage.getItem("rag_user_id");
  if (!id) {
    id = "u_" + (crypto.randomUUID?.() || Date.now().toString(36));
    localStorage.setItem("rag_user_id", id);
  }
  return id;
})();
let sessionId = newSessionId();

function newSessionId() {
  return "s_" + (crypto.randomUUID?.() || Date.now().toString(36));
}

$("new-chat").addEventListener("click", async () => {
  try {
    await fetch("/chat/reset", {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ session_id: sessionId }),
    });
  } catch (_) {
    /* resetting is best-effort; a new session id alone is enough client-side */
  }
  sessionId = newSessionId();
  $("messages").innerHTML = "";
  addMessage("bot", "New chat started. I still remember things from your past chats.");
});

// --- attachments ---
$("attach-btn").addEventListener("click", () => $("files").click());

$("files").addEventListener("change", (e) => {
  for (const f of e.target.files) staged.push(f);
  e.target.value = ""; // allow re-picking the same file
  renderChips();
});

function renderChips() {
  const wrap = $("attachments");
  const chips = $("chips");
  chips.innerHTML = "";
  if (!staged.length) {
    wrap.classList.add("hidden");
    return;
  }
  wrap.classList.remove("hidden");
  staged.forEach((f, i) => {
    const chip = document.createElement("div");
    chip.className = "chip";
    chip.innerHTML = `<span>📄 ${escapeHtml(f.name)}</span>`;
    const x = document.createElement("button");
    x.textContent = "×";
    x.onclick = () => {
      staged.splice(i, 1);
      renderChips();
    };
    chip.appendChild(x);
    chips.appendChild(chip);
  });
}

// --- send (handles upload + question in one go) ---
$("send-btn").addEventListener("click", send);

const box = $("question");
box.addEventListener("input", autoGrow);
box.addEventListener("keydown", (e) => {
  if (e.key === "Enter" && !e.shiftKey) {
    e.preventDefault();
    send();
  }
});

async function send() {
  const question = box.value.trim();
  if (!staged.length && !question) return;

  $("send-btn").disabled = true;

  try {
    if (staged.length) await uploadStaged();
    if (question) {
      addMessage("user", question);
      box.value = "";
      autoGrow();
      await askQuestion(question);
    }
  } finally {
    $("send-btn").disabled = false;
  }
}

async function uploadStaged() {
  const fd = new FormData();
  staged.forEach((f) => fd.append("files", f));
  fd.append("access_level", $("access-level").value);
  const names = staged.map((f) => f.name).join(", ");

  const note = addMessage("system", `Indexing ${names}...`);
  try {
    const res = await fetch("/ingest", { method: "POST", body: fd });
    const data = await res.json();
    if (!res.ok) throw new Error(data.detail || "upload failed");
    const total = data.indexed.reduce((n, d) => n + d.chunks, 0);
    note.textContent = `Indexed ${names} — ${total} chunk(s), ${data.total_chunks} total.`;
  } catch (err) {
    note.textContent = "Couldn't index: " + err.message;
  }
  staged = [];
  renderChips();
}

async function askQuestion(question) {
  const placeholder = addMessage("bot", "thinking...");
  placeholder.classList.add("typing");
  try {
    const res = await fetch("/chat", {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({
        question,
        role: $("role").value,
        session_id: sessionId,
        user_id: userId,
      }),
    });
    const data = await res.json();
    renderAnswer(placeholder, data);
  } catch (err) {
    placeholder.classList.remove("typing");
    placeholder.textContent = "Request failed: " + err.message;
  }
}

// --- rendering ---
function addMessage(kind, text) {
  const wrap = document.createElement("div");
  wrap.className = "msg " + kind;
  const bubble = document.createElement("div");
  bubble.className = "bubble";
  bubble.textContent = text;
  wrap.appendChild(bubble);
  $("messages").appendChild(wrap);
  scrollDown();
  return bubble;
}

function renderAnswer(bubble, data) {
  bubble.classList.remove("typing");
  bubble.innerHTML = "";
  const type = data.type || "answer";

  bubble.appendChild(makeBadge(type));
  if (data.cached) bubble.appendChild(makeBadge("cached"));

  const text = document.createElement("div");
  text.textContent = data.answer || "";
  bubble.appendChild(text);

  const g = data.guardrails;
  if (g && (g.pii_masked?.length || g.private_filtered)) {
    const flags = [];
    if (g.pii_masked?.length) flags.push("masked " + g.pii_masked.join(", "));
    if (g.private_filtered) flags.push("removed internal-only content");
    const note = document.createElement("div");
    note.className = "flags";
    note.textContent = "⚠ guardrails — " + flags.join("; ");
    bubble.appendChild(note);
  }

  appendRefs(bubble, data.reference_chunks || []);
  scrollDown();
}

function makeBadge(kind) {
  const b = document.createElement("span");
  b.className = "badge " + kind;
  b.textContent = kind;
  return b;
}

function appendRefs(bubble, refs) {
  if (!refs.length) return;
  const details = document.createElement("details");
  details.className = "refs";
  const summary = document.createElement("summary");
  summary.textContent = `Reference chunks (${refs.length})`;
  details.appendChild(summary);

  for (const r of refs) {
    const div = document.createElement("div");
    div.className = "ref";
    const scores = Object.entries(r.scores || {})
      .map(([k, v]) => `${k}: ${v}`)
      .join(" · ");
    div.innerHTML = `
      <div class="meta">[${r.n}] ${escapeHtml(r.source)} — ${escapeHtml(r.locator)}</div>
      <div>${escapeHtml(r.snippet)}</div>
      <div class="scores">rrf: ${r.rrf}${scores ? " · " + scores : ""}</div>`;
    details.appendChild(div);
  }
  bubble.appendChild(details);
}

function autoGrow() {
  box.style.height = "auto";
  box.style.height = Math.min(box.scrollHeight, 140) + "px";
}

function scrollDown() {
  const m = $("messages");
  m.scrollTop = m.scrollHeight;
}

function escapeHtml(s) {
  return String(s).replace(/[&<>"]/g, (c) =>
    ({ "&": "&amp;", "<": "&lt;", ">": "&gt;", '"': "&quot;" }[c])
  );
}
