const $ = (id) => document.getElementById(id);
const state = { files: [] };

async function api(path, options = {}) {
  const res = await fetch(path, options);
  const data = await res.json().catch(() => ({}));
  if (!res.ok) throw new Error(typeof data.detail === "string" ? data.detail : "İstek başarısız.");
  return data;
}

function escapeHtml(value) {
  return String(value)
    .replaceAll("&", "&amp;")
    .replaceAll("<", "&lt;")
    .replaceAll(">", "&gt;")
    .replaceAll('"', "&quot;");
}

function renderFiles() {
  $("fileList").innerHTML = state.files
    .map((name) => {
      const href = `/api/files/${encodeURIComponent(name)}`;
      return `<li><a href="${href}" target="_blank" rel="noopener" title="Belgeyi aç">${escapeHtml(name)}</a></li>`;
    })
    .join("");
  $("emptyFiles").hidden = state.files.length > 0;
}

function renderEmpty() {
  $("thread").innerHTML = `<div class="empty">Önce bir PDF veya TXT yükle, sonra sor.</div>`;
}

function formatAnswer(text) {
  const re = /\(Kaynak:\s*([^,\n]+),\s*Sayfa\s*(\d+)\)/g;
  let html = "";
  let last = 0;
  let match;
  while ((match = re.exec(text))) {
    html += escapeHtml(text.slice(last, match.index));
    const file = match[1].trim();
    const page = match[2];
    const href = `/api/files/${encodeURIComponent(file)}#page=${page}`;
    html += `(Kaynak: <a href="${href}" target="_blank" rel="noopener">${escapeHtml(file)}, Sayfa ${page}</a>)`;
    last = match.index + match[0].length;
  }
  html += escapeHtml(text.slice(last));
  return html;
}

function addMessage(role, text) {
  const empty = $("thread").querySelector(".empty");
  if (empty) empty.remove();
  const el = document.createElement("article");
  el.className = `msg ${role}`;
  el.innerHTML = `<div class="who">${role === "user" ? "Sen" : "Asistan"}</div><div class="body"></div>`;
  const body = el.querySelector(".body");
  if (role === "assistant") body.innerHTML = formatAnswer(text);
  else body.textContent = text;
  $("thread").appendChild(el);
  $("thread").scrollTop = $("thread").scrollHeight;
  return el;
}

async function loadFiles() {
  const data = await api("/api/documents");
  state.files = (data.documents || []).map((d) => d.filename);
  renderFiles();
}

async function upload(fileList) {
  if (!fileList.length) return;
  const fd = new FormData();
  [...fileList].forEach((f) => fd.append("files", f));
  const msg = addMessage("assistant", "Belge işleniyor…");
  $("sendBtn").disabled = true;
  try {
    const data = await api("/api/upload", { method: "POST", body: fd });
    state.files = (data.documents || []).map((d) => d.filename);
    renderFiles();
    const names = (data.uploaded || []).map((f) => f.name).join(", ");
    msg.querySelector(".body").textContent = names ? `Yüklendi: ${names}` : "Yüklendi.";
  } catch (err) {
    msg.querySelector(".body").textContent = err.message;
  } finally {
    $("sendBtn").disabled = false;
  }
}

async function chat(query) {
  addMessage("user", query);
  const pending = addMessage("assistant", "Yanıt hazırlanıyor…");
  $("sendBtn").disabled = true;
  try {
    const data = await api("/api/chat", {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ query }),
    });
    pending.querySelector(".body").innerHTML = formatAnswer(data.answer);
  } catch (err) {
    pending.querySelector(".body").textContent = err.message;
  } finally {
    $("sendBtn").disabled = false;
    $("query").focus();
  }
}

$("fileInput").addEventListener("change", (e) => {
  upload(e.target.files);
  e.target.value = "";
});

$("composer").addEventListener("submit", (e) => {
  e.preventDefault();
  const query = $("query").value.trim();
  if (!query) return;
  $("query").value = "";
  chat(query);
});

renderEmpty();
loadFiles().catch(() => {});
