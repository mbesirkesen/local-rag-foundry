const $ = (id) => document.getElementById(id);
const state = { files: [], history: [] };

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
      const safe = encodeURIComponent(name);
      return `<li>
        <a href="${href}" target="_blank" rel="noopener" title="Belgeyi aç">${escapeHtml(name)}</a>
        <button type="button" class="file-del" data-file="${safe}" title="Belgeyi sil" aria-label="Belgeyi sil">Sil</button>
      </li>`;
    })
    .join("");
  $("emptyFiles").hidden = state.files.length > 0;
  const select = $("sourceSelect");
  const current = select.value;
  select.innerHTML = `<option value="">Tüm belgeler</option>` + state.files
    .map((name) => `<option value="${escapeHtml(name)}">${escapeHtml(name)}</option>`)
    .join("");
  if ([...select.options].some((opt) => opt.value === current)) select.value = current;
  else select.value = "";
  $("fileList").querySelectorAll(".file-del").forEach((btn) => {
    btn.addEventListener("click", (e) => {
      e.preventDefault();
      e.stopPropagation();
      removeFile(decodeURIComponent(btn.dataset.file || ""));
    });
  });
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

function verificationLabel(info) {
  if (!info) return "";
  const kind = info.kind || (info.confidence_score >= 60 ? "ok" : "low");
  const status = info.verification_status || "";
  if (kind === "reject" || kind === "empty") return status || "Doğrulama yok";
  const score = Number(info.confidence_score);
  const scoreText = Number.isFinite(score) ? `%${score}` : "";
  return [scoreText, status].filter(Boolean).join(" · ");
}

function setVerification(el, info) {
  if (!el) return;
  el.querySelectorAll(".verify").forEach((node) => node.remove());
  const label = verificationLabel(info);
  if (!label) return;
  const kind = info.kind || (info.confidence_score >= 60 ? "ok" : "low");
  const bar = document.createElement("div");
  bar.className = `verify verify-${kind}`;
  bar.textContent = label;
  el.appendChild(bar);
}

async function loadFiles() {
  const data = await api("/api/documents");
  state.files = (data.documents || []).map((d) => d.filename);
  renderFiles();
}

async function removeFile(name) {
  if (!name) return;
  const short = name.length > 48 ? `${name.slice(0, 45)}…` : name;
  if (!window.confirm(`“${short}” silinsin mi? Dosya ve indeksi kaldırılır.`)) return;
  const msg = addMessage("assistant", "Belge siliniyor…");
  try {
    const data = await api(`/api/documents/${encodeURIComponent(name)}`, { method: "DELETE" });
    state.files = (data.documents || []).map((d) => d.filename);
    renderFiles();
    if (!state.files.length) {
      state.history = [];
      $("thread").innerHTML = "";
      addMessage("assistant", `Silindi: ${name}`);
    } else {
      msg.querySelector(".body").textContent = `Silindi: ${name}`;
    }
  } catch (err) {
    msg.querySelector(".body").textContent = err.message;
  }
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
      body: JSON.stringify({
        query,
        source: $("sourceSelect").value || null,
        history: state.history.slice(-8),
      }),
    });
    const answer = data.answer || "";
    pending.querySelector(".body").innerHTML = formatAnswer(answer);
    setVerification(pending, data.verification);
    state.history.push({ role: "user", content: query });
    state.history.push({ role: "assistant", content: answer });
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

async function loadStatus() {
  const el = $("runtimeStatus");
  if (!el) return;
  try {
    const data = await api("/api/status");
    const name = data.model_name || (data.foundry ? "Foundry" : "Yedek");
    el.textContent = `${data.runtime || "Motor"} · ${name}`;
  } catch {
    el.textContent = "Motor durumu alınamadı";
  }
}

renderEmpty();
loadStatus();
loadFiles().catch(() => {});
