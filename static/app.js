const $ = (id) => document.getElementById(id);
const state = { files: [], history: [] };

const SUGGESTIONS = [
  "Belgenin ana fikrini ve amacını özetle",
  "Metindeki en kritik bulgular nelerdir?",
  "Sayısal veya istatistiki verileri listele",
];

const ICONS = {
  trash: `<svg viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2" stroke-linecap="round" stroke-linejoin="round" aria-hidden="true"><path d="M3 6h18"/><path d="M8 6V4h8v2"/><path d="M19 6l-1 14H6L5 6"/><path d="M10 11v6"/><path d="M14 11v6"/></svg>`,
  copy: `<svg viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2" stroke-linecap="round" stroke-linejoin="round" aria-hidden="true"><rect x="9" y="9" width="13" height="13" rx="2"/><path d="M5 15H4a2 2 0 0 1-2-2V4a2 2 0 0 1 2-2h9a2 2 0 0 1 2 2v1"/></svg>`,
};

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

function shortName(name, max = 42) {
  const text = String(name || "");
  if (text.length <= max) return text;
  const dot = text.lastIndexOf(".");
  const ext = dot > 0 ? text.slice(dot) : "";
  const base = ext ? text.slice(0, dot) : text;
  const keep = Math.max(10, max - ext.length - 1);
  return `${base.slice(0, keep)}…${ext}`;
}

function setRailOpen(open) {
  const rail = $("rail");
  const backdrop = $("railBackdrop");
  const toggle = $("railToggle");
  if (!rail || !toggle) return;
  rail.classList.toggle("is-open", open);
  toggle.setAttribute("aria-expanded", open ? "true" : "false");
  toggle.textContent = open ? "Kapat" : "Belgeler";
  if (backdrop) backdrop.hidden = !open;
}

function renderFiles() {
  $("fileList").innerHTML = state.files
    .map((name) => {
      const href = `/api/files/${encodeURIComponent(name)}`;
      const safe = encodeURIComponent(name);
      const label = shortName(name);
      return `<li>
        <a href="${href}" target="_blank" rel="noopener" title="${escapeHtml(name)}">${escapeHtml(label)}</a>
        <button type="button" class="file-del" data-file="${safe}" title="Belgeyi sil" aria-label="Belgeyi sil">${ICONS.trash}</button>
      </li>`;
    })
    .join("");
  $("emptyFiles").hidden = state.files.length > 0;
  const select = $("sourceSelect");
  const current = select.value;
  select.innerHTML = `<option value="">Tüm belgeler</option>` + state.files
    .map((name) => `<option value="${escapeHtml(name)}" title="${escapeHtml(name)}">${escapeHtml(shortName(name, 48))}</option>`)
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
  const chips = SUGGESTIONS.map(
    (q) => `<button type="button" class="suggest-chip" data-q="${escapeHtml(q)}">${escapeHtml(q)}</button>`
  ).join("");
  $("thread").innerHTML = `
    <div class="empty">
      <strong>Belgeni doğrulanabilir şekilde sor</strong>
      <p class="lead">Önce PDF veya TXT yükle, sonra sor — yanıt kaynak parçalarıyla doğrulanır.</p>
      <div class="suggest">${chips}</div>
    </div>`;
  $("thread").querySelectorAll(".suggest-chip").forEach((btn) => {
    btn.addEventListener("click", () => {
      const q = btn.dataset.q || btn.textContent || "";
      if (!q.trim()) return;
      chat(q.trim());
    });
  });
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

function typingHtml() {
  return `<div class="typing" aria-label="Yanıt hazırlanıyor"><span></span><span></span><span></span></div>`;
}

function addMessage(role, text, { pending = false } = {}) {
  const empty = $("thread").querySelector(".empty");
  if (empty) empty.remove();
  const el = document.createElement("article");
  el.className = `msg ${role}`;
  el.innerHTML = `<div class="who">${role === "user" ? "Sen" : "Asistan"}</div><div class="body"></div>`;
  const body = el.querySelector(".body");
  if (pending) body.innerHTML = typingHtml();
  else if (role === "assistant") body.innerHTML = formatAnswer(text);
  else body.textContent = text;
  $("thread").appendChild(el);
  $("thread").scrollTop = $("thread").scrollHeight;
  return el;
}

function badgeForVerification(info) {
  if (!info) return null;
  const kind = info.kind || (Number(info.confidence_score) >= 60 ? "ok" : "low");
  const score = Number(info.confidence_score);
  let label = info.verification_status || "Doğrulama";
  if (kind === "ok" && Number.isFinite(score)) label = `%${Math.round(score)} Doğrulanmış`;
  else if (kind === "low" && Number.isFinite(score)) label = `%${Math.round(score)} Düşük örtüşme`;
  else if (kind === "reject") label = info.verification_status || "Belgede yok";
  else if (kind === "empty") label = "Doğrulama yok";
  return { kind, label };
}

function renderCitations(chunks) {
  const items = (chunks || []).filter((c) => (c.snippet || c.content || "").trim());
  if (!items.length) return "";
  const rows = items
    .map((c) => {
      const file = c.source_file || "kaynak";
      const page = c.page_number ?? "?";
      const snip = escapeHtml((c.snippet || c.content || "").slice(0, 320));
      const href = `/api/files/${encodeURIComponent(file)}#page=${page}`;
      return `<div class="citation">
        <div class="meta"><a href="${href}" target="_blank" rel="noopener">${escapeHtml(shortName(file, 40))}, Sayfa ${escapeHtml(String(page))}</a></div>
        <div class="snip">${snip}</div>
      </div>`;
    })
    .join("");
  return `<details class="sources">
    <summary>Kullanılan Kaynak Parçaları (${items.length})</summary>
    <div class="sources-body">${rows}</div>
  </details>`;
}

function attachAssistantExtras(el, { answer, verification, chunks, notFound }) {
  if (!el) return;
  el.querySelectorAll(".msg-foot").forEach((node) => node.remove());

  const foot = document.createElement("div");
  foot.className = "msg-foot";

  const badge = badgeForVerification(verification);
  if (badge) {
    const b = document.createElement("span");
    b.className = `badge badge-${badge.kind}`;
    b.textContent = badge.label;
    foot.appendChild(b);
  }

  const showSources =
    !notFound &&
    badge &&
    badge.kind !== "reject" &&
    badge.kind !== "empty";
  const citeHtml = showSources ? renderCitations(chunks) : "";
  if (citeHtml) {
    const wrap = document.createElement("div");
    wrap.style.flexBasis = "100%";
    wrap.innerHTML = citeHtml;
    foot.appendChild(wrap);
  }

  const actions = document.createElement("div");
  actions.className = "msg-actions";
  const copyBtn = document.createElement("button");
  copyBtn.type = "button";
  copyBtn.className = "icon-btn";
  copyBtn.innerHTML = `${ICONS.copy}<span>Kopyala</span>`;
  copyBtn.addEventListener("click", async () => {
    try {
      await navigator.clipboard.writeText(answer || "");
      copyBtn.classList.add("copied");
      copyBtn.querySelector("span").textContent = "Kopyalandı";
      setTimeout(() => {
        copyBtn.classList.remove("copied");
        copyBtn.querySelector("span").textContent = "Kopyala";
      }, 1400);
    } catch {
      copyBtn.querySelector("span").textContent = "Kopyalanamadı";
    }
  });
  actions.appendChild(copyBtn);
  foot.appendChild(actions);
  el.appendChild(foot);
}

function clearChat() {
  state.history = [];
  renderEmpty();
  setRailOpen(false);
  $("query")?.focus();
}

function autoResizeQuery() {
  const el = $("query");
  if (!el) return;
  el.style.height = "auto";
  el.style.height = `${Math.min(el.scrollHeight, 160)}px`;
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
      renderEmpty();
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
  const msg = addMessage("assistant", "", { pending: true });
  msg.querySelector(".who").textContent = "Sistem";
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
  const q = (query || "").trim();
  if (!q) return;
  addMessage("user", q);
  const pending = addMessage("assistant", "", { pending: true });
  $("sendBtn").disabled = true;
  try {
    const data = await api("/api/chat", {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({
        query: q,
        source: $("sourceSelect").value || null,
        history: state.history.slice(-8),
      }),
    });
    const answer = data.answer || "";
    pending.querySelector(".body").innerHTML = formatAnswer(answer);
    const chunks = (data.chunks && (data.chunks.fallback || data.chunks.foundry)) || [];
    attachAssistantExtras(pending, {
      answer,
      verification: data.verification,
      chunks,
      notFound: Boolean(data.not_found),
    });
    state.history.push({ role: "user", content: q });
    state.history.push({ role: "assistant", content: answer });  } catch (err) {
    pending.querySelector(".body").textContent = err.message;
  } finally {
    $("sendBtn").disabled = false;
    $("query").focus();
  }
}

$("fileInput").addEventListener("change", (e) => {
  upload(e.target.files);
  e.target.value = "";
  setRailOpen(false);
});

$("composer").addEventListener("submit", (e) => {
  e.preventDefault();
  const query = $("query").value.trim();
  if (!query) return;
  $("query").value = "";
  autoResizeQuery();
  chat(query);
});

$("query").addEventListener("keydown", (e) => {
  if (e.key === "Enter" && !e.shiftKey) {
    e.preventDefault();
    $("composer").requestSubmit();
  }
});
$("query").addEventListener("input", autoResizeQuery);

$("clearChat")?.addEventListener("click", clearChat);

$("railToggle")?.addEventListener("click", () => {
  const open = !$("rail").classList.contains("is-open");
  setRailOpen(open);
});
$("railBackdrop")?.addEventListener("click", () => setRailOpen(false));
window.addEventListener("keydown", (e) => {
  if (e.key === "Escape") setRailOpen(false);
});

async function loadStatus() {
  const el = $("runtimeStatus");
  if (!el) return;
  const text = el.querySelector(".status-text");
  try {
    const data = await api("/api/status");
    const name = data.model_name || (data.foundry ? "Foundry" : "Yedek");
    const runtime = data.runtime || "Motor";
    el.classList.remove("is-loading", "is-active", "is-fallback");
    el.classList.add(data.foundry ? "is-active" : "is-fallback");
    if (text) text.textContent = `${runtime} · ${name}`;
  } catch {
    el.classList.remove("is-active");
    el.classList.add("is-loading");
    if (text) text.textContent = "Motor durumu alınamadı";
  }
}

renderEmpty();
loadStatus();
loadFiles().catch(() => {});
