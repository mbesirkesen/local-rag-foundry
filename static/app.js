const $ = (id) => document.getElementById(id);
const state = { files: [] };

async function api(path, options = {}) {
  const res = await fetch(path, options);
  const data = await res.json().catch(() => ({}));
  if (!res.ok) throw new Error(typeof data.detail === "string" ? data.detail : "İstek başarısız.");
  return data;
}

function renderFiles() {
  $("fileList").innerHTML = state.files.map((name) => `<li>${name}</li>`).join("");
  $("emptyFiles").hidden = state.files.length > 0;
}

function renderEmpty() {
  $("thread").innerHTML = `<div class="empty">Önce bir PDF veya TXT yükle, sonra sor.</div>`;
}

function addMessage(role, text) {
  const empty = $("thread").querySelector(".empty");
  if (empty) empty.remove();
  const el = document.createElement("article");
  el.className = `msg ${role}`;
  el.innerHTML = `<div class="who">${role === "user" ? "Sen" : "Asistan"}</div><div class="body"></div>`;
  el.querySelector(".body").textContent = text;
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
    pending.querySelector(".body").textContent = data.answer;
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
