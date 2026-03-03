const cfg = window.UVEITIS_UI_CONFIG || {};
const apiBase = (cfg.apiBaseUrl || "").replace(/\/+$/, "");
const apiToken = cfg.apiToken || "";

const apiStatus = document.getElementById("apiStatus");
const profileSelect = document.getElementById("profileSelect");
const fileInput = document.getElementById("fileInput");
const dropZone = document.getElementById("dropZone");
const predictBtn = document.getElementById("predictBtn");
const runMeta = document.getElementById("runMeta");
const overlayImg = document.getElementById("overlayImg");
const preprocImg = document.getElementById("preprocImg");
const chips = document.getElementById("chips");
const predTable = document.getElementById("predTable");

let selectedFile = null;

function authHeaders() {
  return apiToken ? { Authorization: `Bearer ${apiToken}` } : {};
}

function setStatus(text) {
  apiStatus.textContent = text;
}

function toDataUrl(b64) {
  return `data:image/png;base64,${b64}`;
}

function onFilePicked(file) {
  selectedFile = file || null;
  predictBtn.disabled = !selectedFile;
  runMeta.textContent = selectedFile ? `Selected: ${selectedFile.name}` : "";
}

async function loadProfiles() {
  try {
    const res = await fetch(`${apiBase}/v1/profiles`, { headers: authHeaders() });
    if (!res.ok) throw new Error(await res.text());
    const data = await res.json();
    const profiles = data.profiles || {};
    profileSelect.innerHTML = "";
    Object.entries(profiles).forEach(([key, value]) => {
      const opt = document.createElement("option");
      opt.value = key;
      opt.textContent = value.display_name || key;
      profileSelect.appendChild(opt);
    });
    setStatus("API: online");
  } catch (err) {
    setStatus("API: offline");
    runMeta.textContent = `Could not load profiles: ${String(err).slice(0, 180)}`;
  }
}

function renderResult(data) {
  overlayImg.src = toDataUrl(data.images.original_overlay_png_b64);
  preprocImg.src = toDataUrl(data.images.global_preprocessed_png_b64);

  chips.innerHTML = "";
  const counts = data.counts_by_class || {};
  const keys = Object.keys(counts).sort((a, b) => counts[b] - counts[a]);
  if (!keys.length) {
    const c = document.createElement("span");
    c.className = "chip";
    c.textContent = "No findings";
    chips.appendChild(c);
  } else {
    keys.forEach((k) => {
      const c = document.createElement("span");
      c.className = "chip";
      c.textContent = `${k} · ${counts[k]}`;
      chips.appendChild(c);
    });
  }

  predTable.innerHTML = "";
  const preds = data.predictions || [];
  preds.sort((a, b) => Number(b.score || 0) - Number(a.score || 0));
  preds.forEach((p) => {
    const tr = document.createElement("tr");
    const bbox = (p.bbox_xyxy || []).map((x) => Math.round(Number(x)));
    tr.innerHTML = `
      <td>${p.class_name || ""}</td>
      <td>${Number(p.score || 0).toFixed(3)}</td>
      <td>${bbox.join(", ")}</td>
    `;
    predTable.appendChild(tr);
  });

  const t = data.timings_ms || {};
  runMeta.textContent = `Profile: ${data.profile} · total ${t.total ?? "?"} ms · preprocess ${t.preprocess ?? "?"} ms · model ${t.model_inference ?? "?"} ms`;
}

async function runPrediction() {
  if (!selectedFile) return;
  predictBtn.disabled = true;
  runMeta.textContent = "Running inference…";
  try {
    const form = new FormData();
    form.append("file", selectedFile);
    form.append("profile", profileSelect.value);
    const res = await fetch(`${apiBase}/v1/predict`, {
      method: "POST",
      headers: authHeaders(),
      body: form
    });
    if (!res.ok) throw new Error(await res.text());
    const data = await res.json();
    renderResult(data);
  } catch (err) {
    runMeta.textContent = `Inference failed: ${String(err).slice(0, 220)}`;
  } finally {
    predictBtn.disabled = !selectedFile;
  }
}

fileInput.addEventListener("change", (e) => onFilePicked(e.target.files?.[0]));
dropZone.addEventListener("dragover", (e) => {
  e.preventDefault();
  dropZone.classList.add("drag");
});
dropZone.addEventListener("dragleave", () => dropZone.classList.remove("drag"));
dropZone.addEventListener("drop", (e) => {
  e.preventDefault();
  dropZone.classList.remove("drag");
  onFilePicked(e.dataTransfer?.files?.[0]);
});
predictBtn.addEventListener("click", runPrediction);

loadProfiles();
