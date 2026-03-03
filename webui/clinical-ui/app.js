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
const zoomModal = document.getElementById("zoomModal");
const zoomClose = document.getElementById("zoomClose");
const zoomImg = document.getElementById("zoomImg");
const zoomCaption = document.getElementById("zoomCaption");

const globalProgressBar = document.getElementById("globalProgressBar");
const globalProgressPct = document.getElementById("globalProgressPct");
const globalProgressLabel = document.getElementById("globalProgressLabel");
const globalProgressTrack = document.getElementById("globalProgressTrack");

const txBytes = document.getElementById("txBytes");
const txRate = document.getElementById("txRate");
const serverTime = document.getElementById("serverTime");
const roundTripTime = document.getElementById("roundTripTime");

let selectedFile = null;
let apiOnline = false;
let serverAnim = null;
let focusBeforeZoom = null;

function authHeaders() {
  return apiToken ? { Authorization: `Bearer ${apiToken}` } : {};
}

function clamp(value, min, max) {
  return Math.max(min, Math.min(max, value));
}

function humanBytes(value) {
  const units = ["B", "KB", "MB", "GB"];
  let n = Number(value || 0);
  let u = 0;
  while (n >= 1024 && u < units.length - 1) {
    n /= 1024;
    u += 1;
  }
  return `${n.toFixed(u === 0 ? 0 : 2)} ${units[u]}`;
}

function humanMs(ms) {
  const n = Number(ms || 0);
  if (!Number.isFinite(n) || n <= 0) return "-";
  if (n < 1000) return `${Math.round(n)} ms`;
  return `${(n / 1000).toFixed(2)} s`;
}

function stageEl(key) {
  return document.getElementById(`stage-${key}`);
}

function updatePredictState() {
  predictBtn.disabled = !selectedFile || !apiOnline;
}

function setApiBadge(mode, text) {
  apiStatus.classList.remove("online", "offline", "checking");
  apiStatus.classList.add(mode);
  apiStatus.textContent = text;
}

function setGlobalProgress(percent, label) {
  const p = clamp(Number(percent || 0), 0, 100);
  globalProgressBar.style.width = `${p}%`;
  globalProgressPct.textContent = `${Math.round(p)}%`;
  globalProgressLabel.textContent = label;
  globalProgressTrack.setAttribute("aria-valuenow", String(Math.round(p)));
}

function setStage(key, state, progress = null, text = "") {
  const el = stageEl(key);
  if (!el) return;
  el.classList.remove("is-pending", "is-active", "is-done", "is-error");
  el.classList.add(`is-${state}`);

  const stateText = el.querySelector(".stage-state");
  const fill = el.querySelector(".stage-fill");
  if (!stateText || !fill) return;

  fill.classList.remove("indeterminate");
  if (state === "pending") {
    stateText.textContent = text || "En attente";
    fill.style.width = "0%";
    return;
  }
  if (state === "active") {
    stateText.textContent = text || "En cours";
    if (progress === null) {
      fill.classList.add("indeterminate");
      fill.style.width = "45%";
    } else {
      fill.style.width = `${clamp(progress, 0, 100)}%`;
    }
    return;
  }
  if (state === "done") {
    stateText.textContent = text || "Terminé";
    fill.style.width = "100%";
    return;
  }
  stateText.textContent = text || "Erreur";
  fill.style.width = "100%";
}

function resetPipelineForRun() {
  setStage("upload", "pending");
  setStage("preprocess", "pending");
  setStage("inference", "pending");
  setStage("render", "pending");
  setGlobalProgress(apiOnline ? 8 : 0, apiOnline ? "Prêt pour l'analyse" : "API indisponible");
  txBytes.textContent = selectedFile ? `0 B / ${humanBytes(selectedFile.size)}` : "0 B / 0 B";
  txRate.textContent = "0 B/s";
  serverTime.textContent = "-";
  roundTripTime.textContent = "-";
}

function stopServerAnimation() {
  if (serverAnim) {
    clearInterval(serverAnim);
    serverAnim = null;
  }
}

function startServerAnimation() {
  stopServerAnimation();
  const t0 = performance.now();
  setStage("preprocess", "active", null, "En cours");
  setStage("inference", "pending");
  setStage("render", "pending");

  serverAnim = setInterval(() => {
    const elapsed = performance.now() - t0;
    const p = clamp(45 + elapsed / 150, 45, 92);
    setGlobalProgress(p, "Traitement serveur en cours…");

    if (elapsed > 1200) {
      setStage("preprocess", "done", 100, "Terminé");
      setStage("inference", "active", null, "En cours");
    }
    if (elapsed > 3200) {
      setStage("render", "active", null, "En cours");
    }
  }, 160);
}

function toDataUrl(b64) {
  return `data:image/png;base64,${b64}`;
}

function openZoomFrom(img) {
  if (!img?.src) return;
  const caption = img.closest("figure")?.querySelector("figcaption")?.textContent || "Aperçu agrandi";
  zoomImg.src = img.src;
  zoomCaption.textContent = caption;
  focusBeforeZoom = document.activeElement;
  zoomModal.hidden = false;
  document.body.style.overflow = "hidden";
  zoomClose.focus();
}

function closeZoom() {
  zoomModal.hidden = true;
  zoomImg.src = "";
  document.body.style.overflow = "";
  if (focusBeforeZoom && typeof focusBeforeZoom.focus === "function") {
    focusBeforeZoom.focus();
  }
}

function onFilePicked(file) {
  selectedFile = file || null;
  runMeta.textContent = selectedFile ? `Fichier sélectionné : ${selectedFile.name}` : "";
  resetPipelineForRun();
  updatePredictState();
}

function renderResult(data) {
  const imgs = data.images || {};
  overlayImg.src = imgs.original_overlay_png_b64 ? toDataUrl(imgs.original_overlay_png_b64) : "";
  preprocImg.src = imgs.global_preprocessed_png_b64 ? toDataUrl(imgs.global_preprocessed_png_b64) : "";

  chips.innerHTML = "";
  const counts = data.counts_by_class || {};
  const keys = Object.keys(counts).sort((a, b) => counts[b] - counts[a]);
  if (!keys.length) {
    const c = document.createElement("span");
    c.className = "chip";
    c.textContent = "Aucun signe détecté";
    chips.appendChild(c);
  } else {
    for (const key of keys) {
      const c = document.createElement("span");
      c.className = "chip";
      c.textContent = `${key} · ${counts[key]}`;
      chips.appendChild(c);
    }
  }

  predTable.innerHTML = "";
  const preds = [...(data.predictions || [])].sort((a, b) => Number(b.score || 0) - Number(a.score || 0));
  if (!preds.length) {
    const tr = document.createElement("tr");
    const td = document.createElement("td");
    td.colSpan = 3;
    td.textContent = "Aucune prédiction retenue.";
    tr.appendChild(td);
    predTable.appendChild(tr);
    return;
  }

  for (const pred of preds) {
    const tr = document.createElement("tr");
    const tdClass = document.createElement("td");
    const tdScore = document.createElement("td");
    const tdBbox = document.createElement("td");
    const bbox = (pred.bbox_xyxy || []).map((x) => Math.round(Number(x))).join(", ");

    tdClass.textContent = pred.class_name || "-";
    tdScore.textContent = Number(pred.score || 0).toFixed(3);
    tdBbox.textContent = bbox;

    tr.appendChild(tdClass);
    tr.appendChild(tdScore);
    tr.appendChild(tdBbox);
    predTable.appendChild(tr);
  }
}

function postPrediction(formData) {
  return new Promise((resolve, reject) => {
    const xhr = new XMLHttpRequest();
    const startedAt = performance.now();
    let uploadDone = false;

    xhr.open("POST", `${apiBase}/v1/predict`, true);
    xhr.timeout = 180000;

    const headers = authHeaders();
    for (const [key, value] of Object.entries(headers)) {
      xhr.setRequestHeader(key, value);
    }

    xhr.upload.onprogress = (evt) => {
      if (!evt.lengthComputable) return;
      const percent = clamp((evt.loaded / evt.total) * 100, 0, 100);
      setStage("upload", "active", percent, `${Math.round(percent)}%`);
      setGlobalProgress(10 + percent * 0.35, "Transmission de l'image…");
      txBytes.textContent = `${humanBytes(evt.loaded)} / ${humanBytes(evt.total)}`;
      const elapsedSec = Math.max((performance.now() - startedAt) / 1000, 0.001);
      txRate.textContent = `${humanBytes(evt.loaded / elapsedSec)}/s`;

      if (percent >= 100 && !uploadDone) {
        uploadDone = true;
        setStage("upload", "done", 100, "Terminé");
        startServerAnimation();
      }
    };

    xhr.upload.onload = () => {
      if (uploadDone) return;
      uploadDone = true;
      setStage("upload", "done", 100, "Terminé");
      startServerAnimation();
    };

    xhr.onload = () => {
      stopServerAnimation();
      const totalMs = performance.now() - startedAt;
      roundTripTime.textContent = humanMs(totalMs);

      if (xhr.status < 200 || xhr.status >= 300) {
        reject(new Error(xhr.responseText || `HTTP ${xhr.status}`));
        return;
      }
      try {
        resolve(JSON.parse(xhr.responseText));
      } catch (err) {
        reject(new Error(`Réponse JSON invalide: ${String(err)}`));
      }
    };

    xhr.onerror = () => {
      stopServerAnimation();
      reject(new Error("Erreur réseau pendant l'appel API"));
    };

    xhr.ontimeout = () => {
      stopServerAnimation();
      reject(new Error("Délai dépassé côté client (180 s)"));
    };

    xhr.send(formData);
  });
}

async function loadProfiles() {
  setApiBadge("checking", "API : vérification…");
  setStage("api", "active", null, "Vérification");
  try {
    const res = await fetch(`${apiBase}/v1/profiles`, { headers: authHeaders() });
    if (!res.ok) throw new Error(await res.text());

    const data = await res.json();
    const profiles = data.profiles || {};
    profileSelect.innerHTML = "";
    for (const [key, value] of Object.entries(profiles)) {
      const opt = document.createElement("option");
      opt.value = key;
      opt.textContent = value.display_name || key;
      profileSelect.appendChild(opt);
    }

    apiOnline = true;
    setApiBadge("online", "API : en ligne");
    setStage("api", "done", 100, "Connecté");
    setGlobalProgress(8, "Prêt pour l'analyse");
  } catch (err) {
    apiOnline = false;
    setApiBadge("offline", "API : hors ligne");
    setStage("api", "error", 100, "Échec");
    setGlobalProgress(0, "API indisponible");
    runMeta.textContent = `Impossible de charger les profils: ${String(err).slice(0, 180)}`;
  }
  updatePredictState();
}

async function runPrediction() {
  if (!selectedFile || !apiOnline) return;

  predictBtn.disabled = true;
  runMeta.textContent = "Analyse en cours…";
  resetPipelineForRun();

  try {
    const form = new FormData();
    form.append("file", selectedFile);
    form.append("profile", profileSelect.value);

    const data = await postPrediction(form);
    setStage("preprocess", "done", 100, "Terminé");
    setStage("inference", "done", 100, "Terminé");
    setStage("render", "done", 100, "Terminé");
    setGlobalProgress(100, "Analyse terminée");

    const timings = data.timings_ms || {};
    serverTime.textContent = humanMs(timings.total);
    roundTripTime.textContent = roundTripTime.textContent === "-" ? humanMs(timings.total) : roundTripTime.textContent;

    renderResult(data);
    runMeta.textContent = `Profil: ${data.profile} · Prétraitement ${timings.preprocess ?? "?"} ms · Modèle ${timings.model_inference ?? "?"} ms · Total ${timings.total ?? "?"} ms`;
  } catch (err) {
    stopServerAnimation();
    setStage("inference", "error", 100, "Erreur");
    setStage("render", "error", 100, "Erreur");
    setGlobalProgress(100, "Échec de l'analyse");
    runMeta.textContent = `Échec de l'analyse: ${String(err).slice(0, 220)}`;
  } finally {
    updatePredictState();
  }
}

fileInput.addEventListener("change", (e) => onFilePicked(e.target.files?.[0]));

dropZone.addEventListener("dragover", (e) => {
  e.preventDefault();
  dropZone.classList.add("drag");
});

dropZone.addEventListener("dragleave", () => {
  dropZone.classList.remove("drag");
});

dropZone.addEventListener("drop", (e) => {
  e.preventDefault();
  dropZone.classList.remove("drag");
  onFilePicked(e.dataTransfer?.files?.[0]);
});

predictBtn.addEventListener("click", runPrediction);
overlayImg.addEventListener("click", () => openZoomFrom(overlayImg));
preprocImg.addEventListener("click", () => openZoomFrom(preprocImg));
zoomClose.addEventListener("click", closeZoom);
zoomModal.addEventListener("click", (e) => {
  if (e.target === zoomModal) closeZoom();
});
window.addEventListener("keydown", (e) => {
  if (e.key === "Escape" && !zoomModal.hidden) closeZoom();
});

resetPipelineForRun();
loadProfiles();
