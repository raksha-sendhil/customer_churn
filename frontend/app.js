const uploadForm = document.getElementById('upload-form');
const fileInput = document.getElementById('csv-file');
const fileLabel = document.getElementById('file-label');
const submitBtn = document.getElementById('submit-btn');
const resultsBody = document.getElementById('results-body');
const resultsWrap = document.getElementById('results-table-wrap');
const emptyState = document.getElementById('results-state');
const totalCustomers = document.getElementById('total-customers');
const highestRisk = document.getElementById('highest-risk');
const bestModelStat = document.getElementById('best-model');
const priorityNote = document.getElementById('priority-note');
const bestModelPill = document.getElementById('best-model-pill');
const modelCards = document.getElementById('model-cards');

// Modal elements
const modalOverlay = document.getElementById('modal-overlay');
const modalClose = document.getElementById('modal-close');
const modalTitle = document.getElementById('modal-title');
const modalBadge = document.getElementById('modal-badge');
const modalSummary = document.getElementById('modal-summary');
const modalSuggestions = document.getElementById('modal-suggestions');
const modalFeatures = document.getElementById('modal-features');

let currentPredictions = [];

// ── File label ──────────────────────────────────────────────────────────────

fileInput.addEventListener('change', () => {
  fileLabel.textContent = fileInput.files.length ? fileInput.files[0].name : 'Choose CSV file';
});

// ── Nav active state ─────────────────────────────────────────────────────────

document.querySelectorAll('.nav-link').forEach((link) => {
  link.addEventListener('click', () => {
    document.querySelectorAll('.nav-link').forEach((l) => l.classList.remove('active'));
    link.classList.add('active');
  });
});

// ── Helpers ──────────────────────────────────────────────────────────────────

function badgeClass(level) {
  const map = { Critical: 'badge-critical', High: 'badge-high', Moderate: 'badge-medium', Low: 'badge-low' };
  return `badge ${map[level] || 'badge-low'}`;
}

// ── Render results table ──────────────────────────────────────────────────────

function renderPredictions(data) {
  currentPredictions = data.predictions || [];
  totalCustomers.textContent = String(data.totalRows || 0);

  const highest = currentPredictions[0];
  highestRisk.textContent = highest ? `${Math.round(highest.riskScore * 100)}%` : '—';
  const modelName = data.bestModel || 'XGBoost';
  bestModelStat.textContent = modelName;
  bestModelPill.textContent = modelName;
  priorityNote.textContent = highest
    ? `${highest.customerId} is the top-priority account`
    : 'Waiting for upload';

  resultsBody.innerHTML = '';

  if (!currentPredictions.length) {
    resultsWrap.classList.add('hidden');
    emptyState.textContent = 'No results yet — upload a CSV file to begin.';
    emptyState.classList.remove('hidden');
    return;
  }

  emptyState.classList.add('hidden');
  resultsWrap.classList.remove('hidden');

  currentPredictions.forEach((item, index) => {
    const row = document.createElement('tr');
    row.innerHTML = `
      <td>${index + 1}</td>
      <td>${item.customerId}</td>
      <td>${Math.round(item.riskScore * 100)}%</td>
      <td><span class="${badgeClass(item.riskLevel)}">${item.riskLevel}</span></td>
    `;
    row.addEventListener('click', () => {
      highlightRow(index);
      openModal(item);
    });
    resultsBody.appendChild(row);
  });
}

function highlightRow(index) {
  resultsBody.querySelectorAll('tr').forEach((r, i) => r.classList.toggle('active-row', i === index));
}

// ── Modal ────────────────────────────────────────────────────────────────────

function openModal(item) {
  modalTitle.textContent = `Customer ${item.customerId} · ${Math.round(item.riskScore * 100)}% risk`;
  modalBadge.className = badgeClass(item.riskLevel);
  modalBadge.textContent = item.riskLevel;

  // Summary bullets
  modalSummary.innerHTML = (item.summary || []).map((s) => `<p class="modal-bullet">${s}</p>`).join('');

  // Suggestions
  modalSuggestions.innerHTML = (item.suggestions || []).map((t) => `<li>${t}</li>`).join('');

  // Feature impact bars
  const features = item.topFeatures || [];
  const maxAbs = Math.max(...features.map((f) => Math.abs(f.impact)), 0.0001);
  modalFeatures.innerHTML = features.map((f) => {
    const pct = Math.round((Math.abs(f.impact) / maxAbs) * 100);
    const isPositive = f.impact > 0;
    const label = f.feature.replace(/_/g, ' ');
    const sign = isPositive ? '+' : '−';
    return `
      <div class="feat-row">
        <span class="feat-name">${label}</span>
        <div class="feat-bar-wrap">
          <div class="feat-bar ${isPositive ? 'feat-bar-pos' : 'feat-bar-neg'}" style="width:${pct}%"></div>
        </div>
        <span class="feat-val ${isPositive ? 'impact-pos' : 'impact-neg'}">${sign}${Math.abs(f.impact).toFixed(3)}</span>
      </div>
    `;
  }).join('');

  modalOverlay.classList.remove('hidden');
  document.body.classList.add('modal-open');
}

function closeModal() {
  modalOverlay.classList.add('hidden');
  document.body.classList.remove('modal-open');
}

modalClose.addEventListener('click', closeModal);
modalOverlay.addEventListener('click', (e) => { if (e.target === modalOverlay) closeModal(); });
document.addEventListener('keydown', (e) => { if (e.key === 'Escape') closeModal(); });

// ── Upload ────────────────────────────────────────────────────────────────────

uploadForm.addEventListener('submit', async (event) => {
  event.preventDefault();
  if (!fileInput.files.length) return;

  submitBtn.disabled = true;
  submitBtn.textContent = 'Analysing…';
  emptyState.textContent = 'Running risk analysis — this may take a few seconds…';
  emptyState.classList.remove('hidden');
  resultsWrap.classList.add('hidden');

  const formData = new FormData();
  formData.append('file', fileInput.files[0]);

  try {
    const response = await fetch('/api/predict', { method: 'POST', body: formData });
    const data = await response.json();
    if (!response.ok) throw new Error(data.error || 'Prediction failed.');
    renderPredictions(data);
  } catch (error) {
    emptyState.textContent = `Error: ${error.message}`;
    emptyState.classList.remove('hidden');
    resultsWrap.classList.add('hidden');
  } finally {
    submitBtn.disabled = false;
    submitBtn.textContent = 'Run risk analysis';
  }
});

// ── Model comparison ──────────────────────────────────────────────────────────

async function loadModelScores() {
  try {
    const response = await fetch('/api/model-scores');
    const data = await response.json();

    const modelNames = ['XGBoost', 'LightGBM', 'Random Forest', 'Decision Tree'];
    const bestModelName = data.best_model || 'XGBoost';

    // Update sidebar pill
    if (bestModelPill.textContent === '—') bestModelPill.textContent = bestModelName;
    if (bestModelStat.textContent === '—') bestModelStat.textContent = bestModelName;

    modelCards.innerHTML = modelNames.map((name) => {
      const score = data[name] ?? 0;
      const metrics = data.metrics?.[name] || {};
      const isBest = name === bestModelName && score > 0;
      const trained = score > 0;

      const scoreDisplay = trained
        ? `<span class="score-pill">Acc: ${(score * 100).toFixed(1)}%</span>`
        : `<span class="score-pill score-pending">Not yet trained</span>`;

      const metricsDisplay = trained ? `
        <div class="metric-row">
          <span class="metric-item"><span class="metric-label">F1</span> ${(metrics.f1 * 100).toFixed(1)}%</span>
          <span class="metric-item"><span class="metric-label">Precision</span> ${(metrics.precision * 100).toFixed(1)}%</span>
          <span class="metric-item"><span class="metric-label">Recall</span> ${(metrics.recall * 100).toFixed(1)}%</span>
        </div>
      ` : '';

      return `
        <article class="model-card ${isBest ? 'best' : ''}">
          <p class="eyebrow">${isBest ? 'Leading model' : 'Benchmark'}</p>
          <h4>${name}</h4>
          ${scoreDisplay}
          ${metricsDisplay}
        </article>
      `;
    }).join('');
  } catch {
    modelCards.innerHTML = '<p class="small-print">Model comparison data is unavailable right now.</p>';
  }
}

loadModelScores();
