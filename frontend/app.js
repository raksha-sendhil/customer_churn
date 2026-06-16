const uploadForm = document.getElementById('upload-form');
const fileInput = document.getElementById('csv-file');
const fileLabel = document.getElementById('file-label');
const submitBtn = document.getElementById('submit-btn');
const resultsBody = document.getElementById('results-body');
const resultsWrap = document.getElementById('results-table-wrap');
const emptyState = document.getElementById('results-state');
const totalCustomers = document.getElementById('total-customers');
const highestRisk = document.getElementById('highest-risk');
const bestModel = document.getElementById('best-model');
const priorityNote = document.getElementById('priority-note');
const detailTitle = document.getElementById('detail-title');
const detailText = document.getElementById('detail-text');
const adviceList = document.getElementById('advice-list');
const modelCards = document.getElementById('model-cards');
const bestModelPill = document.getElementById('best-model-pill');

let currentPredictions = [];

// Update file label text when the user picks a file.
fileInput.addEventListener('change', () => {
  fileLabel.textContent = fileInput.files.length ? fileInput.files[0].name : 'Choose CSV file';
});

// Keep nav active state in sync with the clicked link.
document.querySelectorAll('.nav-link').forEach((link) => {
  link.addEventListener('click', () => {
    document.querySelectorAll('.nav-link').forEach((l) => l.classList.remove('active'));
    link.classList.add('active');
  });
});

function badgeClass(level) {
  if (level === 'Critical') return 'badge badge-critical';
  if (level === 'High') return 'badge badge-high';
  if (level === 'Moderate') return 'badge badge-medium';
  return 'badge badge-low';
}

function renderPredictions(data) {
  currentPredictions = data.predictions || [];
  totalCustomers.textContent = String(data.totalRows || 0);

  const highest = currentPredictions[0];
  highestRisk.textContent = highest ? `${Math.round(highest.riskScore * 100)}%` : '—';
  bestModel.textContent = data.bestModel || 'XGBoost';
  bestModelPill.textContent = data.bestModel || 'XGBoost';
  priorityNote.textContent = highest
    ? `${highest.customerId} is the highest-priority account`
    : 'Waiting for a file';

  resultsBody.innerHTML = '';
  if (!currentPredictions.length) {
    resultsWrap.classList.add('hidden');
    emptyState.textContent = 'No results yet. Upload a CSV file to begin.';
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
    row.addEventListener('click', () => showDetails(item, index));
    resultsBody.appendChild(row);
  });

  showDetails(currentPredictions[0], 0);
}

function showDetails(item, index) {
  const rows = resultsBody.querySelectorAll('tr');
  rows.forEach((row, i) => row.classList.toggle('active-row', i === index));

  detailTitle.textContent = `Customer ${item.customerId} · ${Math.round(item.riskScore * 100)}% risk`;

  const bulletList = item.summary || [];
  detailText.innerHTML = bulletList.map((entry) => `<p>${entry}</p>`).join('');

  adviceList.innerHTML = (item.suggestions || []).map((tip) => `<li>${tip}</li>`).join('');
}

uploadForm.addEventListener('submit', async (event) => {
  event.preventDefault();
  if (!fileInput.files.length) return;

  // Loading state — disable the button while the request is in flight.
  submitBtn.disabled = true;
  submitBtn.textContent = 'Analysing…';
  emptyState.textContent = 'Running risk analysis — this may take a few seconds…';
  emptyState.classList.remove('hidden');
  resultsWrap.classList.add('hidden');

  const formData = new FormData();
  formData.append('file', fileInput.files[0]);

  try {
    const response = await fetch('/api/predict', {
      method: 'POST',
      body: formData,
    });
    const data = await response.json();
    if (!response.ok) {
      throw new Error(data.error || 'Prediction failed.');
    }
    renderPredictions(data);
  } catch (error) {
    emptyState.textContent = `Error: ${error.message}`;
    emptyState.classList.remove('hidden');
    resultsWrap.classList.add('hidden');
  } finally {
    // Always restore the button regardless of success or failure.
    submitBtn.disabled = false;
    submitBtn.textContent = 'Run risk analysis';
  }
});

async function loadModelScores() {
  try {
    const response = await fetch('/api/model-scores');
    const data = await response.json();
    const models = [
      { name: 'XGBoost', description: 'Current model trained in the backend and ready for live use.', score: data.XGBoost ?? 0 },
      { name: 'Random Forest', description: 'Ready for your teammates to plug in their training score.', score: data['Random Forest'] ?? 0 },
      { name: 'LightGBM', description: 'Ready for your teammates to plug in their training score.', score: data.LightGBM ?? 0 },
      { name: 'Decision Tree', description: 'Ready for your teammates to plug in their training score.', score: data['Decision Tree'] ?? 0 },
    ];

    const bestModelName = data.best_model || 'XGBoost';
    modelCards.innerHTML = models
      .map((model) => {
        const isBest = model.name === bestModelName;
        const scoreDisplay = model.score > 0
          ? `Score: ${model.score.toFixed(4)}`
          : 'Not yet trained';
        return `
          <article class="model-card ${isBest ? 'best' : ''}">
            <p class="eyebrow">${isBest ? 'Leading model' : 'Benchmark'}</p>
            <h4>${model.name}</h4>
            <p>${model.description}</p>
            <p class="score-pill ${model.score === 0 ? 'score-pending' : ''}">${scoreDisplay}</p>
          </article>
        `;
      })
      .join('');
  } catch (error) {
    modelCards.innerHTML = '<p class="small-print">Model comparison data is unavailable right now.</p>';
  }
}

loadModelScores();
