let allAreas = [];
let scoreChart = null;
let featuresChart = null;

async function loadAreas() {
    const res = await fetch('/api/areas');
    if (!res.ok) {
        showError('Failed to load districts. Please try again.');
        return;
    }
    allAreas = await res.json();
    populateSelects();
    document.getElementById('compare-loader').style.display = 'none';
    document.getElementById('controls-content').style.display = 'block';
}

function showError(msg) {
    document.getElementById('error-msg').textContent = msg;
}

function populateSelects() {
    const selects = document.querySelectorAll('.district-select');
    selects.forEach(select => {
        if (select.options.length > 1) return;
        allAreas.forEach(a => {
            const opt = document.createElement('option');
            opt.value = a.area_id;
            opt.textContent = `${a.area_name} (${a.state_name})`;
            select.appendChild(opt);
        });
    });
}

document.getElementById('add-district').addEventListener('click', () => {
    const container = document.getElementById('selectors-container');
    const count = container.children.length;
    if (count >= 5) {
        showError("Maximum 5 districts allowed.");
        return;
    }
    const newSelect = document.createElement('select');
    newSelect.className = 'district-select';
    newSelect.innerHTML = `<option value="">Select District ${count + 1}</option>`;
    container.appendChild(newSelect);
    populateSelects();
    showError("");
});

document.getElementById('compare-btn').addEventListener('click', async () => {
    const selects = Array.from(document.querySelectorAll('.district-select'));
    const area_ids = selects.map(s => parseInt(s.value)).filter(v => !isNaN(v));

    if (area_ids.length < 2) {
        showError("Please select at least 2 districts.");
        return;
    }
    showError("");

    const btn = document.getElementById('compare-btn');
    btn.disabled = true;
    btn.textContent = 'Loading...';

    try {
        const res = await fetch('/api/compare', {
            method: 'POST',
            headers: { 'Content-Type': 'application/json' },
            body: JSON.stringify({ area_ids })
        });
        const data = await res.json();
        if (data.error) {
            showError(data.error);
            return;
        }

        renderResults(data);

    } catch (e) {
        showError("Error fetching comparison data. Is the server running?");
    } finally {
        btn.disabled = false;
        btn.textContent = 'Compare';
    }
});

function renderResults(data) {
    document.getElementById('results-container').style.display = 'block';

    const thead = document.getElementById('table-head');
    const tbody = document.getElementById('table-body');

    thead.innerHTML = '<th>Metric</th>' + data.map(d => `<th>${d.area_name}</th>`).join('');

    let tableHtml = `<tr><td><strong>Score</strong></td>${data.map(d => `<td>${d.predicted_score ? d.predicted_score.toFixed(3) : 'N/A'}</td>`).join('')}</tr>`;
    tableHtml += `<tr><td><strong>Category</strong></td>${data.map(d => `<td>${d.predicted_category}</td>`).join('')}</tr>`;

    let fKeys = new Set();
    data.forEach(d => Object.keys(d.features).forEach(k => {
        if (k !== 'area_id' && k !== 'data_mode') fKeys.add(k);
    }));

    fKeys.forEach(k => {
        tableHtml += `<tr><td><strong>${formatFeatureName(k)}</strong></td>${data.map(d => `<td>${d.features[k] ? parseFloat(d.features[k]).toFixed(2) : 'N/A'}</td>`).join('')}</tr>`;
    });

    tbody.innerHTML = tableHtml;

    const names = data.map(d => d.area_name);
    const scores = data.map(d => d.predicted_score || 0);

    if (scoreChart) scoreChart.destroy();
    scoreChart = new Chart(document.getElementById('scoreChart'), {
        type: 'bar',
        data: {
            labels: names,
            datasets: [{
                label: 'Deprivation Score',
                data: scores,
                backgroundColor: 'rgba(52, 152, 219, 0.8)'
            }]
        }
    });

    const colors = ['#e74c3c', '#2ecc71', '#f1c40f', '#9b59b6', '#34495e'];
    const featDatasets = Array.from(fKeys).map((key, i) => {
        return {
            label: formatFeatureName(key),
            data: data.map(d => d.features[key] || 0),
            backgroundColor: colors[i % colors.length]
        };
    });

    if (featuresChart) featuresChart.destroy();
    featuresChart = new Chart(document.getElementById('featuresChart'), {
        type: 'bar',
        data: {
            labels: names,
            datasets: featDatasets
        }
    });

    document.getElementById('results-container').scrollIntoView({ behavior: 'smooth' });
}

window.onload = loadAreas;
