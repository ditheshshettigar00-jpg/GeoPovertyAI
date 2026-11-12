// --- Synced Input Handlers ---
function syncVal(key, val) {
    val = parseFloat(val);
    document.getElementById(`num_${key}`).value = val;
    updateBadge(key, val);
}

function syncSlider(key, val) {
    val = parseFloat(val);
    document.getElementById(`slider_${key}`).value = val;
    updateBadge(key, val);
}

function updateBadge(key, val) {
    if (key === 'nightlight_mean') document.getElementById('badge_ntl').textContent = val.toFixed(2) + ' nW';
    if (key === 'ndvi_mean') document.getElementById('badge_ndvi').textContent = val.toFixed(2);
    if (key === 'builtup_pct') document.getElementById('badge_builtup').textContent = val.toFixed(1) + ' %';
    if (key === 'population_density') document.getElementById('badge_pop').textContent = Math.round(val) + ' /km²';
    if (key === 'lst_day_c') document.getElementById('badge_lst').textContent = val.toFixed(1) + ' °C';
    if (key === 'elevation_mean_m') document.getElementById('badge_elev').textContent = Math.round(val) + ' m';
    if (key === 'slope_mean_deg') document.getElementById('badge_slope').textContent = val.toFixed(1) + '°';
    if (key === 'rainfall_mm_year') document.getElementById('badge_rain').textContent = Math.round(val) + ' mm/yr';
    if (key === 'cropland_pct') document.getElementById('badge_crop').textContent = val.toFixed(1) + ' %';
    if (key === 'travel_time_city_min') document.getElementById('badge_travel').textContent = Math.round(val) + ' min';
}

// --- Preset Profiles (10 Real Satellite & Geospatial Model Inputs) ---
const PRESETS = {
    rural: {
        nightlight_mean: 0.45, ndvi_mean: 0.46, builtup_pct: 0.5, population_density: 210.0,
        lst_day_c: 31.5, elevation_mean_m: 320.0, slope_mean_deg: 4.5, rainfall_mm_year: 1150.0,
        cropland_pct: 62.0, travel_time_city_min: 75.0
    },
    semiurban: {
        nightlight_mean: 1.85, ndvi_mean: 0.41, builtup_pct: 2.8, population_density: 680.0,
        lst_day_c: 29.8, elevation_mean_m: 190.0, slope_mean_deg: 2.1, rainfall_mm_year: 980.0,
        cropland_pct: 55.0, travel_time_city_min: 25.0
    },
    metro: {
        nightlight_mean: 16.50, ndvi_mean: 0.22, builtup_pct: 24.0, population_density: 6500.0,
        lst_day_c: 32.0, elevation_mean_m: 85.0, slope_mean_deg: 1.4, rainfall_mm_year: 850.0,
        cropland_pct: 12.0, travel_time_city_min: 3.0
    },
    arid: {
        nightlight_mean: 0.70, ndvi_mean: 0.16, builtup_pct: 0.6, population_density: 95.0,
        lst_day_c: 36.2, elevation_mean_m: 260.0, slope_mean_deg: 2.2, rainfall_mm_year: 340.0,
        cropland_pct: 38.0, travel_time_city_min: 85.0
    }
};

function applyPreset(name) {
    const p = PRESETS[name];
    if (!p) return;
    for (const k in p) {
        syncVal(k, p[k]);
        const sliderEl = document.getElementById(`slider_${k}`);
        if (sliderEl) sliderEl.value = p[k];
    }
    runLivePrediction();
}

function readFeatureInput(id, fallback) {
    const parsed = parseFloat(document.getElementById(id)?.value);
    return Number.isFinite(parsed) ? parsed : fallback;
}

// --- Live Prediction API Call ---
async function runLivePrediction() {
    const payload = {
        nightlight_mean: readFeatureInput('num_nightlight_mean', 0.87),
        ndvi_mean: readFeatureInput('num_ndvi_mean', 0.43),
        builtup_pct: readFeatureInput('num_builtup_pct', 1.3),
        population_density: readFeatureInput('num_population_density', 425.0),
        lst_day_c: readFeatureInput('num_lst_day_c', 29.9),
        elevation_mean_m: readFeatureInput('num_elevation_mean_m', 236.0),
        slope_mean_deg: readFeatureInput('num_slope_mean_deg', 2.4),
        rainfall_mm_year: readFeatureInput('num_rainfall_mm_year', 1010.0),
        cropland_pct: readFeatureInput('num_cropland_pct', 54.3),
        travel_time_city_min: readFeatureInput('num_travel_time_city_min', 31.0),
    };

    const resPanel = document.getElementById('liveResultPanel');
    resPanel.style.display = 'block';
    document.getElementById('liveScoreText').textContent = 'Calculating...';

    try {
        const res = await fetch('/api/predict-live', {
            method: 'POST',
            headers: { 'Content-Type': 'application/json' },
            body: JSON.stringify(payload)
        });
        const data = await res.json();
        if (!res.ok) {
            alert(data.error || 'Prediction failed');
            return;
        }

        // Render Results
        document.getElementById('liveScoreText').textContent = data.predicted_score.toFixed(4);
        
        const catBadge = document.getElementById('liveCategoryBadge');
        catBadge.textContent = `${data.predicted_category} Deprivation`;
        catBadge.className = `category-pill cat-${data.predicted_category.toLowerCase()}`;

        if (data.model_name) {
            document.getElementById('topModelBadge').textContent = `⚡ Model: ${data.model_name}`;
        }

        // Gauge percentage
        const pct = Math.min(Math.max((data.predicted_score * 100), 5), 98);
        document.getElementById('liveGaugeFill').style.width = pct + '%';

        document.getElementById('liveBaseValText').textContent = data.base_value.toFixed(4);

        // Render SHAP Contributions
        const contribList = document.getElementById('liveContributionsList');
        contribList.innerHTML = '';

        if (data.contributions && data.contributions.length > 0) {
            const maxShap = Math.max(...data.contributions.map(c => Math.abs(c.shap)), 0.01);
            
            data.contributions.forEach(c => {
                const isReduce = c.shap <= 0;
                const barWidth = Math.min(Math.round((Math.abs(c.shap) / maxShap) * 100), 100);
                const sign = c.shap > 0 ? '+' : '';
                
                const item = document.createElement('div');
                item.className = 'contribution-item';
                item.innerHTML = `
                    <div class="contrib-label">${c.label} (${c.value})</div>
                    <div class="contrib-bar-wrap">
                        <div class="contrib-bar ${isReduce ? 'neg' : 'pos'}" style="width: ${barWidth}%;"></div>
                    </div>
                    <div class="contrib-val" style="color: ${isReduce ? '#059669' : '#dc2626'};">
                        ${sign}${c.shap.toFixed(4)}
                    </div>
                `;
                contribList.appendChild(item);
            });
        }

        document.getElementById('livePolicyCard').innerHTML = `
            <strong>💡 Socioeconomic Analysis:</strong> ${data.interpretation}
        `;

        resPanel.scrollIntoView({ behavior: 'smooth', block: 'nearest' });

    } catch (e) {
        alert('Failed to connect to backend: ' + e.message);
    }
}

// --- Tab Switching ---
function switchTab(tab) {
    const tabCustom = document.getElementById('tabCustom');
    const tabDistrict = document.getElementById('tabDistrict');
    const btnCustom = document.getElementById('tabBtnCustom');
    const btnDistrict = document.getElementById('tabBtnDistrict');

    if (tab === 'custom') {
        tabCustom.style.display = 'block';
        tabDistrict.style.display = 'none';
        btnCustom.classList.add('active');
        btnDistrict.classList.remove('active');
    } else {
        tabCustom.style.display = 'none';
        tabDistrict.style.display = 'block';
        btnDistrict.classList.add('active');
        btnCustom.classList.remove('active');
        if (!window.loadedDistricts) loadAllDistricts();
    }
}

// --- District Explorer Tab Logic ---
let districtsCache = [];
let selectedDistrictFeatures = null;

async function loadAllDistricts() {
    try {
        const res = await fetch('/api/areas');
        districtsCache = await res.json();
        const sel = document.getElementById('districtSelect');
        sel.innerHTML = '<option value="">-- Select from ' + districtsCache.length + ' Districts --</option>';
        districtsCache.forEach(d => {
            const opt = document.createElement('option');
            opt.value = d.area_id;
            opt.textContent = `${d.area_name}, ${d.state_name}`;
            sel.appendChild(opt);
        });
        window.loadedDistricts = true;
    } catch (e) {
        console.error(e);
    }
}

async function onDistrictChange(areaId) {
    if (!areaId) {
        document.getElementById('districtResultCard').style.display = 'none';
        return;
    }

    try {
        const [predRes, featRes, profRes] = await Promise.all([
            fetch(`/api/predict/${areaId}`).then(r => r.json()),
            fetch(`/api/area-features/${areaId}`).then(r => r.json()),
            fetch(`/api/area-profile/${areaId}`).then(r => r.ok ? r.json() : null).catch(() => null)
        ]);

        selectedDistrictFeatures = Object.assign({}, featRes);
        if (profRes && profRes.accessibility && profRes.accessibility.available) {
            profRes.accessibility.items.forEach(it => {
                selectedDistrictFeatures[it.key] = it.value;
            });
        }

        document.getElementById('distNameText').textContent = predRes.area_name;
        document.getElementById('distStateText').textContent = predRes.state_name;
        document.getElementById('distScoreText').textContent = predRes.predicted_score.toFixed(4);

        const catBadge = document.getElementById('distCatBadge');
        catBadge.textContent = predRes.predicted_category;
        catBadge.className = `category-pill cat-${predRes.predicted_category.toLowerCase()}`;

        // 10 Model Inputs
        document.getElementById('distNtlText').textContent = (featRes.nightlight_mean != null ? featRes.nightlight_mean.toFixed(2) : 'N/A') + ' nW';
        document.getElementById('distNdviText').textContent = featRes.ndvi_mean != null ? featRes.ndvi_mean.toFixed(2) : 'N/A';
        document.getElementById('distBuiltupText').textContent = (featRes.builtup_pct != null ? featRes.builtup_pct.toFixed(1) : 'N/A') + '%';
        if (document.getElementById('distPopText')) {
            document.getElementById('distPopText').textContent = featRes.population_density != null ? Math.round(featRes.population_density) + ' /km²' : 'N/A';
        }
        if (document.getElementById('distLstText')) {
            document.getElementById('distLstText').textContent = featRes.lst_day_c != null ? featRes.lst_day_c.toFixed(1) + ' °C' : 'N/A';
        }
        if (document.getElementById('distElevText')) {
            document.getElementById('distElevText').textContent = featRes.elevation_mean_m != null ? Math.round(featRes.elevation_mean_m) + ' m' : 'N/A';
        }
        if (document.getElementById('distSlopeText')) {
            document.getElementById('distSlopeText').textContent = featRes.slope_mean_deg != null ? featRes.slope_mean_deg.toFixed(1) + '°' : 'N/A';
        }
        if (document.getElementById('distRainText')) {
            document.getElementById('distRainText').textContent = featRes.rainfall_mm_year != null ? Math.round(featRes.rainfall_mm_year) + ' mm/yr' : 'N/A';
        }
        if (document.getElementById('distCropText')) {
            document.getElementById('distCropText').textContent = featRes.cropland_pct != null ? featRes.cropland_pct.toFixed(1) + '%' : 'N/A';
        }
        if (document.getElementById('distTravelText')) {
            document.getElementById('distTravelText').textContent = featRes.travel_time_city_min != null ? Math.round(featRes.travel_time_city_min) + ' min' : 'N/A';
        }

        // Satellite Image Preview
        const satContainer = document.getElementById('distSatContainer');
        if (predRes.satellite_available && predRes.satellite_url) {
            document.getElementById('distSatImg').src = predRes.satellite_url;
            document.getElementById('distSatLink').href = predRes.satellite_url;
            satContainer.style.display = 'flex';
        } else {
            satContainer.style.display = 'none';
        }

        document.getElementById('distFullExplainLink').href = `/explanation?area_id=${areaId}`;
        const pageLink = document.getElementById('distPageLink');
        if (pageLink) pageLink.href = `/district/${areaId}`;
        const rptLink = document.getElementById('distReportLink');
        if (rptLink) rptLink.href = `/report/${areaId}`;
        document.getElementById('districtResultCard').style.display = 'block';

    } catch (e) {
        alert('Failed to load district details: ' + e.message);
    }
}

function loadDistrictIntoCustom() {
    if (!selectedDistrictFeatures) return;
    for (let k in selectedDistrictFeatures) {
        if (document.getElementById(`num_${k}`)) {
            syncVal(k, selectedDistrictFeatures[k]);
            const sEl = document.getElementById(`slider_${k}`);
            if (sEl) sEl.value = selectedDistrictFeatures[k];
        }
    }
    switchTab('custom');
    runLivePrediction();
}

// Auto-run initial prediction with default values
window.addEventListener('DOMContentLoaded', () => {
    runLivePrediction();
});
