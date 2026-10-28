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
    if (key === 'builtup_pct') document.getElementById('badge_builtup').textContent = val.toFixed(1);
    if (key === 'population_density') document.getElementById('badge_pop').textContent = Math.round(val) + ' /km²';
    if (key === 'hospital_distance_km') document.getElementById('badge_hospital').textContent = val.toFixed(2) + ' km';
    if (key === 'road_density_km_per_sqkm') document.getElementById('badge_road').textContent = val.toFixed(2) + ' km/km²';
    if (key === 'school_distance_km') document.getElementById('badge_school').textContent = val.toFixed(2) + ' km';
}

function syncTextureVal(val) {
    val = parseFloat(val);
    document.getElementById('num_visual_texture').value = val.toFixed(2);
    document.getElementById('badge_texture').textContent = val.toFixed(2);
}

function syncTextureSlider(val) {
    val = parseFloat(val);
    document.getElementById('slider_visual_texture').value = val.toFixed(2);
    document.getElementById('badge_texture').textContent = val.toFixed(2);
}

// --- Preset Profiles ---
const PRESETS = {
    rural: {
        nightlight_mean: 0.8, ndvi_mean: 0.58, builtup_pct: 75.0, population_density: 180.0,
        hospital_distance_km: 14.5, road_density_km_per_sqkm: 0.85, school_distance_km: 4.8, visual_texture: 0.18
    },
    semiurban: {
        nightlight_mean: 3.2, ndvi_mean: 0.38, builtup_pct: 220.0, population_density: 650.0,
        hospital_distance_km: 4.5, road_density_km_per_sqkm: 2.10, school_distance_km: 1.8, visual_texture: 0.35
    },
    metro: {
        nightlight_mean: 11.5, ndvi_mean: 0.18, builtup_pct: 490.0, population_density: 1850.0,
        hospital_distance_km: 1.8, road_density_km_per_sqkm: 3.90, school_distance_km: 0.8, visual_texture: 0.52
    },
    arid: {
        nightlight_mean: 1.1, ndvi_mean: 0.12, builtup_pct: 60.0, population_density: 90.0,
        hospital_distance_km: 18.0, road_density_km_per_sqkm: 0.60, school_distance_km: 6.2, visual_texture: 0.14
    }
};

function applyPreset(name) {
    const p = PRESETS[name];
    if (!p) return;
    for (let k in p) {
        if (k === 'visual_texture') {
            syncTextureVal(p[k]);
            document.getElementById('slider_visual_texture').value = p[k];
        } else {
            syncVal(k, p[k]);
            document.getElementById(`slider_${k}`).value = p[k];
        }
    }
    runLivePrediction();
}

// --- Live Prediction API Call ---
async function runLivePrediction() {
    const texVal = parseFloat(document.getElementById('num_visual_texture')?.value) || 0.28;
    const payload = {
        nightlight_mean: parseFloat(document.getElementById('num_nightlight_mean')?.value) || 0.0,
        ndvi_mean: parseFloat(document.getElementById('num_ndvi_mean')?.value) || 0.0,
        builtup_pct: parseFloat(document.getElementById('num_builtup_pct')?.value) || 0.0,
        population_density: parseFloat(document.getElementById('num_population_density')?.value) || 0.0,
        hospital_distance_km: parseFloat(document.getElementById('num_hospital_distance_km')?.value) || 4.5,
        road_density_km_per_sqkm: parseFloat(document.getElementById('num_road_density_km_per_sqkm')?.value) || 2.1,
        school_distance_km: parseFloat(document.getElementById('num_school_distance_km')?.value) || 2.0,
    };
    // Map texture slider into vgg_feat_1..vgg_feat_16 vector:
    for (let i = 1; i <= 16; i++) {
        payload[`vgg_feat_${i}`] = parseFloat((texVal * (0.8 + 0.02 * i)).toFixed(4));
    }

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
        const [predRes, featRes] = await Promise.all([
            fetch(`/api/predict/${areaId}`).then(r => r.json()),
            fetch(`/api/area-features/${areaId}`).then(r => r.json())
        ]);

        selectedDistrictFeatures = featRes;

        document.getElementById('distNameText').textContent = predRes.area_name;
        document.getElementById('distStateText').textContent = predRes.state_name;
        document.getElementById('distScoreText').textContent = predRes.predicted_score.toFixed(4);

        const catBadge = document.getElementById('distCatBadge');
        catBadge.textContent = predRes.predicted_category;
        catBadge.className = `category-pill cat-${predRes.predicted_category.toLowerCase()}`;

        document.getElementById('distNtlText').textContent = (featRes.nightlight_mean ? featRes.nightlight_mean.toFixed(2) : 'N/A') + ' nW';
        document.getElementById('distNdviText').textContent = featRes.ndvi_mean ? featRes.ndvi_mean.toFixed(2) : 'N/A';
        document.getElementById('distBuiltupText').textContent = featRes.builtup_pct ? featRes.builtup_pct.toFixed(1) : 'N/A';
        document.getElementById('distHospText').textContent = featRes.hospital_distance_km ? featRes.hospital_distance_km.toFixed(2) + ' km' : 'N/A';
        document.getElementById('distRoadText').textContent = featRes.road_density_km_per_sqkm ? featRes.road_density_km_per_sqkm.toFixed(2) + ' km/km²' : 'N/A';
        document.getElementById('distSchoolText').textContent = featRes.school_distance_km ? featRes.school_distance_km.toFixed(2) + ' km' : 'N/A';
        document.getElementById('distTexText').textContent = featRes.vgg_feat_1 ? featRes.vgg_feat_1.toFixed(2) : 'N/A';

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
        }
    }
    if (selectedDistrictFeatures.vgg_feat_1) {
        syncTextureVal(selectedDistrictFeatures.vgg_feat_1);
    }
    switchTab('custom');
    runLivePrediction();
}

// Auto-run initial prediction with default values
window.addEventListener('DOMContentLoaded', () => {
    runLivePrediction();
});
