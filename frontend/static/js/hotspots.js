let map;
let markersLayer;

async function loadHotspots() {
    const container = document.getElementById('clusters-container');

    map = L.map('hotspots-map').setView([22.5, 79.0], 5);
    L.tileLayer('https://{s}.tile.openstreetmap.org/{z}/{x}/{y}.png', {
        attribution: '&copy; OpenStreetMap contributors',
        maxZoom: 12,
    }).addTo(map);

    try {
        const res = await fetch('/api/hotspots');
        if (!res.ok) throw new Error(`HTTP ${res.status}`);
        const clusters = await res.json();

        // Hide loader, show content
        document.getElementById('hotspots-loader').style.display = 'none';
        document.getElementById('hotspots-map').style.display = 'block';
        setTimeout(() => map.invalidateSize(), 100);

        if (clusters.length === 0) {
            container.innerHTML = "<p>No significant deprivation hotspots detected. This may indicate the demo data is too sparse for DBSCAN clustering.</p>";
            return;
        }

        container.innerHTML = `<h3>Found ${clusters.length} Hotspot${clusters.length > 1 ? 's' : ''}</h3>`;
        markersLayer = L.layerGroup().addTo(map);

        const colors = ['#e74c3c', '#8e44ad', '#2980b9', '#f39c12', '#c0392b'];

        clusters.forEach((c, i) => {
            const color = colors[i % colors.length];

            const card = document.createElement('div');
            card.className = 'cluster-card';
            card.style.borderLeftColor = color;
            card.innerHTML = `
                <h3>Hotspot #${i+1}</h3>
                <p><strong>Avg Score:</strong> ${c.avg_score.toFixed(3)}</p>
                <p><strong>Districts (${c.districts.length}):</strong> ${c.districts.slice(0, 3).map(d => d.area_name).join(', ')}${c.districts.length > 3 ? '...' : ''}</p>
            `;

            card.addEventListener('click', () => {
                map.flyTo([c.center_lat, c.center_lon], 7);
            });

            container.appendChild(card);

            c.districts.forEach(d => {
                const marker = L.circleMarker([d.centroid_lat, d.centroid_lon], {
                    radius: 8,
                    fillColor: color,
                    color: '#fff',
                    weight: 1,
                    fillOpacity: 0.9
                }).addTo(markersLayer);

                marker.bindPopup(`
                    <b>${d.area_name}</b> (${d.state_name})<br>
                    Score: ${d.predicted_score.toFixed(3)}<br>
                    Category: ${d.predicted_category}<br>
                    <em>Hotspot #${i+1}</em>
                `);
            });
        });

    } catch (e) {
        document.getElementById('hotspots-loader').innerHTML = `<div style="color:#dc2626; font-weight:600;">Error loading hotspots: ${e.message}</div>`;
    }
}

window.onload = loadHotspots;
