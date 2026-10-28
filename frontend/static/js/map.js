const categoryColor = { 'Low': '#52b788', 'Medium': '#f4a261', 'High': '#e76f51' };
let allAreas = [];
let geojsonLayer;
let map;

async function loadMapData() {
    document.getElementById('map-loader').style.display = 'flex';
    document.getElementById('map').style.display = 'none';

    map = L.map('map').setView([22.5, 79.0], 5);
    L.tileLayer('https://{s}.tile.openstreetmap.org/{z}/{x}/{y}.png', {
        attribution: '&copy; OpenStreetMap contributors',
        maxZoom: 12,
    }).addTo(map);

    // Legend
    var legend = L.control({position: 'bottomright'});
    legend.onAdd = function (map) {
        var div = L.DomUtil.create('div', 'info legend');
        div.innerHTML += '<h4>Deprivation</h4>';
        div.innerHTML += '<i style="background:' + categoryColor['High'] + '"></i> High<br>';
        div.innerHTML += '<i style="background:' + categoryColor['Medium'] + '"></i> Medium<br>';
        div.innerHTML += '<i style="background:' + categoryColor['Low'] + '"></i> Low<br>';
        return div;
    };
    legend.addTo(map);

    try {
        const res = await fetch('/api/map');
        if (!res.ok) throw new Error("Failed to fetch map data");
        allAreas = await res.json();
        
        // Populate states dropdown
        const states = [...new Set(allAreas.map(a => a.state_name))].sort();
        const stateSelect = document.getElementById('filter-state');
        states.forEach(state => {
            const opt = document.createElement('option');
            opt.value = state;
            opt.textContent = state;
            stateSelect.appendChild(opt);
        });
        
        document.getElementById('map-loader').style.display = 'none';
        document.getElementById('map').style.display = 'block';

        // Force a resize event so leaflet renders properly after unhiding
        setTimeout(() => map.invalidateSize(), 100);
        
        renderMap(allAreas);
        
    } catch (e) {
        console.error("Failed to load map data", e);
        document.getElementById('map-loader').innerHTML = `<div style="color:red; font-weight:bold;">Error: ${e.message}</div>`;
    }
}

function renderMap(areas) {
    if (geojsonLayer) {
        map.removeLayer(geojsonLayer);
    }
    
    // We will build a FeatureCollection
    const features = areas.map(a => {
        let geom = null;
        if (a.boundary_geojson) {
            try { geom = JSON.parse(a.boundary_geojson); } catch(e) {}
        }
        
        // If no polygon, we create a Point feature
        if (!geom && a.centroid_lon != null && a.centroid_lat != null) {
            geom = { type: "Point", coordinates: [a.centroid_lon, a.centroid_lat] };
        }
        
        if (!geom) return null;
        
        return {
            type: "Feature",
            geometry: geom,
            properties: a
        };
    }).filter(f => f !== null);
    
    geojsonLayer = L.geoJSON(features, {
        style: function(feature) {
            const cat = feature.properties.predicted_category;
            const color = categoryColor[cat] || '#999';
            return {
                fillColor: color,
                weight: feature.geometry.type === "Point" ? 1 : 1,
                opacity: 1,
                color: 'white',
                fillOpacity: 0.8
            };
        },
        pointToLayer: function(feature, latlng) {
            return L.circleMarker(latlng, { radius: 8 });
        },
        onEachFeature: function(feature, layer) {
            const p = feature.properties;
            const scoreText = p.predicted_score != null ? p.predicted_score.toFixed(3) : 'N/A';
            const catText = p.predicted_category || 'N/A';
            
            // Hover effect
            layer.on('mouseover', function(e) {
                var popup = L.popup()
                    .setLatLng(e.latlng)
                    .setContent(`<b>${p.area_name}</b><br>${p.state_name}<br>Score: ${scoreText}`)
                    .openOn(map);
                if (layer.setStyle) layer.setStyle({ weight: 3, color: '#666' });
            });
            layer.on('mouseout', function(e) {
                map.closePopup();
                geojsonLayer.resetStyle(layer);
            });
            layer.on('click', function(e) {
                window.location.href = `/district/${p.area_id}`;
            });
        }
    }).addTo(map);
    
    if (features.length > 0) {
        map.fitBounds(geojsonLayer.getBounds());
    }
}

document.getElementById('apply-filters').addEventListener('click', () => {
    const search = document.getElementById('search-district').value.toLowerCase();
    const state = document.getElementById('filter-state').value;
    const category = document.getElementById('filter-category').value;
    
    const filtered = allAreas.filter(a => {
        const matchSearch = a.area_name.toLowerCase().includes(search);
        const matchState = state === 'all' || a.state_name === state;
        const matchCat = category === 'all' || a.predicted_category === category;
        return matchSearch && matchState && matchCat;
    });
    
    renderMap(filtered);
});

window.onload = loadMapData;
