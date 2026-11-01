import sys
import os
import numpy as np
from sklearn.cluster import DBSCAN

sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
from backend.services import db_service
from config import HOTSPOT_EPS, HOTSPOT_MIN_SAMPLES


def identify_hotspots():
    """Identifies geospatial poverty clusters using DBSCAN with haversine metric on radian coordinates.
    Filters districts whose predicted category is 'High'."""
    areas = db_service.get_all_areas_with_predictions()
    
    # Select districts whose predicted category is 'High' instead of the fixed score 0.5
    deprived = [
        a for a in areas 
        if a.get('predicted_category') == 'High' 
        and a.get('centroid_lat') is not None 
        and a.get('centroid_lon') is not None
    ]
    
    if not deprived or len(deprived) < HOTSPOT_MIN_SAMPLES:
        return []
        
    # Extract coordinates and convert to radians: [lat, lon]
    coords = np.array([[a['centroid_lat'], a['centroid_lon']] for a in deprived])
    coords_rad = np.radians(coords)
    
    # DBSCAN with haversine metric and constants from config.py
    clustering = DBSCAN(eps=HOTSPOT_EPS, min_samples=HOTSPOT_MIN_SAMPLES, metric="haversine").fit(coords_rad)
    
    clusters = {}
    for i, label in enumerate(clustering.labels_):
        if label == -1:
            continue  # noise point
        if label not in clusters:
            clusters[label] = {
                'cluster_id': int(label),
                'districts': [],
                'avg_score': 0.0,
                'states': set()
            }
        clusters[label]['districts'].append({
            'area_id': deprived[i]['area_id'],
            'area_name': deprived[i]['area_name'],
            'state_name': deprived[i]['state_name'],
            'predicted_score': deprived[i]['predicted_score'],
            'predicted_category': deprived[i]['predicted_category'],
            'centroid_lat': deprived[i]['centroid_lat'],
            'centroid_lon': deprived[i]['centroid_lon']
        })
        if deprived[i].get('state_name'):
            clusters[label]['states'].add(deprived[i]['state_name'])
        
    result = []
    for c in clusters.values():
        scores = [d['predicted_score'] for d in c['districts'] if d['predicted_score'] is not None]
        c['avg_score'] = sum(scores) / len(scores) if scores else 0.0
        
        lats = [d['centroid_lat'] for d in c['districts']]
        lons = [d['centroid_lon'] for d in c['districts']]
        c['center_lat'] = sum(lats) / len(lats)
        c['center_lon'] = sum(lons) / len(lons)
        c['states'] = sorted(list(c['states']))
        
        result.append(c)
        
    # Sort clusters by severity (avg_score descending)
    result.sort(key=lambda x: x['avg_score'], reverse=True)
    return result
