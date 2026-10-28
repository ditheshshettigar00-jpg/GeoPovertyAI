import sys
import os
import numpy as np
from sklearn.cluster import DBSCAN

sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
from backend.services import db_service

def identify_hotspots():
    areas = db_service.get_all_areas_with_predictions()
    
    # Identify deprived areas (score > 0.5 or Medium/High)
    deprived = [a for a in areas if a['predicted_score'] is not None and a['predicted_score'] >= 0.5]
    
    if not deprived:
        return []
        
    # Extract coordinates
    coords = np.array([[a['centroid_lat'], a['centroid_lon']] for a in deprived])
    
    # DBSCAN clustering
    # eps=2.5 degrees (roughly 250km), min_samples=3 areas
    clustering = DBSCAN(eps=2.5, min_samples=3).fit(coords)
    
    clusters = {}
    for i, label in enumerate(clustering.labels_):
        if label == -1:
            continue # noise
        if label not in clusters:
            clusters[label] = {
                'cluster_id': int(label),
                'districts': [],
                'avg_score': 0
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
        
    result = []
    for c in clusters.values():
        scores = [d['predicted_score'] for d in c['districts']]
        c['avg_score'] = sum(scores) / len(scores)
        
        lats = [d['centroid_lat'] for d in c['districts']]
        lons = [d['centroid_lon'] for d in c['districts']]
        c['center_lat'] = sum(lats) / len(lats)
        c['center_lon'] = sum(lons) / len(lons)
        
        result.append(c)
        
    # Sort clusters by severity (avg_score descending)
    result.sort(key=lambda x: x['avg_score'], reverse=True)
    return result
