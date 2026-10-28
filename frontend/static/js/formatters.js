/**
 * formatters.js
 * Centralized human-readable label mapping and formatting utilities
 * across the entire GeoPoverty AI web interface.
 */

const FEATURE_LABELS = {
    'nightlight_mean': 'Nighttime Lights Radiance',
    'ndvi_mean': 'Vegetation Index (NDVI)',
    'builtup_pct': 'Built-Up Settlement Density',
    'population_density': 'Population Density',
    'road_density_km_per_sqkm': 'Road Network Density',
    'school_distance_km': 'Distance to Nearest School',
    'hospital_distance_km': 'Distance to Nearest Hospital',
    's2_image_count': 'Cloud-Free Satellite Images',
    'deprivation_score': 'Deprivation Index Score',
    'predicted_score': 'Predicted Poverty Score',
    'predicted_category': 'Poverty Category',
    'model_name': 'Model Algorithm',
    'data_mode': 'Data Mode',
    'data_source': 'Data Source',
    'data_year': 'Data Year',
    'area_id': 'District ID',
    'area_name': 'District Name',
    'state_name': 'State',
    'centroid_lat': 'Latitude',
    'centroid_lon': 'Longitude',
    'latitude': 'Latitude',
    'longitude': 'Longitude'
};

function formatFeatureName(key) {
    if (!key) return '';
    if (FEATURE_LABELS[key]) return FEATURE_LABELS[key];
    // If key starts with vgg_feat_, format as VGG Visual Feature #
    if (key.startsWith('vgg_feat_')) {
        return 'Visual Texture Feature ' + key.replace('vgg_feat_', '#');
    }
    // General fallback: replace all underscores with spaces and capitalize each word
    return key
        .replace(/_/g, ' ')
        .replace(/\b\w/g, c => c.toUpperCase());
}
