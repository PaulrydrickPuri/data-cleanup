/**
 * Configuration constants for the collision detection system
 * Centralizes API endpoints and default parameters
 */

// Define base API URL depending on environment
const API_BASE = "http://localhost:5000";

// API endpoints
export const API_UPLOAD = `${API_BASE}/api/upload`;
export const API_STATUS = jobId => `${API_BASE}/api/status/${jobId}`;
export const API_RESULTS = jobId => `${API_BASE}/api/results/${jobId}`;
export const API_RESULTS_JSON = jobId => `${API_BASE}/api/results/${jobId}/results_${jobId}.json`;

// Default parameters for collision detection
export const PARAM_DEFAULTS = {
  collision_threshold: 0.2,
  motion_threshold: 20,
  yellow_flag_ratio: 0.7
};

// File upload limits
export const UPLOAD_LIMITS = {
  maxFilesize: 500, // MB
  acceptedFiles: "video/*"
};
