/**
 * Handles status polling for job processing
 * Checks job status at regular intervals and updates UI
 */
import { API_STATUS } from "./config.js";
import { renderStatus, renderResults, renderError } from "./renderer.js";

let statusInterval = null;

/**
 * Starts polling for job status updates
 * @param {string} jobId - The ID of the processing job
 * @param {number} interval - Polling interval in milliseconds (default: 2000ms)
 */
export function startPolling(jobId, interval = 2000) {
  // Clear any existing interval
  if (statusInterval) {
    clearInterval(statusInterval);
  }
  
  // Start new polling interval
  statusInterval = setInterval(async () => {
    try {
      // Fetch status from API
      const response = await fetch(API_STATUS(jobId));
      if (!response.ok) {
        throw new Error(`Server responded with ${response.status}`);
      }
      
      const data = await response.json();
      console.log('Status data:', data);
      
      // Normalize response format
      const statusData = data.success && data.status ? data.status : data;
      statusData.job_id = jobId; // Ensure job_id is available
      
      // Update UI with the status data
      renderStatus(statusData);
      
      // Handle completion
      if (statusData.status === "completed") {
        clearInterval(statusInterval);
        renderResults(statusData);
      }
      // Handle error
      else if (statusData.status === "error" || statusData.status === "failed") {
        clearInterval(statusInterval);
        renderError(statusData.error || statusData.message || "Processing failed");
      }
    } catch (error) {
      console.error('Error checking status:', error);
      renderError(`Status check failed: ${error.message}`);
      clearInterval(statusInterval);
    }
  }, interval);
  
  return stopPolling; // Return function to stop polling
}

/**
 * Stops the status polling
 */
export function stopPolling() {
  if (statusInterval) {
    clearInterval(statusInterval);
    statusInterval = null;
  }
}
