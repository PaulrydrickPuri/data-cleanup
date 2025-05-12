/**
 * Handles DOM updates and UI rendering
 * Responsible for updating the UI based on application state
 */
import { API_RESULTS, API_RESULTS_JSON } from "./config.js";
import { setupBoundingBoxCanvas } from "./canvas.js";

// Store collision data for export
let collisionData = [];

/**
 * Shows the upload progress UI
 */
export function showUploadProgress() {
  // Show status container
  const statusContainer = document.getElementById("status");
  statusContainer.hidden = false;
  
  // Set initial status
  document.getElementById("status-title").textContent = "Uploading Video";
  document.getElementById("stage").textContent = "Upload";
  document.getElementById("progress").style.width = "0%";
  document.getElementById("progress-text").textContent = "0%";
  
  // Hide results if showing
  document.getElementById("results").hidden = true;
}

/**
 * Updates the UI based on processing status
 * @param {Object} statusData - The processing status data
 */
export function renderStatus(statusData) {
  // Update progress
  const progress = statusData.progress || 0;
  document.getElementById("progress").style.width = `${progress}%`;
  document.getElementById("progress-text").textContent = `${Math.round(progress)}%`;
  
  // Update stage
  if (statusData.stage) {
    document.getElementById("stage").textContent = statusData.stage;
  }
  
  // Update title based on status
  if (statusData.status === "processing") {
    document.getElementById("status-title").textContent = "Processing Video";
  } else if (statusData.status === "completed") {
    document.getElementById("status-title").textContent = "Processing Complete";
  } else if (statusData.status === "error" || statusData.status === "failed") {
    document.getElementById("status-title").textContent = "Processing Error";
  }
}

/**
 * Renders the processing results
 * @param {Object} resultData - The processing result data
 */
export function renderResults(resultData) {
  console.log('Showing results with data:', resultData);
  
  // Show results container
  const resultsContainer = document.getElementById("results");
  resultsContainer.hidden = false;
  
  // Load result video
  loadResultVideo(resultData);
  
  // Load and parse result data
  loadResultData(resultData);
}

/**
 * Loads the processed video
 * @param {Object} resultData - The processing result data
 */
function loadResultVideo(resultData) {
  let videoPath = null;
  
  // Try to get video path from different possible sources
  if (resultData.processed_videos && resultData.processed_videos.length > 0) {
    videoPath = resultData.processed_videos[0].url;
  } else if (resultData.result_path) {
    videoPath = resultData.result_path;
  } else {
    // Fallback to a constructed path based on job ID
    videoPath = API_RESULTS(resultData.job_id) + `/processed_${resultData.job_id}.mp4`;
  }
  
  console.log('Setting video source to:', videoPath);
  
  // Set video source
  const videoElement = document.getElementById("result-video");
  videoElement.src = videoPath;
  videoElement.load();
  videoElement.controls = true;
}

/**
 * Loads the result data JSON and populates the UI
 * @param {Object} resultData - The processing result data
 */
function loadResultData(resultData) {
  fetch(API_RESULTS_JSON(resultData.job_id))
    .then(response => {
      if (!response.ok) {
        throw new Error('Could not load results file');
      }
      return response.json();
    })
    .then(resultsJson => {
      console.log('Loaded results JSON:', resultsJson);
      
      // Extract collision events
      if (resultsJson.collision_events) {
        // Store for export
        collisionData = resultsJson.collision_events;
        
        // Populate table
        populateCollisionTable(resultsJson.collision_events);
        
        // Setup canvas overlays if bounding boxes are available
        if (resultsJson.bounding_boxes) {
          setupBoundingBoxCanvas(
            document.getElementById("result-video"),
            document.getElementById("bbox-overlay"),
            resultsJson.bounding_boxes
          );
        }
      }
    })
    .catch(error => {
      console.error('Error loading results JSON:', error);
      // Use whatever data is available in the status response
      if (resultData.collisions) {
        collisionData = resultData.collisions;
        populateCollisionTable(resultData.collisions);
      }
    });
}

/**
 * Populates the collision data table
 * @param {Array} collisions - The collision event data
 */
function populateCollisionTable(collisions) {
  const table = document.getElementById("collision-table");
  
  // Clear existing table content
  table.innerHTML = '';
  
  // Create table header
  const thead = document.createElement('thead');
  thead.innerHTML = `
    <tr>
      <th>Time</th>
      <th>Objects</th>
      <th>IoU Score</th>
      <th>Severity</th>
    </tr>
  `;
  table.appendChild(thead);
  
  // Create table body
  const tbody = document.createElement('tbody');
  
  if (!collisions || !collisions.length) {
    const row = document.createElement('tr');
    row.innerHTML = '<td colspan="4">No collision data available</td>';
    tbody.appendChild(row);
  } else {
    // Sort collisions by timestamp
    collisions.sort((a, b) => (a.timestamp || a.frame || 0) - (b.timestamp || b.frame || 0));
    
    // Add each collision to the table
    collisions.forEach((collision, index) => {
      const row = document.createElement('tr');
      
      // Time cell
      const timeCell = formatTimeCell(collision, index);
      
      // Objects cell
      const objectsCell = formatObjectsCell(collision, index);
      
      // IoU Score cell
      const iouCell = formatIouCell(collision);
      
      // Severity cell
      const severityCell = formatSeverityCell(collision);
      
      // Add cells to row
      row.appendChild(timeCell);
      row.appendChild(objectsCell);
      row.appendChild(iouCell);
      row.appendChild(severityCell);
      
      // Add row to table
      tbody.appendChild(row);
    });
  }
  
  table.appendChild(tbody);
}

/**
 * Formats the time cell for a collision
 * @param {Object} collision - The collision data
 * @param {number} index - The collision index
 * @returns {HTMLTableCellElement} The formatted time cell
 */
function formatTimeCell(collision, index) {
  const cell = document.createElement('td');
  
  if (collision.timestamp) {
    const minutes = Math.floor(collision.timestamp / 60);
    const seconds = Math.floor(collision.timestamp % 60);
    cell.textContent = `${minutes.toString().padStart(2, '0')}:${seconds.toString().padStart(2, '0')}`;
  } else if (collision.frame) {
    cell.textContent = `Frame ${collision.frame}`;
  } else {
    cell.textContent = `Event ${index + 1}`;
  }
  
  return cell;
}

/**
 * Formats the objects cell for a collision
 * @param {Object} collision - The collision data
 * @param {number} index - The collision index
 * @returns {HTMLTableCellElement} The formatted objects cell
 */
function formatObjectsCell(collision, index) {
  const cell = document.createElement('td');
  
  if (collision.person_id && collision.vehicle_id) {
    cell.textContent = `Person #${collision.person_id} & Vehicle #${collision.vehicle_id}`;
  } else if (collision.objects) {
    cell.textContent = collision.objects;
  } else {
    cell.textContent = `Collision #${index + 1}`;
  }
  
  return cell;
}

/**
 * Formats the IoU score cell for a collision
 * @param {Object} collision - The collision data
 * @returns {HTMLTableCellElement} The formatted IoU cell
 */
function formatIouCell(collision) {
  const cell = document.createElement('td');
  
  if (collision.iou !== undefined) {
    cell.textContent = collision.iou.toFixed(3);
  } else if (collision.score !== undefined) {
    cell.textContent = collision.score.toFixed(3);
  } else {
    cell.textContent = 'N/A';
  }
  
  return cell;
}

/**
 * Formats the severity cell for a collision
 * @param {Object} collision - The collision data
 * @returns {HTMLTableCellElement} The formatted severity cell
 */
function formatSeverityCell(collision) {
  const cell = document.createElement('td');
  const iouValue = collision.iou || collision.score || 0;
  
  if (iouValue > 0.5) {
    cell.textContent = 'High';
    cell.className = 'severity-high';
  } else if (iouValue > 0.3) {
    cell.textContent = 'Medium';
    cell.className = 'severity-medium';
  } else {
    cell.textContent = 'Low';
    cell.className = 'severity-low';
  }
  
  return cell;
}

/**
 * Shows an error message
 * @param {string} message - The error message to display
 */
export function showError(message) {
  document.getElementById("status-title").textContent = "Error";
  document.getElementById("stage").textContent = message;
}

/**
 * Alias for showError for backward compatibility
 * @param {string} message - The error message to display
 */
export function renderError(message) {
  showError(message);
}

/**
 * Get the stored collision data
 * @returns {Array} The collision data array
 */
export function getCollisionData() {
  return collisionData;
}
