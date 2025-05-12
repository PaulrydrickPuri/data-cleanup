/**
 * Main application entry point
 * Orchestrates all modules and initializes the application
 */
import { initUploader, resetUploader } from './uploader.js';
import { exportCSV, exportPDFReport } from './export.js';

// Global job ID for the current processing job
let currentJobId = null;

/**
 * Initialize the application when DOM is loaded
 */
// Demo mode - set to true to run without backend connection
const DEMO_MODE = true;

document.addEventListener('DOMContentLoaded', () => {
  // Show a temporary demo message if demo mode is enabled
  if (DEMO_MODE) {
    showDemoMessage();
  }

  // Initialize the uploader
  const dropzoneElement = document.getElementById('dz');
  if (dropzoneElement) {
    if (DEMO_MODE) {
      // In demo mode, we set up a simplified dropzone that doesn't try to upload
      setupDemoDropzone(dropzoneElement);
    } else {
      // Normal mode with backend connection
      initUploader(dropzoneElement);
    }
  }
  
  // Set up event listeners for export buttons
  setupExportButtons();
  
  // Set up reset button
  setupResetButton();
});

/**
 * Set up event listeners for export buttons
 */
function setupExportButtons() {
  // CSV Export button
  const csvButton = document.getElementById('btn-csv');
  if (csvButton) {
    csvButton.addEventListener('click', () => {
      exportCSV(currentJobId);
    });
  }
  
  // PDF Export button
  const pdfButton = document.getElementById('btn-pdf');
  if (pdfButton) {
    pdfButton.addEventListener('click', () => {
      exportPDFReport(currentJobId);
    });
  }
}

/**
 * Set up event listener for reset button
 */
function setupResetButton() {
  const resetButton = document.getElementById('btn-reset');
  if (resetButton) {
    resetButton.addEventListener('click', () => {
      // Reset uploader
      resetUploader();
      
      // Hide results and status sections
      document.getElementById('results').hidden = true;
      document.getElementById('status').hidden = true;
      
      // Clear job ID
      currentJobId = null;
    });
  }
}

/**
 * Update the current job ID
 * This function is called from the poller module
 * @param {string} jobId - The new job ID
 */
export function setCurrentJobId(jobId) {
  currentJobId = jobId;
  console.log('Set current job ID:', currentJobId);
}

/**
 * Shows a demo message at the top of the page
 */
function showDemoMessage() {
  const demoMessage = document.createElement('div');
  demoMessage.className = 'demo-banner';
  demoMessage.innerHTML = `
    <strong>Demo Mode Active:</strong> 
    This is a preview of the new modular frontend architecture.
    <button id="show-demo-results" class="btn">Show Demo Results</button>
  `;
  
  // Insert at the top of the container
  const container = document.querySelector('#app');
  container.insertBefore(demoMessage, container.firstChild);
  
  // Add click handler for the demo button
  document.getElementById('show-demo-results').addEventListener('click', showDemoResults);
  
  // Add styles for the demo banner
  const style = document.createElement('style');
  style.textContent = `
    .demo-banner {
      background-color: #f8f9fa;
      border: 1px solid #dee2e6;
      border-radius: 4px;
      padding: 10px 15px;
      margin-bottom: 20px;
      display: flex;
      justify-content: space-between;
      align-items: center;
    }
  `;
  document.head.appendChild(style);
}

/**
 * Sets up a demo dropzone that doesn't try to connect to a backend
 * @param {HTMLElement} dropzoneElement - The DOM element for Dropzone
 */
function setupDemoDropzone(dropzoneElement) {
  // Create a minimal Dropzone instance with autoProcessQueue set to false
  const dropzone = new Dropzone(dropzoneElement, {
    url: "/demo-upload", // This URL won't be used
    autoProcessQueue: false,
    acceptedFiles: "video/*",
    maxFilesize: 500,
    createImageThumbnails: false,
    dictDefaultMessage: "<i class='fa fa-cloud-upload'></i> Drag & drop videos here or click to upload (Demo Mode)"
  });
  
  // When files are added, show the demo processing UI
  dropzone.on("addedfile", function(file) {
    console.log("File added to demo dropzone:", file.name);
    
    // Show the status container
    document.getElementById("status").hidden = false;
    document.getElementById("status-title").textContent = "Processing Demo Video";
    document.getElementById("stage").textContent = "Demo Mode";
    
    // Simulate progress over 3 seconds
    let progress = 0;
    const progressBar = document.getElementById("progress");
    const progressText = document.getElementById("progress-text");
    
    const interval = setInterval(() => {
      progress += 5;
      progressBar.style.width = `${progress}%`;
      progressText.textContent = `${progress}%`;
      
      if (progress >= 100) {
        clearInterval(interval);
        // Show demo results after "processing"
        setTimeout(showDemoResults, 500);
      }
    }, 150);
  });
}

/**
 * Shows demo collision detection results
 */
function showDemoResults() {
  // Set a demo job ID
  currentJobId = "demo-" + Date.now();
  
  // Hide the status container and show results
  document.getElementById("status").hidden = true;
  const resultsContainer = document.getElementById("results");
  resultsContainer.hidden = false;
  
  // Create demo collision data
  const demoCollisions = [
    { timestamp: 3.2, person_id: 1, vehicle_id: 3, iou: 0.32, frame: 96 },
    { timestamp: 7.5, person_id: 2, vehicle_id: 1, iou: 0.51, frame: 225 },
    { timestamp: 12.8, person_id: 1, vehicle_id: 2, iou: 0.28, frame: 384 },
    { timestamp: 19.4, person_id: 3, vehicle_id: 3, iou: 0.45, frame: 582 },
    { timestamp: 26.0, person_id: 2, vehicle_id: 1, iou: 0.38, frame: 780 }
  ];
  
  // Import functions from renderer module
  import('./renderer.js').then(module => {
    // Populate the table with demo data
    const populateTable = document.createElement('script');
    populateTable.textContent = `
      // Create table structure
      const table = document.getElementById("collision-table");
      table.innerHTML = '';
      const thead = document.createElement('thead');
      thead.innerHTML = '<tr><th>Time</th><th>Objects</th><th>IoU Score</th><th>Severity</th></tr>';
      table.appendChild(thead);
      
      const tbody = document.createElement('tbody');
      ${demoCollisions.map((collision, i) => `
        // Create row for collision ${i}
        const row${i} = document.createElement('tr');
        
        // Time
        const time${i} = document.createElement('td');
        time${i}.textContent = "${Math.floor(collision.timestamp / 60)}:${(Math.floor(collision.timestamp % 60)).toString().padStart(2, '0')}";
        
        // Objects
        const objects${i} = document.createElement('td');
        objects${i}.textContent = "Person #${collision.person_id} & Vehicle #${collision.vehicle_id}";
        
        // IoU
        const iou${i} = document.createElement('td');
        iou${i}.textContent = "${collision.iou.toFixed(3)}";
        
        // Severity
        const severity${i} = document.createElement('td');
        severity${i}.textContent = "${collision.iou > 0.5 ? 'High' : (collision.iou > 0.3 ? 'Medium' : 'Low')}";
        severity${i}.className = "severity-${collision.iou > 0.5 ? 'high' : (collision.iou > 0.3 ? 'medium' : 'low')}";
        
        // Add cells to row
        row${i}.appendChild(time${i});
        row${i}.appendChild(objects${i});
        row${i}.appendChild(iou${i});
        row${i}.appendChild(severity${i});
        
        // Add row to table
        tbody.appendChild(row${i});
      `).join('')}
      
      table.appendChild(tbody);
    `;
    
    document.head.appendChild(populateTable);
    
    // Set a demo video source
    const video = document.getElementById('result-video');
    video.src = 'https://commondatastorage.googleapis.com/gtv-videos-bucket/sample/BigBuckBunny.mp4';
    video.controls = true;
    video.load();
  });
}
