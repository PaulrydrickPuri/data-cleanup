/**
 * Handles file uploads using the Dropzone library
 * Sets up the uploader and processes form submissions
 */
import { API_UPLOAD, PARAM_DEFAULTS, UPLOAD_LIMITS } from "./config.js";
import { showUploadProgress, showError } from "./renderer.js";
import { startPolling } from "./poller.js";

let dropzoneInstance = null;

/**
 * Initializes the Dropzone uploader
 * @param {HTMLElement} dropzoneEl - The DOM element for Dropzone
 * @returns {Dropzone} - The initialized Dropzone instance
 */
export function initUploader(dropzoneEl) {
  // Configure Dropzone
  dropzoneInstance = new Dropzone(dropzoneEl, {
    url: API_UPLOAD,
    acceptedFiles: UPLOAD_LIMITS.acceptedFiles,
    maxFilesize: UPLOAD_LIMITS.maxFilesize,
    createImageThumbnails: false,
    dictDefaultMessage: 
      "<i class='fa fa-cloud-upload'></i> Drag & drop videos here or click to upload",
    addRemoveLinks: true
  });

  // Set up event handlers
  dropzoneInstance.on("sending", (file, xhr, formData) => {
    // Add collision detection parameters to form data
    Object.entries(PARAM_DEFAULTS).forEach(([key, value]) => {
      formData.append(key, value);
    });
    
    // Show upload progress UI
    showUploadProgress();
  });

  dropzoneInstance.on("uploadprogress", (file, progress) => {
    // Update progress bar during upload
    document.getElementById("progress").style.width = `${progress}%`;
    document.getElementById("progress-text").textContent = `${Math.round(progress)}%`;
  });

  dropzoneInstance.on("success", (file, response) => {
    console.log("Upload successful:", response);
    
    // Start polling for processing status
    startPolling(response.job_id);
  });

  dropzoneInstance.on("error", (file, errorMessage) => {
    console.error("Upload error:", errorMessage);
    showError(typeof errorMessage === 'string' ? errorMessage : "Upload failed. Check file format and size.");
  });

  return dropzoneInstance;
}

/**
 * Resets the uploader to its initial state
 */
export function resetUploader() {
  if (dropzoneInstance) {
    dropzoneInstance.removeAllFiles();
  }
}
