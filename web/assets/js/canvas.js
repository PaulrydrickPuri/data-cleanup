/**
 * Handles canvas operations for bounding box visualization
 * Manages drawing on the canvas overlay for the video player
 */

/**
 * Sets up the canvas for bounding box visualization
 * @param {HTMLVideoElement} videoElement - The video element
 * @param {HTMLCanvasElement} canvasElement - The canvas element
 * @param {Array} boundingBoxes - The bounding box data
 */
export function setupBoundingBoxCanvas(videoElement, canvasElement, boundingBoxes) {
  if (!boundingBoxes || !boundingBoxes.length) {
    console.warn('No bounding box data available');
    return;
  }
  
  const ctx = canvasElement.getContext('2d');
  
  // Set canvas dimensions to match video
  videoElement.onloadedmetadata = () => {
    resizeCanvas(videoElement, canvasElement);
    
    // Add event listener for time updates to draw bounding boxes
    videoElement.addEventListener('timeupdate', () => {
      drawBoundingBoxes(ctx, canvasElement.width, canvasElement.height, 
                        videoElement.currentTime, boundingBoxes);
    });
  };
  
  // Handle browser resize events
  window.addEventListener('resize', () => resizeCanvas(videoElement, canvasElement));
}

/**
 * Resizes the canvas to match the video dimensions
 * @param {HTMLVideoElement} videoElement - The video element
 * @param {HTMLCanvasElement} canvasElement - The canvas element
 */
function resizeCanvas(videoElement, canvasElement) {
  const videoRect = videoElement.getBoundingClientRect();
  canvasElement.width = videoElement.videoWidth || videoRect.width;
  canvasElement.height = videoElement.videoHeight || videoRect.height;
  
  // Position canvas over video (if needed)
  canvasElement.style.position = 'absolute';
  canvasElement.style.top = `${videoRect.top}px`;
  canvasElement.style.left = `${videoRect.left}px`;
}

/**
 * Draws bounding boxes on the canvas
 * @param {CanvasRenderingContext2D} ctx - The canvas rendering context
 * @param {number} width - The canvas width
 * @param {number} height - The canvas height
 * @param {number} currentTime - The current video time
 * @param {Array} boundingBoxes - The bounding box data
 */
function drawBoundingBoxes(ctx, width, height, currentTime, boundingBoxes) {
  // Clear canvas
  ctx.clearRect(0, 0, width, height);
  
  // Find boxes for current time
  const currentBoxes = boundingBoxes.filter(box => {
    return box.timestamp <= currentTime && 
           currentTime < (box.timestamp + (box.duration || 0.5));
  });
  
  if (currentBoxes.length === 0) return;
  
  // Draw each box
  let collisionDetected = false;
  
  currentBoxes.forEach(box => {
    // Set color based on object type and collision status
    let boxColor;
    
    if (box.collision) {
      boxColor = 'rgba(255, 0, 0, 0.8)'; // Red for collision
      collisionDetected = true;
    } else {
      switch(box.class) {
        case 'person':
          boxColor = 'rgba(0, 255, 0, 0.6)'; // Green
          break;
        case 'vehicle':
        case 'car':
        case 'truck':
          boxColor = 'rgba(0, 0, 255, 0.6)'; // Blue
          break;
        default:
          boxColor = 'rgba(255, 255, 0, 0.6)'; // Yellow
      }
    }
    
    // Scale coordinates to canvas dimensions
    const x = box.x * width;
    const y = box.y * height;
    const w = box.width * width;
    const h = box.height * height;
    
    // Draw rectangle
    ctx.strokeStyle = boxColor;
    ctx.lineWidth = 3;
    ctx.strokeRect(x, y, w, h);
    
    // Draw label
    if (box.class || box.id) {
      ctx.fillStyle = boxColor;
      ctx.font = '14px Arial';
      const label = box.class ? `${box.class} ${box.id || ''}` : `Object ${box.id || ''}`;
      const metrics = ctx.measureText(label);
      const labelHeight = 20;
      
      // Label background
      ctx.fillRect(x, y - labelHeight, metrics.width + 10, labelHeight);
      
      // Label text
      ctx.fillStyle = 'rgba(255, 255, 255, 1.0)';
      ctx.fillText(label, x + 5, y - 5);
    }
  });
  
  // Show collision indicator if collision detected
  const collisionIndicator = document.getElementById('collision-indicator');
  if (collisionIndicator) {
    collisionIndicator.style.display = collisionDetected ? 'block' : 'none';
  }
}
