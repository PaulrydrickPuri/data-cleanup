/**
 * Handles export functionality for collision data
 * Provides CSV and PDF export capabilities
 */
import { getCollisionData } from './renderer.js';

/**
 * Exports collision data as CSV
 * @param {string} jobId - The job ID for the filename
 */
export function exportCSV(jobId) {
  const collisionData = getCollisionData();
  
  if (!collisionData || !collisionData.length) {
    alert('No collision data available to export');
    return;
  }
  
  // Create CSV content
  let csvContent = 'Time,Objects,IoU Score,Severity\n';
  
  collisionData.forEach((collision, index) => {
    // Format time
    const time = formatTime(collision, index);
    
    // Format objects
    const objects = formatObjects(collision, index);
    
    // Format IoU score
    const iou = formatIouScore(collision);
    
    // Format severity
    const severity = calculateSeverity(collision);
    
    // Escape fields to handle commas and quotes
    const escapeCSV = (field) => {
      return `"${field.toString().replace(/"/g, '""')}"`;
    };
    
    // Add row to CSV
    csvContent += `${escapeCSV(time)},${escapeCSV(objects)},${escapeCSV(iou)},${escapeCSV(severity)}\n`;
  });
  
  // Create downloadable link
  const blob = new Blob([csvContent], { type: 'text/csv;charset=utf-8;' });
  const url = URL.createObjectURL(blob);
  const link = document.createElement('a');
  link.setAttribute('href', url);
  link.setAttribute('download', `collision_data_${jobId || 'export'}.csv`);
  link.style.visibility = 'hidden';
  document.body.appendChild(link);
  link.click();
  document.body.removeChild(link);
}

/**
 * Exports collision data as PDF report
 * @param {string} jobId - The job ID for the filename
 */
export function exportPDFReport(jobId) {
  const collisionData = getCollisionData();
  
  if (!collisionData || !collisionData.length) {
    alert('No collision data available to export');
    return;
  }
  
  try {
    // Create PDF using jsPDF
    const { jsPDF } = window.jspdf;
    if (!jsPDF) {
      throw new Error('jsPDF library not available');
    }
    
    const doc = new jsPDF();
    
    // Add title
    doc.setFontSize(18);
    doc.text('Collision Detection Report', 105, 15, { align: 'center' });
    
    // Add timestamp
    doc.setFontSize(10);
    const now = new Date();
    doc.text(`Generated: ${now.toLocaleString()}`, 105, 22, { align: 'center' });
    
    // Add parameters
    doc.setFontSize(12);
    doc.text('Detection Parameters:', 14, 30);
    doc.setFontSize(10);
    // Get parameters from DOM (or from a better state management in future)
    doc.text(`Total Collisions: ${collisionData.length}`, 20, 38);
    
    // Add collision table
    doc.setFontSize(12);
    doc.text('Collision Events:', 14, 50);
    
    // Table headers and data
    const tableColumn = ['Time', 'Objects', 'IoU Score', 'Severity'];
    const tableRows = [];
    
    // Populate table data
    collisionData.forEach((collision, index) => {
      tableRows.push([
        formatTime(collision, index),
        formatObjects(collision, index),
        formatIouScore(collision),
        calculateSeverity(collision)
      ]);
    });
    
    // Check if autoTable plugin is available
    if (typeof doc.autoTable === 'function') {
      // Create table
      doc.autoTable({
        head: [tableColumn],
        body: tableRows,
        startY: 55,
        theme: 'striped',
        headStyles: { fillColor: [41, 128, 185] },
        alternateRowStyles: { fillColor: [240, 240, 240] }
      });
    } else {
      // Fallback if autoTable is not available
      doc.setFontSize(10);
      doc.text('Table plugin not available. Raw data:', 14, 55);
      
      let y = 65;
      tableRows.forEach((row, i) => {
        doc.text(`${i+1}. ${row.join(' | ')}`, 20, y);
        y += 7;
        if (y > 280) {
          doc.addPage();
          y = 20;
        }
      });
    }
    
    // Save PDF
    doc.save(`collision_report_${jobId || 'export'}.pdf`);
  } catch (error) {
    console.error('PDF generation error:', error);
    alert(`Could not generate PDF: ${error.message}`);
  }
}

/**
 * Helper function to format time
 * @param {Object} collision - The collision data
 * @param {number} index - The collision index
 * @returns {string} Formatted time string
 */
function formatTime(collision, index) {
  if (collision.timestamp) {
    const minutes = Math.floor(collision.timestamp / 60);
    const seconds = Math.floor(collision.timestamp % 60);
    return `${minutes.toString().padStart(2, '0')}:${seconds.toString().padStart(2, '0')}`;
  } else if (collision.frame) {
    return `Frame ${collision.frame}`;
  }
  return `Event ${index + 1}`;
}

/**
 * Helper function to format objects
 * @param {Object} collision - The collision data
 * @param {number} index - The collision index
 * @returns {string} Formatted objects string
 */
function formatObjects(collision, index) {
  if (collision.person_id && collision.vehicle_id) {
    return `Person #${collision.person_id} & Vehicle #${collision.vehicle_id}`;
  } else if (collision.objects) {
    return collision.objects;
  }
  return `Collision #${index + 1}`;
}

/**
 * Helper function to format IoU score
 * @param {Object} collision - The collision data
 * @returns {string} Formatted IoU score
 */
function formatIouScore(collision) {
  if (collision.iou !== undefined) {
    return collision.iou.toFixed(3);
  } else if (collision.score !== undefined) {
    return collision.score.toFixed(3);
  }
  return 'N/A';
}

/**
 * Helper function to calculate severity
 * @param {Object} collision - The collision data
 * @returns {string} Severity level
 */
function calculateSeverity(collision) {
  const iouValue = collision.iou || collision.score || 0;
  
  if (iouValue > 0.5) {
    return 'High';
  } else if (iouValue > 0.3) {
    return 'Medium';
  }
  return 'Low';
}
