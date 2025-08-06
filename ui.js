// ui.js
// Handles UI elements like FPS counter and WebGPU warning
export function setupUI() {
    // FPS Counter
    let fpsCounter = document.getElementById('fps-counter');
    if (!fpsCounter) {
        fpsCounter = document.createElement('div');
        fpsCounter.id = 'fps-counter';
        fpsCounter.className = 'fps-counter';
        document.body.appendChild(fpsCounter);
    } else {
        // Ensure it is visible and styled
        fpsCounter.className = 'fps-counter';
        fpsCounter.style.display = '';
    }
    // WebGPU warning will be handled in main.js as before
}
