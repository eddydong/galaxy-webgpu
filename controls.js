// controls.js

// Handles UI elements like FPS counter and WebGPU warning
export function setupFPS() {
  // FPS Counter container with separate lines for render & physics
  let fpsCounter = document.getElementById('fps-counter');
  if (!fpsCounter) {
    fpsCounter = document.createElement('div');
    fpsCounter.id = 'fps-counter';
    fpsCounter.className = 'fps-counter';
    document.body.appendChild(fpsCounter);
  }
  fpsCounter.className = 'fps-counter';
  fpsCounter.style.display = '';
  // Ensure child lines exist
  let renderLine = document.getElementById('render-fps');
  if (!renderLine) {
    renderLine = document.createElement('div');
    renderLine.id = 'render-fps';
    renderLine.textContent = 'FPS: --';
    fpsCounter.appendChild(renderLine);
  }
  let physicsLine = document.getElementById('physics-fps');
  if (!physicsLine) {
    physicsLine = document.createElement('div');
    physicsLine.id = 'physics-fps';
    physicsLine.style.fontSize = '0.9em';
    physicsLine.style.marginTop = '4px';
    physicsLine.textContent = 'Physics: --';
    fpsCounter.appendChild(physicsLine);
  }
    // WebGPU warning will be handled in main.js as before
}

// Handles the control panel logic and events
export function setupControlPanel({ params, device, simParamsBuffer, particleBuffers, reinitParticles, toggleAutoSpin }) {
    const controlPanel = document.createElement('div');
    controlPanel.className = 'control-panel';
    controlPanel.innerHTML = `
      <label class="control-label">
        <span>Gravity: <span id="g-value">30</span></span>
        <input type="range" id="g-slider" min="1" max="100" step="1" value="30">
      </label>
      <label class="control-label">
        <span>Time Step: <span id="dt-value">30</span></span>
        <input type="range" id="dt-slider" min="1" max="100" step="1" value="30">
      </label>
      <label class="control-label">
        <span>Particles: <span id="star-value">${params.numParticles}</span></span>
        <input type="range" id="star-slider" min="100" max="30000" step="100" value="${params.numParticles}">
      </label>
      <button id="reset-btn" class="reset-button">Reset</button>
      <label class="control-label">
        <span>Spin</span>
        <input type="checkbox" id="auto-spin-checkbox" checked>
      </label>
    `;
    document.body.appendChild(controlPanel);

    const gSlider = controlPanel.querySelector('#g-slider');
    const gValue = controlPanel.querySelector('#g-value');
    const dtSlider = controlPanel.querySelector('#dt-slider');
    const dtValue = controlPanel.querySelector('#dt-value');
    const starSlider = controlPanel.querySelector('#star-slider');
    const starValue = controlPanel.querySelector('#star-value');
    const resetBtn = controlPanel.querySelector('#reset-btn');
    const autoSpinCheckbox = controlPanel.querySelector('#auto-spin-checkbox');

    let pendingNumParticles = params.numParticles;

    gSlider.addEventListener('input', () => {
        const gSliderValue = parseInt(gSlider.value);
        params.G = gSliderValue * 1e-11;
        gValue.textContent = gSliderValue;
        device.queue.writeBuffer(simParamsBuffer, 0, new Float32Array([params.G, params.dt]));
    });
    dtSlider.addEventListener('input', () => {
        const dtSliderValue = parseInt(dtSlider.value);
        params.dt = dtSliderValue * 0.00001;
        dtValue.textContent = dtSliderValue;
        device.queue.writeBuffer(simParamsBuffer, 0, new Float32Array([params.G, params.dt]));
    });

    starSlider.addEventListener('input', () => {
        pendingNumParticles = parseInt(starSlider.value);
        starValue.textContent = pendingNumParticles;
    });

    resetBtn.addEventListener('click', () => {
        params.numParticles = pendingNumParticles;
        reinitParticles();
    });

    autoSpinCheckbox.addEventListener('change', () => {
        toggleAutoSpin(autoSpinCheckbox.checked);
    });
}

export function showAlert(message) {
	let notice = document.getElementById('alert-notice');
	if (!notice) {
		notice = document.createElement('div');
		notice.id = 'alert-notice';
		Object.assign(notice.style, {
			position: 'fixed',
			top: '50%',
			left: '50%',
			transform: 'translate(-50%, -50%)',
			background: 'rgba(0,0,0,0.85)',
			color: 'white',
			padding: '32px 48px',
			borderRadius: '16px',
			fontSize: '1.3em',
			fontFamily: 'sans-serif',
			zIndex: '100'
		});
		document.body.appendChild(notice);
	}
	notice.textContent = message;
}
