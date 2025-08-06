// controls.js
// Handles the control panel logic and events
export function setupControlPanel({ params, device, simParamsBuffer, particleBuffers, reinitParticles, toggleAutoSpin }) {
    const controlPanel = document.createElement('div');
    controlPanel.className = 'control-panel';
    controlPanel.innerHTML = `
      <label class="control-label">
        <span>Gravity: <span id="g-value">25</span></span>
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
        params.G = gSliderValue * 0.0000001;
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
