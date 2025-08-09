import { setupFPS, setupControlPanel } from './controls.js';
import { computeShaderModule, renderShaderModule } from './shader.js';

async function main() {
    const distanceIndicator = document.getElementById('distance-indicator');
    const canvas = document.getElementById('webgpu-canvas');

    // Always use lighter blend mode
    let useLighterBlend = true;

    // Camera controls
    let zoom = 1; // slightly zoomed out
    let rotX = 0; // tilt for better perspective
    let rotY = 0; // rotate around Y axis for better view
    const near = 0.001;
    const far = 30.0;
    let cameraZ = 2.0;

    let dragging = false;
    let lastMouseX = 0;
    let lastMouseY = 0;

    const params = {
        numParticles: 5000,
        G: 30 * 1e-11, // internal value (0.3 * 0.000001)
        dt: 0.0003, // internal value (30 * 0.00001)
    };    

    canvas.addEventListener('wheel', (e) => {
        e.preventDefault();
        // Dolly the camera instead of scaling particle positions
        // Negative deltaY (wheel up) moves camera closer
        const factor = Math.exp(e.deltaY * 0.001);
        cameraZ = Math.min(20.0, Math.max(0.01, cameraZ * factor));
        // Keep a derived zoom value for UI/other effects if needed (optional smoothing)
        zoom = 2.0 / cameraZ; // maintain inverse relation only for any UI reads; shader no longer scales positions
    });

    canvas.addEventListener('mousedown', (e) => {
        dragging = true;
        lastMouseX = e.clientX;
        lastMouseY = e.clientY;
    });
    window.addEventListener('mouseup', () => { dragging = false; });
    window.addEventListener('mousemove', (e) => {
        if (dragging && e.buttons === 1) {
            const dx = e.clientX - lastMouseX;
            const dy = e.clientY - lastMouseY;
            rotY += dx * 0.005; // horizontal drag rotates Y
            rotX -= dy * 0.005; // vertical drag rotates X
            lastMouseX = e.clientX;
            lastMouseY = e.clientY;
        }
    });

    function setCanvasSize() {
        const dpr = window.devicePixelRatio || 1;
        canvas.width = window.innerWidth * dpr;
        canvas.height = window.innerHeight * dpr;
        canvas.style.width = `${window.innerWidth}px`;
        canvas.style.height = `${window.innerHeight}px`;
    }
    setCanvasSize();
    window.addEventListener('resize', setCanvasSize);

    if (!navigator.gpu) {
        // Show notice for non-WebGPU browsers
        const notice = document.createElement('div');
        notice.textContent = 'WebGPU is not supported in this browser. Please use Google Chrome or other WebGPU-ready browsers.';
        notice.style.position = 'fixed';
        notice.style.top = '50%';
        notice.style.left = '50%';
        notice.style.transform = 'translate(-50%, -50%)';
        notice.style.background = 'rgba(0,0,0,0.85)';
        notice.style.color = 'white';
        notice.style.padding = '32px 48px';
        notice.style.borderRadius = '16px';
        notice.style.fontSize = '1.3em';
        notice.style.fontFamily = 'sans-serif';
        notice.style.zIndex = '100';
        document.body.appendChild(notice);
        throw new Error("WebGPU not supported on this browser.");
    }

    const adapter = await navigator.gpu.requestAdapter();
    if (!adapter) {
        throw new Error("No appropriate GPUAdapter found.");
    }

    const device = await adapter.requestDevice();
    // Buffer for max radius output (one float per workgroup)
    let numWorkgroups = Math.ceil(params.numParticles / 64);
    // Single float (atomic u32 bits) buffer for global max distance
    let maxRadiusBuffer = device.createBuffer({
        size: 4,
        usage: GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_SRC | GPUBufferUsage.COPY_DST,
    });
    let maxRadiusReadBuffer = device.createBuffer({
        size: 4,
        usage: GPUBufferUsage.COPY_DST | GPUBufferUsage.MAP_READ,
    });
    let maxReadPending = false;
    let lastMaxDist = 0;
    const context = canvas.getContext('webgpu');
    const canvasFormat = navigator.gpu.getPreferredCanvasFormat();
    context.configure({
        device: device,
        format: canvasFormat,
    });

    function createParticleData(numParticles) {
        const data = new Float32Array(numParticles * 8);
        for (let i = 0; i < numParticles; i++) {
            // Random position in a ball (uniformly distributed in a sphere)
            let u = Math.random();
            let v = Math.random();
            let w = Math.random();
            let theta = 2 * Math.PI * u;
            let phi = Math.acos(2 * v - 1);
            let r = Math.cbrt(w) * 1.0; // radius, cube root for uniform sphere
            let x = r * Math.sin(phi) * Math.cos(theta);
            let y = r * Math.sin(phi) * Math.sin(theta);
            let z = r * Math.cos(phi);

            // Assign random mass, skewed towards lower values
            // Mass: mostly low, very rarely high
            const mass = 0.01 + Math.pow(Math.random()*0.99, 12) * 99999999.9;
            // Radius based on mass (mass is proportional to volume, so radius is proportional to cbrt(mass))
            const radius = 0.01 + Math.cbrt(mass / 100000000.0) * 0.04;

            // The data for each particle is stored as 8 floats (32 bytes),
            // matching the 'Particle' struct in the compute shader.
            // The layout is:
            // - floats 0-3: position (x, y, z) and mass (w)
            // - floats 4-7: velocity (vx, vy, vz) and radius (w)

            // Position
            data[i * 8 + 0] = x;
            data[i * 8 + 1] = y;
            data[i * 8 + 2] = z;
            data[i * 8 + 3] = mass;

            // Initial velocity: zero
            data[i * 8 + 4] = 0;
            data[i * 8 + 5] = 0;
            data[i * 8 + 6] = 0;
            
            data[i * 8 + 7] = radius;
        }
        return data;
    }

    const particleData = createParticleData(params.numParticles);

    const particleBuffers = [
        device.createBuffer({
            size: particleData.byteLength,
            usage: GPUBufferUsage.STORAGE | GPUBufferUsage.VERTEX | GPUBufferUsage.COPY_DST,
            mappedAtCreation: true,
        }),
        device.createBuffer({
            size: particleData.byteLength,
            usage: GPUBufferUsage.STORAGE | GPUBufferUsage.VERTEX | GPUBufferUsage.COPY_DST,
        })
    ];

    new Float32Array(particleBuffers[0].getMappedRange()).set(particleData);
    particleBuffers[0].unmap();

    const simParamsData = new Float32Array([params.G, params.dt]);
    const simParamsBuffer = device.createBuffer({
        size: simParamsData.byteLength,
        usage: GPUBufferUsage.UNIFORM | GPUBufferUsage.COPY_DST,
    });
    device.queue.writeBuffer(simParamsBuffer, 0, simParamsData);

    const aspect = canvas.width / canvas.height;
    const renderParamsData = new Float32Array([zoom, rotX, rotY, 0.0, near, far, cameraZ, aspect]);
    const renderParamsBuffer = device.createBuffer({
        size: 8 * 4, // 8 floats (32 bytes)
        usage: GPUBufferUsage.UNIFORM | GPUBufferUsage.COPY_DST,
    });
    device.queue.writeBuffer(renderParamsBuffer, 0, renderParamsData);

    const computeModule = computeShaderModule(device);
    const renderModule = renderShaderModule(device);

    // Create quad vertex buffer for particle rendering
    const quadVertexBuffer = device.createBuffer({
        size: 4 * 2 * 4, // 4 vertices, 2 floats (xy) each, 4 bytes per float
        usage: GPUBufferUsage.VERTEX,
        mappedAtCreation: true,
    });
    new Float32Array(quadVertexBuffer.getMappedRange()).set([
        -1.0, -1.0,
         1.0, -1.0,
        -1.0,  1.0,
         1.0,  1.0,
    ]);
    quadVertexBuffer.unmap();

    const computePipeline = device.createComputePipeline({
        layout: 'auto',
        compute: {
            module: computeModule,
            entryPoint: 'main',
        },
    });

    let renderPipeline, renderBindGroup;
    function recreateRenderPipeline() {
        renderPipeline = device.createRenderPipeline({
            layout: 'auto',
            vertex: {
                module: renderModule,
                entryPoint: 'vs_main',
                buffers: [
                    {
                        arrayStride: 32, // 8 floats * 4 bytes
                        stepMode: 'instance',
                        attributes: [
                            { // position
                                shaderLocation: 0,
                                offset: 0,
                                format: 'float32x3',
                            },
                            { // velocity
                                shaderLocation: 3,
                                offset: 16,
                                format: 'float32x4',
                            }
                        ],
                    },
                    {
                        arrayStride: 8,
                        stepMode: 'vertex',
                        attributes: [{
                            shaderLocation: 2, // updated from 1
                            offset: 0,
                            format: 'float32x2',
                        }],
                    }
                ],
            },
            fragment: {
                module: renderModule,
                entryPoint: 'fs_main',
                targets: [{
                    format: canvasFormat,
                    blend: useLighterBlend ? {
                        color: {
                            srcFactor: 'src-alpha',
                            dstFactor: 'one',
                            operation: 'add',
                        },
                        alpha: {
                            srcFactor: 'zero',
                            dstFactor: 'one-minus-src-alpha',
                            operation: 'add',
                        },
                    } : {
                        color: {
                            srcFactor: 'src-alpha',
                            dstFactor: 'one-minus-src-alpha',
                            operation: 'add',
                        },
                        alpha: {
                            srcFactor: 'one',
                            dstFactor: 'one-minus-src-alpha',
                            operation: 'add',
                        },
                    },
                }],
            },
            primitive: {
                topology: 'triangle-strip',
            },
        });
        renderBindGroup = device.createBindGroup({
            layout: renderPipeline.getBindGroupLayout(0),
            entries: [
                { binding: 0, resource: { buffer: renderParamsBuffer } },
            ],
        });
    }
    recreateRenderPipeline();


    let autoSpin = true;
    function toggleAutoSpin(enabled) {
        autoSpin = enabled;
    }

    setupFPS();
    setupControlPanel({
        params,
        device,
        simParamsBuffer,
        particleBuffers,
        reinitParticles: () => {
            const newParticleData = createParticleData(params.numParticles);
            // Resize buffers
            particleBuffers[0].destroy && particleBuffers[0].destroy();
            particleBuffers[1].destroy && particleBuffers[1].destroy();
            particleBuffers[0] = device.createBuffer({
                size: newParticleData.byteLength,
                usage: GPUBufferUsage.STORAGE | GPUBufferUsage.VERTEX | GPUBufferUsage.COPY_DST,
                mappedAtCreation: true,
            });
            particleBuffers[1] = device.createBuffer({
                size: newParticleData.byteLength,
                usage: GPUBufferUsage.STORAGE | GPUBufferUsage.VERTEX | GPUBufferUsage.COPY_DST,
            });
            new Float32Array(particleBuffers[0].getMappedRange()).set(newParticleData);
            particleBuffers[0].unmap();
            // Recreate max radius buffers (num particles may have changed)
            if (maxReadPending) {
                try { maxRadiusReadBuffer.unmap(); } catch(e){}
                maxReadPending = false;
            }
            maxRadiusBuffer.destroy && maxRadiusBuffer.destroy();
            maxRadiusReadBuffer.destroy && maxRadiusReadBuffer.destroy();
            numWorkgroups = Math.ceil(params.numParticles / 64);
            maxRadiusBuffer = device.createBuffer({
                size: 4,
                usage: GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_SRC | GPUBufferUsage.COPY_DST,
            });
            maxRadiusReadBuffer = device.createBuffer({
                size: 4,
                usage: GPUBufferUsage.COPY_DST | GPUBufferUsage.MAP_READ,
            });
            lastMaxDist = 0;
            t = 0;
        },
        toggleAutoSpin
    });

    const fpsCounter = document.getElementById('fps-counter'); // created for setupFPS
    let lastTime = performance.now();
    let frameCount = 0;

    let t = 0;
    let prevFrameTime = performance.now();
    function frame() {
    // (Readback moved to after compute submission)
        const currentTime = performance.now();
        frameCount++;
        if (currentTime - lastTime >= 1000) {
            fpsCounter.textContent = `FPS: ${frameCount}`;
            frameCount = 0;
            lastTime = currentTime;
        }

        // Calculate frame time in seconds, and clamp to avoid large jumps
        const frameTimeSec = Math.min(Math.max((currentTime - prevFrameTime) * 0.001, 0.001), 1/30);
        prevFrameTime = currentTime;

        if (autoSpin && !dragging) {
            rotY += 0.001;
        }

        // Update camera params
        renderParamsData[0] = zoom;
        renderParamsData[1] = rotX;
        renderParamsData[2] = rotY;
        // near, far remain constant
        renderParamsData[6] = cameraZ;
        const aspect = canvas.width / canvas.height;
        renderParamsData[7] = aspect;
        device.queue.writeBuffer(renderParamsBuffer, 0, renderParamsData);

        // Scale dt by frame time for consistent simulation speed
        const effectiveDt = params.dt * (frameTimeSec / (1/60)); // normalized to 60fps
        device.queue.writeBuffer(simParamsBuffer, 0, new Float32Array([params.G, effectiveDt]));

        const commandEncoder = device.createCommandEncoder();

        // Compute pass
    // Reset global max (atomic u32) to 0 before compute
    device.queue.writeBuffer(maxRadiusBuffer, 0, new Uint32Array([0]));
    const computePass = commandEncoder.beginComputePass();
        computePass.setPipeline(computePipeline);
        const computeBindGroup = device.createBindGroup({
            layout: computePipeline.getBindGroupLayout(0),
            entries: [
                { binding: 0, resource: { buffer: particleBuffers[t % 2] } },
                { binding: 1, resource: { buffer: particleBuffers[(t + 1) % 2] } },
                { binding: 2, resource: { buffer: simParamsBuffer } },
                { binding: 3, resource: { buffer: maxRadiusBuffer } },
            ],
        });
        computePass.setBindGroup(0, computeBindGroup);
        computePass.dispatchWorkgroups(numWorkgroups);
        computePass.end();

        // Schedule copy of per-workgroup maxima to staging buffer every 10 frames
        const doRead = (frameCount % 10) === 0 && !maxReadPending;
        if (doRead) {
            commandEncoder.copyBufferToBuffer(maxRadiusBuffer, 0, maxRadiusReadBuffer, 0, 4);
            maxReadPending = true;
        }

        // Render pass
        const textureView = context.getCurrentTexture().createView();
        const renderPass = commandEncoder.beginRenderPass({
            colorAttachments: [{
                view: textureView,
                clearValue: { r: 0, g: 0, b: 0, a: 1 },
                loadOp: 'clear',
                storeOp: 'store',
            }],
        });
        renderPass.setPipeline(renderPipeline);
        renderPass.setBindGroup(0, renderBindGroup);
        renderPass.setVertexBuffer(0, particleBuffers[(t + 1) % 2]);
        renderPass.setVertexBuffer(1, quadVertexBuffer);
        renderPass.draw(4, params.numParticles, 0, 0);
        renderPass.end();

        device.queue.submit([commandEncoder.finish()]);

        // Map and read after GPU work submission (non-blocking)
        if (doRead) {
                maxRadiusReadBuffer.mapAsync(GPUMapMode.READ).then(() => {
                    const arr = new Uint32Array(maxRadiusReadBuffer.getMappedRange());
                    const u32 = arr[0];
                    lastMaxDist = new Float32Array(new Uint32Array([u32]).buffer)[0];
                    if (distanceIndicator) {
                        distanceIndicator.textContent = `R: ${lastMaxDist.toExponential(1)}`;
                    }
                    maxRadiusReadBuffer.unmap();
                    maxReadPending = false;
                }).catch(() => { maxReadPending = false; });
        } else if (distanceIndicator && frameCount % 10 !== 0) {
            // Keep indicator refreshed even if not reading this frame
            distanceIndicator.textContent = `R: ${lastMaxDist.toExponential(1)}`;
        }
        t++;
        requestAnimationFrame(frame);
    }

    requestAnimationFrame(frame);
}

main()
