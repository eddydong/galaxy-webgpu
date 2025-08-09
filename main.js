import { setupFPS, setupControlPanel } from './controls.js';
import { computeShaderModule, renderShaderModule } from './shader.js';
import { mat4, vec3, quat } from './gl-matrix-lite.js';

async function main() {
    const distanceIndicator = document.getElementById('distance-indicator');
    const canvas = document.getElementById('webgpu-canvas');

    // Always use lighter blend mode
    let useLighterBlend = true;

    // Camera controls (using local minimal gl-matrix implementation)
        let zoom = 1;
        // View rotation matrix (mat4) separate from simulation space
        const viewMat = mat4.create(); // identity
        const tmpMat = mat4.create();
        const tmpQuat = quat.create();
        // Reusable vectors
        const vRight = vec3.fromValues(1,0,0);
        const vUp = vec3.fromValues(0,1,0);
    const near = 0.001;
    const far = 30.0;
    let cameraZ = 2.0;

    let dragging = false;
    let lastMouseX = 0;
    let lastMouseY = 0;

    const params = {
        numParticles: 30000,
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
                lastMouseX = e.clientX;
                lastMouseY = e.clientY;

                // Derive current basis vectors from viewMat
                const right = vec3.set(vRight, viewMat[0], viewMat[1], viewMat[2]);
                const up = vec3.set(vUp, viewMat[4], viewMat[5], viewMat[6]);
                vec3.normalize(right, right);
                vec3.normalize(up, up);

                const pitchAngle = -dy * 0.005; // vertical drag: rotate around screen-right
                const yawAngle = -dx * 0.005;   // reversed horizontal drag

                // Yaw (around up)
                quat.setAxisAngle(tmpQuat, up, yawAngle);
                mat4.fromQuat(tmpMat, tmpQuat);
                mat4.multiply(viewMat, tmpMat, viewMat);

                // Pitch (around right) - recompute right after yaw
                right[0] = viewMat[0]; right[1] = viewMat[1]; right[2] = viewMat[2];
                vec3.normalize(right, right);
                quat.setAxisAngle(tmpQuat, right, pitchAngle);
                mat4.fromQuat(tmpMat, tmpQuat);
                mat4.multiply(viewMat, tmpMat, viewMat);

                // Orthonormalize (Gram-Schmidt) to prevent drift
                const r0x = viewMat[0], r0y = viewMat[1], r0z = viewMat[2];
                const r1x = viewMat[4], r1y = viewMat[5], r1z = viewMat[6];
                let len0 = Math.hypot(r0x,r0y,r0z);
                viewMat[0]=r0x/len0; viewMat[1]=r0y/len0; viewMat[2]=r0z/len0;
                // Make up orthogonal to right
                let dotRU = viewMat[0]*r1x + viewMat[1]*r1y + viewMat[2]*r1z;
                let ux = r1x - dotRU*viewMat[0];
                let uy = r1y - dotRU*viewMat[1];
                let uz = r1z - dotRU*viewMat[2];
                let lenU = Math.hypot(ux,uy,uz);
                ux/=lenU; uy/=lenU; uz/=lenU;
                viewMat[4]=ux; viewMat[5]=uy; viewMat[6]=uz;
                // Forward = right x up
                const fx = viewMat[1]*viewMat[6] - viewMat[2]*viewMat[5];
                const fy = viewMat[2]*viewMat[4] - viewMat[0]*viewMat[6];
                const fz = viewMat[0]*viewMat[5] - viewMat[1]*viewMat[4];
                viewMat[8]=fx; viewMat[9]=fy; viewMat[10]=fz;
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
    // Choose a larger workgroup size (safe upper bound 256) to reduce dispatch count
    const WORKGROUP_SIZE = Math.min(256, device.limits?.maxComputeWorkgroupSizeX || 256);
    // Buffer for max radius output (one float per workgroup)
    let numWorkgroups = Math.ceil(params.numParticles / WORKGROUP_SIZE);
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
    const READBACK_INTERVAL = 30; // frames between max-radius readbacks (was 10)
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
    const renderParamsData = new Float32Array(24); // 6 vec4 slots
    function writeRenderParams(){
        // params0
        renderParamsData[0] = zoom;
        // params1
        renderParamsData[4] = near; renderParamsData[5] = far; renderParamsData[6] = cameraZ; renderParamsData[7] = aspect;
        // view matrix rows (rotation only)
        renderParamsData[8]  = viewMat[0]; renderParamsData[9]  = viewMat[1]; renderParamsData[10] = viewMat[2]; renderParamsData[11] = 0.0;
        renderParamsData[12] = viewMat[4]; renderParamsData[13] = viewMat[5]; renderParamsData[14] = viewMat[6]; renderParamsData[15] = 0.0;
        renderParamsData[16] = viewMat[8]; renderParamsData[17] = viewMat[9]; renderParamsData[18] = viewMat[10]; renderParamsData[19] = 0.0;
        renderParamsData[20] = 0.0; renderParamsData[21] = 0.0; renderParamsData[22] = 0.0; renderParamsData[23] = 1.0;
    }
    writeRenderParams();
    const renderParamsBuffer = device.createBuffer({
        size: 24 * 4, // 24 floats (96 bytes)
        usage: GPUBufferUsage.UNIFORM | GPUBufferUsage.COPY_DST,
    });
    device.queue.writeBuffer(renderParamsBuffer, 0, renderParamsData);

    const computeModule = computeShaderModule(device, WORKGROUP_SIZE);
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
    // Pre-create ping-pong bind groups (we'll recreate if particle buffers change)
    let computeBindGroups = [
        device.createBindGroup({
            layout: computePipeline.getBindGroupLayout(0),
            entries: [
                { binding: 0, resource: { buffer: particleBuffers[0] } },
                { binding: 1, resource: { buffer: particleBuffers[1] } },
                { binding: 2, resource: { buffer: simParamsBuffer } },
                { binding: 3, resource: { buffer: maxRadiusBuffer } },
            ],
        }),
        device.createBindGroup({
            layout: computePipeline.getBindGroupLayout(0),
            entries: [
                { binding: 0, resource: { buffer: particleBuffers[1] } },
                { binding: 1, resource: { buffer: particleBuffers[0] } },
                { binding: 2, resource: { buffer: simParamsBuffer } },
                { binding: 3, resource: { buffer: maxRadiusBuffer } },
            ],
        })
    ];

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
            numWorkgroups = Math.ceil(params.numParticles / WORKGROUP_SIZE);
            maxRadiusBuffer = device.createBuffer({
                size: 4,
                usage: GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_SRC | GPUBufferUsage.COPY_DST,
            });
            maxRadiusReadBuffer = device.createBuffer({
                size: 4,
                usage: GPUBufferUsage.COPY_DST | GPUBufferUsage.MAP_READ,
            });
            // Recreate compute bind groups with new buffers
            computeBindGroups = [
                device.createBindGroup({
                    layout: computePipeline.getBindGroupLayout(0),
                    entries: [
                        { binding: 0, resource: { buffer: particleBuffers[0] } },
                        { binding: 1, resource: { buffer: particleBuffers[1] } },
                        { binding: 2, resource: { buffer: simParamsBuffer } },
                        { binding: 3, resource: { buffer: maxRadiusBuffer } },
                    ],
                }),
                device.createBindGroup({
                    layout: computePipeline.getBindGroupLayout(0),
                    entries: [
                        { binding: 0, resource: { buffer: particleBuffers[1] } },
                        { binding: 1, resource: { buffer: particleBuffers[0] } },
                        { binding: 2, resource: { buffer: simParamsBuffer } },
                        { binding: 3, resource: { buffer: maxRadiusBuffer } },
                    ],
                })
            ];
            lastMaxDist = 0;
            t = 0;
        },
        toggleAutoSpin
    });

    const fpsCounter = document.getElementById('fps-counter'); // container from setupFPS
    const renderFpsCounter = document.getElementById('render-fps');
    const physicsFpsCounter = document.getElementById('physics-fps');
    let lastTime = performance.now();
    let frameCount = 0;
    // Temporal decoupling: fixed physics rate (60Hz) while rendering every frame
    const TARGET_PHYSICS_HZ = 60;
    const PHYSICS_STEP = 1 / TARGET_PHYSICS_HZ; // seconds
    let physicsAccumulator = 0;
    let physicsFrameCount = 0; // counts only executed physics steps
    let physicsStepsThisSecond = 0;
    let lastPhysicsTime = performance.now();

    let t = 0;
    let prevFrameTime = performance.now();
    function frame() {
        // (Readback moved to after compute submission)
        const currentTime = performance.now();
        frameCount++;
        if (currentTime - lastTime >= 1000) {
            if (renderFpsCounter) renderFpsCounter.textContent = `FPS: ${frameCount}`;
            frameCount = 0;
            lastTime = currentTime;
        }
        // Frame delta (clamp large spikes)
        let frameDeltaSec = (currentTime - prevFrameTime) * 0.001;
        if (frameDeltaSec > 0.25) frameDeltaSec = 0.25; // avoid spiraling
        prevFrameTime = currentTime;
        physicsAccumulator += frameDeltaSec;

        if (autoSpin && !dragging) {
            // Apply a gentle continuous yaw around the current 'up' axis of the view matrix
            const upX = viewMat[4], upY = viewMat[5], upZ = viewMat[6];
            quat.setAxisAngle(tmpQuat, [upX, upY, upZ], 0.001);
            mat4.fromQuat(tmpMat, tmpQuat);
            mat4.multiply(viewMat, tmpMat, viewMat);
            // Orthonormalize (same logic as mouse drag) to avoid drift
            const r0x = viewMat[0], r0y = viewMat[1], r0z = viewMat[2];
            const r1x = viewMat[4], r1y = viewMat[5], r1z = viewMat[6];
            let len0 = Math.hypot(r0x,r0y,r0z);
            viewMat[0]=r0x/len0; viewMat[1]=r0y/len0; viewMat[2]=r0z/len0;
            let dotRU = viewMat[0]*r1x + viewMat[1]*r1y + viewMat[2]*r1z;
            let ux = r1x - dotRU*viewMat[0];
            let uy = r1y - dotRU*viewMat[1];
            let uz = r1z - dotRU*viewMat[2];
            let lenU = Math.hypot(ux,uy,uz);
            ux/=lenU; uy/=lenU; uz/=lenU;
            viewMat[4]=ux; viewMat[5]=uy; viewMat[6]=uz;
            const fx = viewMat[1]*viewMat[6] - viewMat[2]*viewMat[5];
            const fy = viewMat[2]*viewMat[4] - viewMat[0]*viewMat[6];
            const fz = viewMat[0]*viewMat[5] - viewMat[1]*viewMat[4];
            viewMat[8]=fx; viewMat[9]=fy; viewMat[10]=fz;
        }

        // Update camera params
    const aspect = canvas.width / canvas.height;
    renderParamsData[5] = far; // unchanged but keep explicit if needed
    renderParamsData[6] = cameraZ;
    renderParamsData[7] = aspect;
    renderParamsData[0] = zoom;
    // update view rows
    renderParamsData[8]  = viewMat[0]; renderParamsData[9]  = viewMat[1]; renderParamsData[10] = viewMat[2];
    renderParamsData[12] = viewMat[4]; renderParamsData[13] = viewMat[5]; renderParamsData[14] = viewMat[6];
    renderParamsData[16] = viewMat[8]; renderParamsData[17] = viewMat[9]; renderParamsData[18] = viewMat[10];
    device.queue.writeBuffer(renderParamsBuffer, 0, renderParamsData);

        // We may advance physics in fixed-size steps; run zero or more physics steps this frame
        // Each physics step performs one compute dispatch
        let stepsThisFrame = 0;
        while (physicsAccumulator >= PHYSICS_STEP) {
            physicsAccumulator -= PHYSICS_STEP;
            stepsThisFrame++;
            // Write (constant) sim params (fixed dt, no scaling now)
            simParamsData[0] = params.G;
            simParamsData[1] = params.dt; // treat params.dt as per-step integration interval
            device.queue.writeBuffer(simParamsBuffer, 0, simParamsData);
            const commandEncoder = device.createCommandEncoder();
            // Compute pass (only when stepping physics)
            device.queue.writeBuffer(maxRadiusBuffer, 0, new Uint32Array([0])); // reset max distance atomic
            const computePass = commandEncoder.beginComputePass();
            computePass.setPipeline(computePipeline);
            computePass.setBindGroup(0, computeBindGroups[t % 2]);
            computePass.dispatchWorkgroups(numWorkgroups);
            computePass.end();
            // Readback scheduling based on physics step count
            const doRead = (physicsFrameCount % READBACK_INTERVAL) === 0 && !maxReadPending;
            if (doRead) {
                commandEncoder.copyBufferToBuffer(maxRadiusBuffer, 0, maxRadiusReadBuffer, 0, 4);
                maxReadPending = true;
            }
            // Submit compute work immediately (allows overlap with potential next render CPU tasks)
            device.queue.submit([commandEncoder.finish()]);
            // Handle readback promise (non-blocking)
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
            } else if (distanceIndicator) {
                distanceIndicator.textContent = `R: ${lastMaxDist.toExponential(1)}`;
            }
            t++;
            physicsFrameCount++;
            physicsStepsThisSecond++;
            // Update ping-pong bind groups index implicitly via t
        }
        // If no physics step executed, we still refresh indicator occasionally
        if (stepsThisFrame === 0 && distanceIndicator && (frameCount % 120) === 0) {
            distanceIndicator.textContent = `R: ${lastMaxDist.toExponential(1)}`;
        }

        // Physics FPS update once per second
        if (currentTime - lastPhysicsTime >= 1000) {
            if (physicsFpsCounter) physicsFpsCounter.textContent = `Physics: ${physicsStepsThisSecond}`;
            physicsStepsThisSecond = 0;
            lastPhysicsTime = currentTime;
        }

        // Prepare to render using the latest completed physics buffer (t points to next write index)
        const srcBufferIndex = (t) % 2; // last written destination is (t%2) after increment in loop
        const renderSrc = particleBuffers[srcBufferIndex];
        // Render pass (always each frame)
        const commandEncoder = device.createCommandEncoder();

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
    renderPass.setVertexBuffer(0, renderSrc);
        renderPass.setVertexBuffer(1, quadVertexBuffer);
        renderPass.draw(4, params.numParticles, 0, 0);
        renderPass.end();

    device.queue.submit([commandEncoder.finish()]);
        requestAnimationFrame(frame);
    }

    requestAnimationFrame(frame);
}

main()
