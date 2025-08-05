async function main() {
    const canvas = document.getElementById('webgpu-canvas');

    // Always use lighter blend mode
    let useLighterBlend = true;

    // Camera controls
    let zoom = 0.8; // slightly zoomed out
    let rotX = Math.PI; // tilt for better perspective
    let rotY = Math.PI; // rotate around Y axis for better view
    let dragging = false;
    let lastMouseX = 0;
    let lastMouseY = 0;

    const params = {
        numParticles: 15000,
        particleSize: 0.02,
        G: 0.00003, // internal value (30 * 0.000001)
        dt: 0.0003, // internal value (30 * 0.00001)
    };    

    // [particleSize, zoom, rotX, rotY, near, far, cameraZ]
    const near = 0.01;
    const far = 100.0;
    let cameraZ = 2.0;

    canvas.addEventListener('wheel', (e) => {
        e.preventDefault();
        zoom *= Math.exp(-e.deltaY * 0.001); // normal zoom direction
        zoom = Math.max(0.1, Math.min(zoom, 10.0));
        // Keep the center at the screen center by adjusting cameraZ
        cameraZ = 2.0 / zoom;
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
            rotY -= dx * 0.005; // reversed horizontal drag rotates Y
            rotX -= dy * 0.005; // reversed vertical drag rotates X
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
    const context = canvas.getContext('webgpu');
    const canvasFormat = navigator.gpu.getPreferredCanvasFormat();
    context.configure({
        device: device,
        format: canvasFormat,
    });

    // pos, vel
    const particleAndVelocityData = new Float32Array(params.numParticles * 8); 

    // Initialize particles in a disk, each with random mass
    for (let i = 0; i < params.numParticles; i++) {
        // Random position in a ball (uniformly distributed in a sphere)
        let u = Math.random();
        let v = Math.random();
        let w = Math.random();
        let theta = 2 * Math.PI * u;
        let phi = Math.acos(2 * v - 1);
        let r = Math.cbrt(w) * 0.9; // radius, cube root for uniform sphere
        let x = r * Math.sin(phi) * Math.cos(theta);
        let y = r * Math.sin(phi) * Math.sin(theta);
        let z = r * Math.cos(phi);

        // Assign random mass between 0.5 and 5.0
        const mass = 0.5 + Math.random() * 99.5;

        // Position
        particleAndVelocityData[i * 8 + 0] = x;
        particleAndVelocityData[i * 8 + 1] = y;
        particleAndVelocityData[i * 8 + 2] = z;
        particleAndVelocityData[i * 8 + 3] = mass;

        // Initial velocity: zero (no preferred orbit direction in a ball)
        particleAndVelocityData[i * 8 + 4] = 0;
        particleAndVelocityData[i * 8 + 5] = 0;
        particleAndVelocityData[i * 8 + 6] = 0;
        particleAndVelocityData[i * 8 + 7] = 0; // unused
    }

    const particleBuffers = [
        device.createBuffer({
            size: particleAndVelocityData.byteLength,
            usage: GPUBufferUsage.STORAGE | GPUBufferUsage.VERTEX | GPUBufferUsage.COPY_DST,
            mappedAtCreation: true,
        }),
        device.createBuffer({
            size: particleAndVelocityData.byteLength,
            usage: GPUBufferUsage.STORAGE | GPUBufferUsage.VERTEX | GPUBufferUsage.COPY_DST,
        })
    ];

    new Float32Array(particleBuffers[0].getMappedRange()).set(particleAndVelocityData);
    particleBuffers[0].unmap();

    const simParamsData = new Float32Array([params.G, params.dt]);
    const simParamsBuffer = device.createBuffer({
        size: simParamsData.byteLength,
        usage: GPUBufferUsage.UNIFORM | GPUBufferUsage.COPY_DST,
    });
    device.queue.writeBuffer(simParamsBuffer, 0, simParamsData);

    const renderParamsData = new Float32Array([params.particleSize, zoom, rotX, rotY, near, far, cameraZ, 0.0]);
    const renderParamsBuffer = device.createBuffer({
        size: 8 * 4, // 8 floats (32 bytes)
        usage: GPUBufferUsage.UNIFORM | GPUBufferUsage.COPY_DST,
    });
    device.queue.writeBuffer(renderParamsBuffer, 0, renderParamsData);

    const computeShaderModule = device.createShaderModule({
        code: `
            struct Particle {
                pos: vec4<f32>,
                vel: vec4<f32>,
            };

            struct Particles {
                particles: array<Particle>,
            };

            @group(0) @binding(0) var<storage, read> particlesA: Particles;
            @group(0) @binding(1) var<storage, read_write> particlesB: Particles;
            
            @group(0) @binding(2) var<uniform> sim_params: vec2<f32>; // G, dt

            const softening = 0.01;

            @compute @workgroup_size(64)
            fn main(@builtin(global_invocation_id) global_id: vec3<u32>) {
                let idx = global_id.x;
                var p = particlesA.particles[idx];
                var acc = vec3<f32>(0.0, 0.0, 0.0);

                for (var i = 0u; i < arrayLength(&particlesA.particles); i = i + 1u) {
                    if (i == idx) {
                        continue;
                    }
                    let other = particlesA.particles[i];
                    let r = other.pos.xyz - p.pos.xyz;
                    let dist_sq = dot(r, r) + softening;
                    let inv_dist = 1.0 / sqrt(dist_sq);
                    let inv_dist_cube = inv_dist * inv_dist * inv_dist;
                    acc = acc + sim_params.x * other.pos.w * r * inv_dist_cube;
                }
                
                let vel_xyz = p.vel.xyz + acc * sim_params.y;
                let pos_xyz = p.pos.xyz + vel_xyz * sim_params.y;
                
                p.pos = vec4<f32>(pos_xyz, p.pos.w);
                p.vel = vec4<f32>(vel_xyz, p.vel.w);

                particlesB.particles[idx] = p;
            }
        `,
    });

    const renderShaderModule = device.createShaderModule({
        code: `
            struct VSOutput {
                @builtin(position) pos: vec4<f32>,
                @location(0) uv: vec2<f32>,
            };

            // [size, zoom, rotX, rotY, near, far, cameraZ]
            struct RenderParams {
                params0: vec4<f32>, // [size, zoom, rotX, rotY]
                params1: vec4<f32>, // [near, far, cameraZ, unused]
            }
            @group(0) @binding(0) var<uniform> render_params: RenderParams;
            fn get_size() -> f32 { return render_params.params0.x; }
            fn get_zoom() -> f32 { return render_params.params0.y; }
            fn get_rotX() -> f32 { return render_params.params0.z; }
            fn get_rotY() -> f32 { return render_params.params0.w; }
            fn get_near() -> f32 { return render_params.params1.x; }
            fn get_far() -> f32 { return render_params.params1.y; }
            fn get_cameraZ() -> f32 { return render_params.params1.z; }

            fn rotateY(v: vec3<f32>, angle: f32) -> vec3<f32> {
                let c = cos(angle);
                let s = sin(angle);
                return vec3<f32>(c*v.x + s*v.z, v.y, -s*v.x + c*v.z);
            }
            fn rotateX(v: vec3<f32>, angle: f32) -> vec3<f32> {
                let c = cos(angle);
                let s = sin(angle);
                return vec3<f32>(v.x, c*v.y - s*v.z, s*v.y + c*v.z);
            }

            fn perspective_project(pos: vec3<f32>, near: f32, far: f32, cameraZ: f32) -> vec4<f32> {
                // Simple perspective projection
                let fovY = 1.0; // ~53 degrees
                let aspect = 1.0; // You can pass aspect as a uniform if needed
                let f = 1.0 / tan(fovY * 0.5);
                let nf = 1.0 / (near - far);
                let x = pos.x * f / aspect;
                let y = pos.y * f;
                let z = (pos.z - cameraZ);
                let projZ = (far + near) * nf * z + (2.0 * far * near) * nf;
                return vec4<f32>(x, y, projZ, -z);
            }

            @vertex
            fn vs_main(@location(0) particle_pos: vec3<f32>, @location(1) quad_pos: vec2<f32>) -> VSOutput {
                var out: VSOutput;
                var pos = particle_pos;
                // Rotate relative to screen axes: Y first (horizontal drag), then X (vertical drag)
                pos = rotateY(pos, get_rotY()); // horizontal drag, screen Y axis
                pos = rotateX(pos, -get_rotX()); // vertical drag, screen X axis, reversed
                pos *= get_zoom(); // zoom
                let near = get_near();
                let far = get_far();
                let cameraZ = get_cameraZ();
                let worldPos = pos + vec3<f32>(quad_pos * get_size(), 0.0);
                out.pos = perspective_project(worldPos, near, far, cameraZ);
                out.uv = quad_pos * 0.5 + 0.5;
                return out;
            }

            @fragment
            fn fs_main(@location(0) uv: vec2<f32>) -> @location(0) vec4<f32> {
                let dist = distance(uv, vec2<f32>(0.5, 0.5));
                var alpha: f32;
                if (dist < 0.06) {
                    alpha = 1.0;
                } else if (dist < 0.08) {
                    alpha = mix(1.0, 0.15, (dist - 0.06) / 0.02);
                } else if (dist < 0.5) {
                    alpha = mix(0.15, 0.0, (dist - 0.08) / 0.42);
                } else {
                    alpha = 0.0;
                }
                return vec4<f32>(0.32, 0.4, 1.0, alpha);
            }
        `
    });

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
            module: computeShaderModule,
            entryPoint: 'main',
        },
    });

    let renderPipeline, renderBindGroup;
    function recreateRenderPipeline() {
        renderPipeline = device.createRenderPipeline({
            layout: 'auto',
            vertex: {
                module: renderShaderModule,
                entryPoint: 'vs_main',
                buffers: [
                    {
                        arrayStride: 32,
                        stepMode: 'instance',
                        attributes: [{
                            shaderLocation: 0,
                            offset: 0,
                            format: 'float32x3',
                        }],
                    },
                    {
                        arrayStride: 8,
                        stepMode: 'vertex',
                        attributes: [{
                            shaderLocation: 1,
                            offset: 0,
                            format: 'float32x2',
                        }],
                    }
                ],
            },
            fragment: {
                module: renderShaderModule,
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
                            srcFactor: 'one',
                            dstFactor: 'one',
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

    // Add control panel at bottom center
    const controlPanel = document.createElement('div');
    controlPanel.style.position = 'fixed';
    controlPanel.style.left = '50%';
    controlPanel.style.bottom = '32px';
    controlPanel.style.transform = 'translateX(-50%)';
    controlPanel.style.zIndex = '20';
    controlPanel.style.background = 'rgba(0,0,0,0.7)';
    controlPanel.style.color = 'white';
    controlPanel.style.padding = '18px 32px';
    controlPanel.style.borderRadius = '16px';
    controlPanel.style.fontFamily = 'sans-serif';
    controlPanel.style.display = 'flex';
    controlPanel.style.flexDirection = 'row';
    controlPanel.style.gap = '32px';

    controlPanel.innerHTML = `
      <label style="display:flex;flex-direction:column;align-items:center;gap:6px;">
        <span>Gravity (G): <span id="g-value">30</span></span>
        <input type="range" id="g-slider" min="1" max="100" step="1" value="30" style="width:160px;">
      </label>
      <label style="display:flex;flex-direction:column;align-items:center;gap:6px;">
        <span>Time Step (dt): <span id="dt-value">30</span></span>
        <input type="range" id="dt-slider" min="1" max="100" step="1" value="30" style="width:160px;">
      </label>
      <label style="display:flex;flex-direction:column;align-items:center;gap:6px;">
        <span>Star Count: <span id="star-value">${params.numParticles}</span></span>
        <input type="range" id="star-slider" min="100" max="30000" step="100" value="${params.numParticles}" style="width:160px;">
      </label>
    `;
    document.body.appendChild(controlPanel);

    // Control panel slider logic
    const gSlider = controlPanel.querySelector('#g-slider');
    const gValue = controlPanel.querySelector('#g-value');
    const dtSlider = controlPanel.querySelector('#dt-slider');
    const dtValue = controlPanel.querySelector('#dt-value');
    const starSlider = controlPanel.querySelector('#star-slider');
    const starValue = controlPanel.querySelector('#star-value');

    gSlider.addEventListener('input', () => {
        const gSliderValue = parseInt(gSlider.value);
        // Gravity range: min 0.000001 (slider 1), max 0.01 (slider 100)
        params.G = gSliderValue * 0.000001 * 10;
        gValue.textContent = gSliderValue;
        device.queue.writeBuffer(simParamsBuffer, 0, new Float32Array([params.G, params.dt]));
    });
    dtSlider.addEventListener('input', () => {
        const dtSliderValue = parseInt(dtSlider.value);
        params.dt = dtSliderValue * 0.00001;
        dtValue.textContent = dtSliderValue;
        device.queue.writeBuffer(simParamsBuffer, 0, new Float32Array([params.G, params.dt]));
    });
    starSlider.addEventListener('change', () => {
        params.numParticles = parseInt(starSlider.value);
        starValue.textContent = params.numParticles;
        // Re-initialize particles and buffers
        const newParticleData = new Float32Array(params.numParticles * 8);
        for (let i = 0; i < params.numParticles; i++) {
            let u = Math.random();
            let v = Math.random();
            let w = Math.random();
            let theta = 2 * Math.PI * u;
            let phi = Math.acos(2 * v - 1);
            let r = Math.cbrt(w) * 0.9;
            let x = r * Math.sin(phi) * Math.cos(theta);
            let y = r * Math.sin(phi) * Math.sin(theta);
            let z = r * Math.cos(phi);
            const mass = 0.5 + Math.random() * 99.5;
            newParticleData[i * 8 + 0] = x;
            newParticleData[i * 8 + 1] = y;
            newParticleData[i * 8 + 2] = z;
            newParticleData[i * 8 + 3] = mass;
            newParticleData[i * 8 + 4] = 0;
            newParticleData[i * 8 + 5] = 0;
            newParticleData[i * 8 + 6] = 0;
            newParticleData[i * 8 + 7] = 0;
        }
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
        // Reset frame counter to avoid buffer mismatch
        t = 0;
    });

    const fpsCounter = document.getElementById('fps-counter');
    let lastTime = performance.now();
    let frameCount = 0;

    let t = 0;
    let prevFrameTime = performance.now();
    function frame() {
        const currentTime = performance.now();
        frameCount++;
        if (currentTime - lastTime >= 1000) {
            fpsCounter.textContent = `FPS: ${frameCount}`;
            frameCount = 0;
            lastTime = currentTime;
        }

        // Calculate frame time in seconds
        const frameTimeSec = Math.max((currentTime - prevFrameTime) * 0.001, 0.001); // avoid zero
        prevFrameTime = currentTime;

        // Update camera params
        renderParamsData[1] = zoom;
        renderParamsData[2] = rotX;
        renderParamsData[3] = rotY;
        // near, far remain constant
        renderParamsData[6] = cameraZ;
        device.queue.writeBuffer(renderParamsBuffer, 0, renderParamsData);

        // Scale dt by frame time for consistent simulation speed
        const effectiveDt = params.dt * (frameTimeSec / (1/60)); // normalized to 60fps
        device.queue.writeBuffer(simParamsBuffer, 0, new Float32Array([params.G, effectiveDt]));

        const commandEncoder = device.createCommandEncoder();

        // Compute pass
        const computePass = commandEncoder.beginComputePass();
        computePass.setPipeline(computePipeline);
        const computeBindGroup = device.createBindGroup({
            layout: computePipeline.getBindGroupLayout(0),
            entries: [
                { binding: 0, resource: { buffer: particleBuffers[t % 2] } },
                { binding: 1, resource: { buffer: particleBuffers[(t + 1) % 2] } },
                { binding: 2, resource: { buffer: simParamsBuffer } },
            ],
        });
        computePass.setBindGroup(0, computeBindGroup);
        computePass.dispatchWorkgroups(Math.ceil(params.numParticles / 64));
        computePass.end();

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
        t++;
        requestAnimationFrame(frame);
    }

    requestAnimationFrame(frame);
}

main();
