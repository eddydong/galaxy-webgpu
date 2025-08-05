async function main() {
    const canvas = document.getElementById('webgpu-canvas');

    // Add blend mode toggle UI
    const blendSwitch = document.createElement('label');
    blendSwitch.style.position = 'fixed';
    blendSwitch.style.top = '16px';
    blendSwitch.style.left = '16px';
    blendSwitch.style.zIndex = '10';
    blendSwitch.style.background = 'rgba(0,0,0,0.5)';
    blendSwitch.style.color = 'white';
    blendSwitch.style.padding = '6px 12px';
    blendSwitch.style.borderRadius = '8px';
    blendSwitch.style.fontFamily = 'sans-serif';
    blendSwitch.innerHTML = `<input type="checkbox" id="blend-toggle" checked> Lighter Blend`;
    document.body.appendChild(blendSwitch);

    const blendToggle = blendSwitch.querySelector('#blend-toggle');
    blendToggle.addEventListener('change', () => {
        useLighterBlend = blendToggle.checked;
        recreateRenderPipeline();
    });

    let useLighterBlend = true;

    // Camera controls
    let zoom = 0.5; // slightly zoomed out
    let rotX = Math.PI; // tilt for better perspective
    let rotY = Math.PI; // rotate around Y axis for better view
    let dragging = false;
    let lastMouseX = 0;
    let lastMouseY = 0;

    const params = {
        numParticles: 20000,
        particleSize: 0.025,
        G: 0.00001, 
        dt: 0.0001,
    };    

    // [particleSize, zoom, rotX, rotY, near, far, cameraZ]
    const near = 0.01;
    const far = 100.0;
    const cameraZ = 2.0;

    canvas.addEventListener('wheel', (e) => {
        e.preventDefault();
        zoom *= Math.exp(-e.deltaY * 0.001);
        zoom = Math.max(0.1, Math.min(zoom, 10.0));
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
            rotX += dx * 0.005; // left/right drag rotates X
            rotY += dy * 0.005; // up/down drag rotates Y
            rotX = Math.max(-Math.PI/2, Math.min(Math.PI/2, rotX));
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
        const r = Math.random() * 0.9;
        const theta = Math.random() * 2 * Math.PI;
        const z = (Math.random() - 0.5) * 1.0;

        // Assign random mass between 0.5 and 5.0
        const mass = 0.5 + Math.random() * 99.5;

        // Position
        particleAndVelocityData[i * 8 + 0] = r * Math.cos(theta);
        particleAndVelocityData[i * 8 + 1] = r * Math.sin(theta);
        particleAndVelocityData[i * 8 + 2] = z;
        particleAndVelocityData[i * 8 + 3] = mass;

        // Initial velocity for orbit (adjusted for mass)
        const speed = Math.sqrt(0.1 / (r + 0.1));
        particleAndVelocityData[i * 8 + 4] = -speed * Math.sin(theta);
        particleAndVelocityData[i * 8 + 5] = speed * Math.cos(theta);
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

    const renderParamsData = new Float32Array([params.particleSize, zoom, rotX, rotY, near, far, cameraZ]);
    const renderParamsBuffer = device.createBuffer({
        size: renderParamsData.byteLength,
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

            @group(0) @binding(0) var<uniform> render_params: vec4<f32>; // [size, zoom, rotX, rotY]

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

            @vertex
            fn vs_main(@location(0) particle_pos: vec3<f32>, @location(1) quad_pos: vec2<f32>) -> VSOutput {
                var out: VSOutput;
                var pos = particle_pos;
                pos = rotateY(pos, render_params.z);
                pos = rotateX(pos, render_params.y);
                pos *= render_params.w;
                out.pos = vec4<f32>(pos + vec3<f32>(quad_pos * render_params.x, 0.0), 1.0);
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
                    alpha = mix(0.12, 0.0, (dist - 0.08) / 0.42);
                } else {
                    alpha = 0.0;
                }
                return vec4<f32>(0.32, 0.4, 1.0, alpha);
            }
        `,
    });

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

    const fpsCounter = document.getElementById('fps-counter');
    let lastTime = performance.now();
    let frameCount = 0;

    let t = 0;
    function frame() {
        const currentTime = performance.now();
        frameCount++;
        if (currentTime - lastTime >= 1000) {
            fpsCounter.textContent = `FPS: ${frameCount}`;
            frameCount = 0;
            lastTime = currentTime;
        }

        // Update camera params
        renderParamsData[1] = zoom;
        renderParamsData[2] = rotX;
        renderParamsData[3] = rotY;
        // near, far, cameraZ remain constant
        device.queue.writeBuffer(renderParamsBuffer, 0, renderParamsData);

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
        // ...existing code...
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
