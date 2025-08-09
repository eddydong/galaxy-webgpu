export function computeShaderModule(device, workgroupSize = 64){
    return device.createShaderModule({
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
            // Single global atomic max (stores bit pattern of largest positive distance)
            @group(0) @binding(3) var<storage, read_write> globalMaxBits: atomic<u32>;

            const softening = 0.01;

            // Shared-memory tiling: reduces global memory reads from O(N) per particle to O(N / tileSize)
            var<workgroup> tilePosMass: array<vec4<f32>, ${workgroupSize}>; // pos.xyz, mass

            @compute @workgroup_size(${/* embed chosen size */ ''}${workgroupSize})
            fn main(
                @builtin(global_invocation_id) global_id: vec3<u32>,
                @builtin(local_invocation_id) local_id: vec3<u32>
            ) {
                let idx = global_id.x;
                let total = arrayLength(&particlesA.particles);
                let isActive = idx < total;
                // Initialize current particle data only if active; provide fallbacks otherwise
                var p: Particle;
                if (isActive) { p = particlesA.particles[idx]; }
                var acc = vec3<f32>(0.0, 0.0, 0.0);
                var selfPos: vec3<f32>;
                if (isActive) { selfPos = p.pos.xyz; } else { selfPos = vec3<f32>(0.0,0.0,0.0); }
                // Iterate over tiles uniformly for all threads (no early return)
                var base: u32 = 0u;
                loop {
                    if (base >= total) { break; }
                    let loadIndex = base + local_id.x;
                    if (loadIndex < total) {
                        tilePosMass[local_id.x] = particlesA.particles[loadIndex].pos; // pos.w = mass
                    } else {
                        tilePosMass[local_id.x] = vec4<f32>(0.0);
                    }
                    workgroupBarrier();
                    let limit = min(u32(${workgroupSize}), total - base);
                    if (isActive) {
                        for (var j: u32 = 0u; j < limit; j = j + 1u) {
                            let globalJ = base + j;
                            if (globalJ == idx) { continue; }
                            let otherPosMass = tilePosMass[j];
                            let r = otherPosMass.xyz - selfPos;
                            let dist_sq = dot(r, r) + softening;
                            let inv_dist = inverseSqrt(dist_sq);
                            let inv_dist2 = inv_dist * inv_dist;
                            let inv_dist3 = inv_dist2 * inv_dist;
                            acc = acc + sim_params.x * otherPosMass.w * r * inv_dist3;
                        }
                    }
                    workgroupBarrier();
                    base = base + u32(${workgroupSize});
                }
                if (isActive) {
                    let vel_xyz = p.vel.xyz + acc * sim_params.y;
                    let pos_xyz = selfPos + vel_xyz * sim_params.y;
                    p.pos = vec4<f32>(pos_xyz, p.pos.w);
                    p.vel = vec4<f32>(vel_xyz, p.vel.w);
                    particlesB.particles[idx] = p;
                    let dist = length(pos_xyz);
                    let candidate = bitcast<u32>(dist);
                    loop {
                        let old = atomicLoad(&globalMaxBits);
                        if (candidate <= old) { break; }
                        let result = atomicCompareExchangeWeak(&globalMaxBits, old, candidate);
                        if (result.exchanged) { break; }
                    }
                }
            }
        `,
    });
}

export function renderShaderModule(device){
    return device.createShaderModule({
        code: `
            struct VSOutput {
                @builtin(position) pos: vec4<f32>,
                @location(0) uv: vec2<f32>,
                @location(1) color: vec3<f32>,
            };

            // [zoom, rotX, rotY, unused, near, far, cameraZ, aspect]
            // Render params with view rotation matrix rows
            // [zoom, unused0, unused1, unused2]
            // [near, far, cameraZ, aspect]
            // viewRow0, viewRow1, viewRow2, viewRow3 (mat4, row-major, last row typically 0,0,0,1)
            struct RenderParams {
                params0: vec4<f32>,
                params1: vec4<f32>,
                viewRow0: vec4<f32>,
                viewRow1: vec4<f32>,
                viewRow2: vec4<f32>,
                viewRow3: vec4<f32>,
            };
            @group(0) @binding(0) var<uniform> render_params: RenderParams;
            fn get_zoom() -> f32 { return render_params.params0.x; }
            fn get_viewport_height() -> f32 { return render_params.params0.y; }
            fn get_near() -> f32 { return render_params.params1.x; }
            fn get_far() -> f32 { return render_params.params1.y; }
            fn get_cameraZ() -> f32 { return render_params.params1.z; }
            fn get_aspect() -> f32 { return render_params.params1.w; }
            fn apply_view(v: vec3<f32>) -> vec3<f32> {
                return vec3<f32>(
                    dot(render_params.viewRow0.xyz, v),
                    dot(render_params.viewRow1.xyz, v),
                    dot(render_params.viewRow2.xyz, v)
                );
            }

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
                let aspect = get_aspect();
                let f = 1.0 / tan(fovY * 0.5);
                let nf = 1.0 / (near - far);
                let x = pos.x * f / aspect;
                let y = pos.y * f;
                let z = (pos.z - cameraZ);
                let projZ = (far + near) * nf * z + (2.0 * far * near) * nf;
                return vec4<f32>(x, y, projZ, -z);
            }

            @vertex
            fn vs_main(
                @location(0) particle_pos: vec3<f32>, 
                @location(2) quad_pos: vec2<f32>,
                @location(3) particle_vel: vec4<f32>
            ) -> VSOutput {
                var out: VSOutput;
                var pos = particle_pos;
                // Rotate relative to screen axes: X first (vertical drag), then Y (horizontal drag)
                // Apply view rotation matrix (screen-space controlled orientation)
                pos = apply_view(pos);
                // Camera dolly zoom: we no longer scale world positions; zoom affects only cameraZ in JS
                let near = get_near();
                let far = get_far();
                let cameraZ = get_cameraZ();
                let base_radius = particle_vel.w;
                // Depth relative to camera (camera looks down -Z after view rotation)
                let depth = cameraZ - pos.z; // positive in front
                let safeDepth = max(depth, 0.001);
                // Mirror perspective constants (must match perspective_project)
                let fovY = 1.0;
                let f = 1.0 / tan(fovY * 0.5);
                // Projected NDC radius then pixel radius
                let r_ndc = (base_radius * f) / safeDepth;
                let pixelRadius = r_ndc * (get_viewport_height() * 0.5);
                // LOD & clamp
                var usePoint = pixelRadius < 1.5;
                // Size tuning: reduce maximum on-screen radius to avoid oversized bright blobs
                let clampedPixelRadius = min(pixelRadius, 16.0); // was 24.0
                let scaled_ndc = clampedPixelRadius / (get_viewport_height() * 0.5);
                let scaled_world = scaled_ndc * safeDepth / f * 0.75; // shrink overall size (0.75 factor)
                var worldPos: vec3<f32>;
                if (usePoint) {
                    worldPos = pos; // collapse quad
                } else {
                    worldPos = pos + vec3<f32>(quad_pos * scaled_world, 0.0);
                }
                out.pos = perspective_project(worldPos, near, far, cameraZ);
                if (usePoint) {
                    out.uv = vec2<f32>(0.5, 0.5);
                } else {
                    out.uv = quad_pos * 0.5 + 0.5;
                }

                // Calculate color based on velocity
                let speed = length(particle_vel.xyz);
                let speed_t = smoothstep(0.0, 20.0, speed); // normalize speed to 0-1

                // Star color spectrum: red -> yellow -> green -> blue -> purple
                var color: vec3<f32>;
                if (speed_t < 0.15) { // Red to Yellow
                    let t = speed_t / 0.15;
                    color = mix(vec3<f32>(1.0, 0.2, 0.0), vec3<f32>(0.9, 0.8, 0.0), t);
                } else if (speed_t < 0.3) { // Yellow to Green
                    let t = (speed_t - 0.15) / 0.15;
                    color = mix(vec3<f32>(0.9, 0.8, 0.0), vec3<f32>(0.0, 0.8, 0.4), t);
                } else if (speed_t < 0.45) { // Green to Blue
                    let t = (speed_t - 0.3) / 0.15;
                    color = mix(vec3<f32>(0.0, 0.8, 0.4), vec3<f32>(0.1, 0.1, 1.0), t);
                } else { // Blue to UV Purple
                    let t = (speed_t - 0.9) / 0.05;
                    color = mix(vec3<f32>(0.1, 0.1, 1.0), vec3<f32>(0.2, 0.1, 0.8), t);
                }
                out.color = color;

                return out;
            }

            @fragment
            fn fs_main(@location(0) uv: vec2<f32>, @location(1) color: vec3<f32>) -> @location(0) vec4<f32> {
                let dist = distance(uv, vec2<f32>(0.5, 0.5));
                if (dist > 0.5) { discard; } // early reject to cut overdraw
                // Simplified falloff: strong core, linear fade
                var alpha: f32;
                if (dist < 0.04) {
                    alpha = 1.0;
                } else {
                    // Faster fade (end at 0.4 instead of 0.5 effective) for less bloom stacking
                    alpha = max(0.0, 1.0 - (dist - 0.04) / 0.36);
                }
                // Dim overall brightness to reduce additive blowout
                let finalColor = color * 0.65;
                return vec4<f32>(finalColor, alpha * 0.85); // slightly reduce alpha too
            }
        `
    });
}
