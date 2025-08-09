// Minimal subset of gl-matrix needed by this project (mat4, vec3, quat)
// Implemented from standard quaternion/vector/matrix math (conceptual parity with gl-matrix, trimmed for size).

export const vec3 = {
  fromValues(x, y, z) { return new Float32Array([x, y, z]); },
  set(out, x, y, z) { out[0] = x; out[1] = y; out[2] = z; return out; },
  normalize(out, a) {
    let x = a[0], y = a[1], z = a[2];
    let len = Math.hypot(x, y, z);
    if (len > 0) { len = 1 / len; out[0] = x * len; out[1] = y * len; out[2] = z * len; }
    else { out[0] = 0; out[1] = 0; out[2] = 0; }
    return out;
  }
};

export const quat = {
  create() { return new Float32Array([0, 0, 0, 1]); },
  setAxisAngle(out, axis, rad) {
    let x = axis[0], y = axis[1], z = axis[2];
    let len = Math.hypot(x, y, z);
    if (len === 0) { out[0] = out[1] = out[2] = 0; out[3] = 1; return out; }
    len = 1 / len; x *= len; y *= len; z *= len;
    const half = rad * 0.5; const s = Math.sin(half); const c = Math.cos(half);
    out[0] = x * s; out[1] = y * s; out[2] = z * s; out[3] = c;
    return out;
  }
};

export const mat4 = {
  create() { const out = new Float32Array(16); out[0] = out[5] = out[10] = out[15] = 1; return out; },
  fromQuat(out, q) {
    const x = q[0], y = q[1], z = q[2], w = q[3];
    const x2 = x + x, y2 = y + y, z2 = z + z;
    const xx = x * x2, xy = x * y2, xz = x * z2;
    const yy = y * y2, yz = y * z2, zz = z * z2;
    const wx = w * x2, wy = w * y2, wz = w * z2;
    out[0] = 1 - (yy + zz); out[1] = xy + wz;       out[2] = xz - wy;       out[3] = 0;
    out[4] = xy - wz;       out[5] = 1 - (xx + zz); out[6] = yz + wx;       out[7] = 0;
    out[8] = xz + wy;       out[9] = yz - wx;       out[10] = 1 - (xx + yy);out[11] = 0;
    out[12] = 0;            out[13] = 0;            out[14] = 0;            out[15] = 1;
    return out;
  },
  multiply(out, a, b) {
    const a00 = a[0], a01 = a[1], a02 = a[2], a03 = a[3];
    const a10 = a[4], a11 = a[5], a12 = a[6], a13 = a[7];
    const a20 = a[8], a21 = a[9], a22 = a[10], a23 = a[11];
    const a30 = a[12], a31 = a[13], a32 = a[14], a33 = a[15];
    let b0, b1, b2, b3;
    b0 = b[0]; b1 = b[1]; b2 = b[2]; b3 = b[3];
    out[0] = a00*b0 + a10*b1 + a20*b2 + a30*b3;
    out[1] = a01*b0 + a11*b1 + a21*b2 + a31*b3;
    out[2] = a02*b0 + a12*b1 + a22*b2 + a32*b3;
    out[3] = a03*b0 + a13*b1 + a23*b2 + a33*b3;
    b0 = b[4]; b1 = b[5]; b2 = b[6]; b3 = b[7];
    out[4] = a00*b0 + a10*b1 + a20*b2 + a30*b3;
    out[5] = a01*b0 + a11*b1 + a21*b2 + a31*b3;
    out[6] = a02*b0 + a12*b1 + a22*b2 + a32*b3;
    out[7] = a03*b0 + a13*b1 + a23*b2 + a33*b3;
    b0 = b[8]; b1 = b[9]; b2 = b[10]; b3 = b[11];
    out[8] = a00*b0 + a10*b1 + a20*b2 + a30*b3;
    out[9] = a01*b0 + a11*b1 + a21*b2 + a31*b3;
    out[10] = a02*b0 + a12*b1 + a22*b2 + a32*b3;
    out[11] = a03*b0 + a13*b1 + a23*b2 + a33*b3;
    b0 = b[12]; b1 = b[13]; b2 = b[14]; b3 = b[15];
    out[12] = a00*b0 + a10*b1 + a20*b2 + a30*b3;
    out[13] = a01*b0 + a11*b1 + a21*b2 + a31*b3;
    out[14] = a02*b0 + a12*b1 + a22*b2 + a32*b3;
    out[15] = a03*b0 + a13*b1 + a23*b2 + a33*b3;
    return out;
  }
};

export const glMatrix = { mat4, vec3, quat };
