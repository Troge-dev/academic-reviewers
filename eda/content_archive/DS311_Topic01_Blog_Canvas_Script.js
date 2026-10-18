// Standalone JavaScript Logic for Anscombe & Leverage Canvas Sandbox
// Reusable across any future layout or web framework.

const canvas = document.getElementById('baker-canvas');
let ctx = canvas ? canvas.getContext('2d') : null;

const ANSCOMBE_PRESETS = {
  anscombe1: [
    {x: 10.0, y: 8.04}, {x: 8.0, y: 6.95}, {x: 13.0, y: 7.58}, {x: 9.0, y: 8.81},
    {x: 11.0, y: 8.33}, {x: 14.0, y: 9.96}, {x: 6.0, y: 7.24}, {x: 4.0, y: 4.26},
    {x: 12.0, y: 10.84}, {x: 7.0, y: 4.82}, {x: 5.0, y: 5.68}
  ],
  anscombe2: [
    {x: 10.0, y: 9.14}, {x: 8.0, y: 8.14}, {x: 13.0, y: 8.74}, {x: 9.0, y: 8.77},
    {x: 11.0, y: 9.26}, {x: 14.0, y: 8.10}, {x: 6.0, y: 6.13}, {x: 4.0, y: 3.10},
    {x: 12.0, y: 9.13}, {x: 7.0, y: 7.26}, {x: 5.0, y: 4.74}
  ],
  anscombe3: [
    {x: 10.0, y: 7.46}, {x: 8.0, y: 6.77}, {x: 13.0, y: 12.74}, {x: 9.0, y: 7.11},
    {x: 11.0, y: 7.81}, {x: 14.0, y: 8.84}, {x: 6.0, y: 6.08}, {x: 4.0, y: 5.39},
    {x: 12.0, y: 8.15}, {x: 7.0, y: 6.42}, {x: 5.0, y: 5.73}
  ],
  anscombe4: [
    {x: 8.0, y: 6.58}, {x: 8.0, y: 5.76}, {x: 8.0, y: 7.71}, {x: 8.0, y: 8.84},
    {x: 8.0, y: 8.47}, {x: 8.0, y: 7.04}, {x: 8.0, y: 5.25}, {x: 8.0, y: 5.56},
    {x: 8.0, y: 7.91}, {x: 8.0, y: 6.89}, {x: 19.0, y: 12.50}
  ]
};

let currentPoints = JSON.parse(JSON.stringify(ANSCOMBE_PRESETS.anscombe1));

const X_MIN = 2, X_MAX = 20, Y_MIN = 2, Y_MAX = 14;

function initCanvasSize() {
  if (!canvas) return;
  const rect = canvas.getBoundingClientRect();
  canvas.width = rect.width * window.devicePixelRatio;
  canvas.height = rect.height * window.devicePixelRatio;
  ctx = canvas.getContext('2d');
  ctx.scale(window.devicePixelRatio, window.devicePixelRatio);
  redraw();
}

function dataToPixel(x, y) {
  const rect = canvas.getBoundingClientRect();
  const padLeft = 45, padRight = 25, padTop = 25, padBottom = 40;
  const w = rect.width - padLeft - padRight;
  const h = rect.height - padTop - padBottom;
  const px = padLeft + ((x - X_MIN) / (X_MAX - X_MIN)) * w;
  const py = padTop + ((Y_MAX - y) / (Y_MAX - Y_MIN)) * h;
  return { px, py };
}

function pixelToData(px, py) {
  const rect = canvas.getBoundingClientRect();
  const padLeft = 45, padRight = 25, padTop = 25, padBottom = 40;
  const w = rect.width - padLeft - padRight;
  const h = rect.height - padTop - padBottom;
  const x = X_MIN + ((px - padLeft) / w) * (X_MAX - X_MIN);
  const y = Y_MAX - ((py - padTop) / h) * (Y_MAX - Y_MIN);
  return { x: Math.max(X_MIN, Math.min(X_MAX, x)), y: Math.max(Y_MIN, Math.min(Y_MAX, y)) };
}

function computeStats(pts) {
  const n = pts.length;
  if (n < 2) return { n, r: 0, slope: 0, intercept: 0, r2: 0 };
  let sx = 0, sy = 0, sxx = 0, syy = 0, sxy = 0;
  for (const p of pts) {
    sx += p.x; sy += p.y;
    sxx += p.x * p.x; syy += p.y * p.y;
    sxy += p.x * p.y;
  }
  const meanX = sx / n, meanY = sy / n;
  const num = sxy - n * meanX * meanY;
  const denX = sxx - n * meanX * meanX;
  const denY = syy - n * meanY * meanY;
  const slope = denX !== 0 ? num / denX : 0;
  const intercept = meanY - slope * meanX;
  const denom = Math.sqrt(Math.max(0, denX * denY));
  const r = denom !== 0 ? num / denom : 0;
  const r2 = r * r;
  return { n, r, slope, intercept, r2 };
}

function redraw() {
  if (!canvas || !ctx) return;
  const rect = canvas.getBoundingClientRect();
  const width = rect.width;
  const height = rect.height;

  ctx.clearRect(0, 0, width, height);

  // Background
  ctx.fillStyle = '#FFFFFF';
  ctx.fillRect(0, 0, width, height);

  // Grid
  ctx.strokeStyle = '#EBE5DE';
  ctx.lineWidth = 1;
  for (let x = 4; x <= 18; x += 2) {
    const p1 = dataToPixel(x, Y_MIN);
    const p2 = dataToPixel(x, Y_MAX);
    ctx.beginPath();
    ctx.moveTo(p1.px, p1.py);
    ctx.lineTo(p2.px, p2.py);
    ctx.stroke();

    ctx.fillStyle = '#827C75';
    ctx.font = '10px "JetBrains Mono", monospace';
    ctx.textAlign = 'center';
    ctx.fillText(x, p1.px, height - 15);
  }

  for (let y = 4; y <= 12; y += 2) {
    const p1 = dataToPixel(X_MIN, y);
    const p2 = dataToPixel(X_MAX, y);
    ctx.beginPath();
    ctx.moveTo(p1.px, p1.py);
    ctx.lineTo(p2.px, p2.py);
    ctx.stroke();

    ctx.fillStyle = '#827C75';
    ctx.font = '10px "JetBrains Mono", monospace';
    ctx.textAlign = 'right';
    ctx.fillText(y, 35, p1.py + 3);
  }

  const stats = computeStats(currentPoints);

  // OLS Line
  if (currentPoints.length >= 2) {
    const xLeft = X_MIN;
    const yLeft = stats.slope * xLeft + stats.intercept;
    const xRight = X_MAX;
    const yRight = stats.slope * xRight + stats.intercept;

    const pLeft = dataToPixel(xLeft, yLeft);
    const pRight = dataToPixel(xRight, yRight);

    ctx.strokeStyle = '#5A822B';
    ctx.lineWidth = 2.5;
    ctx.beginPath();
    ctx.moveTo(pLeft.px, pLeft.py);
    ctx.lineTo(pRight.px, pRight.py);
    ctx.stroke();
  }

  // Points
  for (const p of currentPoints) {
    const { px, py } = dataToPixel(p.x, p.y);
    ctx.beginPath();
    ctx.arc(px, py, 6, 0, Math.PI * 2);
    ctx.fillStyle = '#D56638';
    ctx.fill();
    ctx.lineWidth = 1.8;
    ctx.strokeStyle = '#2A2825';
    ctx.stroke();
  }

  // Update Scorecards
  const statN = document.getElementById('stat-n');
  const statR = document.getElementById('stat-r');
  const statSlope = document.getElementById('stat-slope');
  const statR2 = document.getElementById('stat-r2');

  if (statN) statN.textContent = stats.n;
  if (statR) statR.textContent = (stats.r >= 0 ? '+' : '') + stats.r.toFixed(3);
  if (statSlope) statSlope.textContent = stats.slope.toFixed(3);
  if (statR2) statR2.textContent = (stats.r2 * 100).toFixed(1) + '%';
}

function loadCanvasPreset(presetName) {
  if (!ANSCOMBE_PRESETS[presetName]) return;
  currentPoints = JSON.parse(JSON.stringify(ANSCOMBE_PRESETS[presetName]));
  document.querySelectorAll('.btn-baker').forEach(b => b.classList.remove('active'));
  const btn = event ? event.target : null;
  if (btn) btn.classList.add('active');
  redraw();
}

function clearCanvas() {
  currentPoints = [];
  document.querySelectorAll('.btn-baker').forEach(b => b.classList.remove('active'));
  redraw();
}

if (canvas) {
  canvas.addEventListener('click', (e) => {
    const rect = canvas.getBoundingClientRect();
    const px = e.clientX - rect.left;
    const py = e.clientY - rect.top;
    const dataCoord = pixelToData(px, py);
    currentPoints.push({ x: parseFloat(dataCoord.x.toFixed(2)), y: parseFloat(dataCoord.y.toFixed(2)) });
    redraw();
  });
  window.addEventListener('resize', initCanvasSize);
  window.addEventListener('DOMContentLoaded', initCanvasSize);
}
