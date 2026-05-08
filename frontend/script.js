'use strict';
/* ═══════════════════════════════════════════════════════════════
   VoiceID — Frontend Script
   
   HIGH-QUALITY RECORDING PIPELINE:
   Browser MediaRecorder → AudioContext → resample to 16 kHz →
   16-bit PCM → WAV blob → POST to /api/record
   
   This guarantees Resemblyzer-optimal input without any lossy
   intermediate format. All conversion happens client-side using
   the Web Audio API, so even if the browser records in opus/webm,
   we always send a clean 16kHz mono WAV to the server.
═══════════════════════════════════════════════════════════════ */

const API = 'http://localhost:5000/api';
const API_URL = 'http://localhost:5000/api';

// ── Global state ─────────────────────────────────────────────────────────────
const state = {
  compareFiles: { slot1: null, slot2: null },  // {filename, filepath} after upload
  trainFiles:   [],   // [{filename, filepath, originalName, size, source}]
};

// ── Recorder instances per context ───────────────────────────────────────────
// Each context gets its own RecorderEngine instance
const recorders = {};

// ── Init ─────────────────────────────────────────────────────────────────────
document.addEventListener('DOMContentLoaded', () => {
  initBackground();
  initNav();
  initSourceToggles();
  initAnalyze();
  initCompare();
  initTrain();
  initIdentify();
  checkHealth();
  loadSpeakers();
  addLogoutButton();
});

// ══════════════════════════════════════════════════════════════
//  HIGH-QUALITY RECORDER ENGINE
//  Captures raw PCM via AudioContext, downsamples to 16kHz,
//  encodes as 16-bit PCM WAV entirely in the browser.
// ══════════════════════════════════════════════════════════════
class RecorderEngine {
  constructor(canvasId, timerId) {
    this.canvas    = document.getElementById(canvasId);
    this.timerEl   = document.getElementById(timerId);
    this.stream    = null;
    this.audioCtx  = null;
    this.processor = null;
    this.analyser  = null;
    this.pcmBufs   = [];       // raw Float32 chunks at native rate
    this.nativeRate = 44100;
    this.isRecording = false;
    this.startTime   = 0;
    this.timerInt    = null;
    this.animFrame   = null;
  }

  async start() {
    // Request mic with best quality constraints
    this.stream = await navigator.mediaDevices.getUserMedia({
      audio: {
        channelCount:     1,
        sampleRate:       { ideal: 48000 },
        echoCancellation: true,
        noiseSuppression: true,
        autoGainControl:  true,
      }
    });

    this.audioCtx  = new (window.AudioContext || window.webkitAudioContext)();
    this.nativeRate = this.audioCtx.sampleRate;

    const source    = this.audioCtx.createMediaStreamSource(this.stream);
    this.analyser   = this.audioCtx.createAnalyser();
    this.analyser.fftSize = 256;

    // ScriptProcessor captures raw PCM (works in all browsers)
    this.processor  = this.audioCtx.createScriptProcessor(4096, 1, 1);
    this.pcmBufs    = [];

    this.processor.onaudioprocess = (e) => {
      const buf = e.inputBuffer.getChannelData(0);
      this.pcmBufs.push(new Float32Array(buf));
    };

    source.connect(this.analyser);
    source.connect(this.processor);
    this.processor.connect(this.audioCtx.destination);

    this.isRecording = true;
    this.startTime   = Date.now();
    this._startTimer();
    this._drawWaveform();
  }

  stop() {
    this.isRecording = false;
    clearInterval(this.timerInt);
    cancelAnimationFrame(this.animFrame);

    if (this.processor) {
      this.processor.disconnect();
      this.processor.onaudioprocess = null;
    }
    if (this.stream) {
      this.stream.getTracks().forEach(t => t.stop());
    }
    if (this.audioCtx && this.audioCtx.state !== 'closed') {
      this.audioCtx.close();
    }
  }

  /**
   * Build a 16kHz 16-bit mono WAV Blob from captured chunks.
   * Uses linear interpolation for resampling (sufficient quality
   * for 48→16kHz; soxr-quality is overkill for voice embeddings).
   */
  getWavBlob() {
    // Concatenate all PCM chunks
    const totalLen  = this.pcmBufs.reduce((s, b) => s + b.length, 0);
    const fullBuf   = new Float32Array(totalLen);
    let offset = 0;
    for (const b of this.pcmBufs) {
      fullBuf.set(b, offset);
      offset += b.length;
    }

    // Downsample to 16000 Hz
    const TARGET_SR = 16000;
    const ratio     = this.nativeRate / TARGET_SR;
    const outLen    = Math.round(totalLen / ratio);
    const resampled = new Float32Array(outLen);

    for (let i = 0; i < outLen; i++) {
      const pos  = i * ratio;
      const idx  = Math.floor(pos);
      const frac = pos - idx;
      const s0   = fullBuf[idx]     || 0;
      const s1   = fullBuf[idx + 1] || 0;
      resampled[i] = s0 + frac * (s1 - s0);   // linear interp
    }

    // RMS normalize to -30 dBFS (matches Resemblyzer hparam audio_norm_target_dBFS)
    const rms = Math.sqrt(resampled.reduce((s, v) => s + v * v, 0) / resampled.length);
    const targetRms = Math.pow(10, -30 / 20);
    const gain = rms > 0 ? targetRms / rms : 1;
    for (let i = 0; i < resampled.length; i++) {
      resampled[i] = Math.max(-1, Math.min(1, resampled[i] * gain));
    }

    // Encode as 16-bit PCM WAV
    return this._encodeWav(resampled, TARGET_SR);
  }

  getDurationSec() {
    return (Date.now() - this.startTime) / 1000;
  }

  _encodeWav(samples, sampleRate) {
    const numSamples = samples.length;
    const buffer     = new ArrayBuffer(44 + numSamples * 2);
    const view       = new DataView(buffer);
    const int16      = new Int16Array(buffer, 44);

    // PCM int16 conversion
    for (let i = 0; i < numSamples; i++) {
      int16[i] = Math.round(samples[i] * 32767);
    }

    // RIFF header
    const write = (off, str) => { for (let i = 0; i < str.length; i++) view.setUint8(off + i, str.charCodeAt(i)); };
    write(0,  'RIFF');
    view.setUint32(4,  36 + numSamples * 2, true);
    write(8,  'WAVE');
    write(12, 'fmt ');
    view.setUint32(16, 16,         true);  // chunk size
    view.setUint16(20, 1,          true);  // PCM
    view.setUint16(22, 1,          true);  // mono
    view.setUint32(24, sampleRate, true);
    view.setUint32(28, sampleRate * 2, true); // byte rate
    view.setUint16(32, 2,          true);  // block align
    view.setUint16(34, 16,         true);  // bits per sample
    write(36, 'data');
    view.setUint32(40, numSamples * 2, true);

    return new Blob([buffer], { type: 'audio/wav' });
  }

  _startTimer() {
    this.timerInt = setInterval(() => {
      if (!this.timerEl) return;
      const s = Math.floor((Date.now() - this.startTime) / 1000);
      this.timerEl.textContent = `${Math.floor(s / 60)}:${String(s % 60).padStart(2, '0')}`;
    }, 500);
  }

  _drawWaveform() {
    if (!this.canvas || !this.analyser) return;
    const ctx  = this.canvas.getContext('2d');
    const data = new Uint8Array(this.analyser.frequencyBinCount);
    const W    = this.canvas.width;
    const H    = this.canvas.height;

    const draw = () => {
      if (!this.isRecording) {
        ctx.clearRect(0, 0, W, H);
        return;
      }
      this.animFrame = requestAnimationFrame(draw);
      this.analyser.getByteTimeDomainData(data);
      ctx.clearRect(0, 0, W, H);

      ctx.strokeStyle = '#00c896';
      ctx.lineWidth   = 1.5;
      ctx.beginPath();
      const step = W / data.length;

      for (let i = 0; i < data.length; i++) {
        const y = ((data[i] - 128) / 128) * (H / 2) + H / 2;
        i === 0 ? ctx.moveTo(0, y) : ctx.lineTo(i * step, y);
      }
      ctx.stroke();

      // RMS energy bar at bottom
      const rms = Math.sqrt(data.reduce((s, v) => s + (v - 128) ** 2, 0) / data.length) / 128;
      ctx.fillStyle = `rgba(0,200,150,${rms * 0.5})`;
      ctx.fillRect(0, H - 4, W * rms * 3, 4);
    };
    draw();
  }
}

// ── Pending WAV blobs for recorded compare/identify slots ───────────────────
const pendingBlobs = {};  // key → Blob

// ── Active recording session (train: accumulates clips) ─────────────────────
let trainRecBlob = null;  // last recorded blob before save

// ══════════════════════════════════════════════════════════════
//  Source toggle (upload ↔ record)
// ══════════════════════════════════════════════════════════════
function initSourceToggles() {
  document.querySelectorAll('.src-btn').forEach(btn => {
    btn.addEventListener('click', () => {
      const target = btn.dataset.target;
      const src    = btn.dataset.src;
      const group  = btn.closest('.source-toggle');
      group.querySelectorAll('.src-btn').forEach(b => b.classList.remove('active'));
      btn.classList.add('active');

      const uploadPanel = document.getElementById(`${target}-upload-panel`);
      const recordPanel = document.getElementById(`${target}-record-panel`);

      if (src === 'upload') {
        uploadPanel?.classList.remove('hidden');
        recordPanel?.classList.add('hidden');
      } else {
        uploadPanel?.classList.add('hidden');
        recordPanel?.classList.remove('hidden');
      }
    });
  });
}

// ══════════════════════════════════════════════════════════════
//  Generic record toggle (used by analyze, identify, compare1/2)
// ══════════════════════════════════════════════════════════════
async function toggleRecord(ctx) {
  if (!recorders[ctx]) recorders[ctx] = new RecorderEngine(`${ctx}-canvas`, `${ctx}-timer`);
  const rec     = recorders[ctx];
  const recBtn  = document.getElementById(`${ctx}-rec-btn`);
  const sendBtn = document.getElementById(`${ctx}-send-btn`);

  if (!rec.isRecording) {
    // Start
    try {
      await rec.start();
      recBtn.classList.add('recording');
      recBtn.innerHTML = '<span class="rec-dot"></span> Arrêter';
      if (sendBtn) sendBtn.classList.add('hidden');
    } catch (err) {
      toast('Microphone inaccessible: ' + err.message, 'error');
    }
  } else {
    // Stop
    rec.stop();
    recBtn.classList.remove('recording');
    recBtn.innerHTML = '<span class="rec-dot"></span> Réenregistrer';

    const dur = rec.getDurationSec();
    if (dur < 1.5) {
      toast('Enregistrement trop court (min 1.5s)', 'warning');
      recBtn.innerHTML = '<span class="rec-dot"></span> Commencer l\'enregistrement';
      return;
    }

    pendingBlobs[ctx] = rec.getWavBlob();
    if (sendBtn) sendBtn.classList.remove('hidden');
    toast(`Enregistrement prêt — ${dur.toFixed(1)}s · WAV 16kHz`, 'success');

    // Reset engine so next click starts fresh
    recorders[ctx] = null;
  }
}

// ── Send recorded blob to server ─────────────────────────────────────────────
async function sendRecording(ctx) {
  const blob = pendingBlobs[ctx];
  if (!blob) { toast('Aucun enregistrement disponible', 'error'); return; }

  if (ctx === 'analyze') {
    await doUploadAnalyze(blob, true);
  } else if (ctx === 'identify') {
    await doIdentifyRecord(blob);
  } else if (ctx === 'compare1' || ctx === 'compare2') {
    await handleCompareRecord(ctx, blob);
  }
  delete pendingBlobs[ctx];
}

// ══════════════════════════════════════════════════════════════
//  Background Particles
// ══════════════════════════════════════════════════════════════
function initBackground() {
  const canvas = document.getElementById('bg-canvas');
  if (!canvas) return;
  const ctx = canvas.getContext('2d');
  let w, h, particles;
  const resize = () => { w = canvas.width = innerWidth; h = canvas.height = innerHeight; };
  const mkP = () => particles = Array.from({length:55}, () => ({
    x: Math.random()*w, y: Math.random()*h,
    vx: (Math.random()-.5)*.3, vy: (Math.random()-.5)*.3,
    r: Math.random()*1.5+.5, a: Math.random()*.4+.1
  }));
  const draw = () => {
    ctx.clearRect(0,0,w,h);
    particles.forEach(p => {
      p.x+=p.vx; p.y+=p.vy;
      if(p.x<0)p.x=w; if(p.x>w)p.x=0;
      if(p.y<0)p.y=h; if(p.y>h)p.y=0;
      ctx.beginPath(); ctx.arc(p.x,p.y,p.r,0,Math.PI*2);
      ctx.fillStyle=`rgba(0,200,150,${p.a})`; ctx.fill();
    });
    for(let i=0;i<particles.length;i++)
      for(let j=i+1;j<particles.length;j++){
        const dx=particles[i].x-particles[j].x, dy=particles[i].y-particles[j].y;
        const d=Math.sqrt(dx*dx+dy*dy);
        if(d<120){ ctx.beginPath(); ctx.moveTo(particles[i].x,particles[i].y);
          ctx.lineTo(particles[j].x,particles[j].y);
          ctx.strokeStyle=`rgba(0,200,150,${.05*(1-d/120)})`; ctx.lineWidth=.5; ctx.stroke(); }
      }
    requestAnimationFrame(draw);
  };
  resize(); mkP(); draw();
  window.addEventListener('resize', () => { resize(); mkP(); });
}

// ══════════════════════════════════════════════════════════════
//  Navigation
// ══════════════════════════════════════════════════════════════
function initNav() {
  const sidebar = document.getElementById('sidebar');
  document.querySelectorAll('.nav-btn').forEach(btn => {
    btn.addEventListener('click', () => {
      document.querySelectorAll('.nav-btn').forEach(b => b.classList.remove('active'));
      btn.classList.add('active');
      document.querySelectorAll('.tab-panel').forEach(p => p.classList.remove('active'));
      document.getElementById(`tab-${btn.dataset.tab}`)?.classList.add('active');
      if (btn.dataset.tab === 'speakers') loadSpeakers();
      if (innerWidth <= 900) sidebar.classList.remove('open');
    });
  });
  document.getElementById('nav-toggle')?.addEventListener('click', () => sidebar.classList.toggle('open'));
  document.addEventListener('click', e => {
    if (innerWidth<=900 && sidebar.classList.contains('open')
        && !sidebar.contains(e.target) && e.target.id!=='nav-toggle')
      sidebar.classList.remove('open');
  });
}

// ══════════════════════════════════════════════════════════════
//  ANALYSE TAB
// ══════════════════════════════════════════════════════════════
function initAnalyze() {
  const zone  = document.getElementById('analyze-dropzone');
  const input = document.getElementById('analyze-input');
  if (zone)  zone.addEventListener('click', () => input.click());
  if (input) input.addEventListener('change', e => { if(e.target.files[0]) doUploadAnalyze(e.target.files[0]); });
  setupDrop(zone, files => doUploadAnalyze(files[0]));
}

async function doUploadAnalyze(fileOrBlob, isRecorded = false) {
  const resultEl = document.getElementById('analyze-result');
  showLoading('Extraction de l\'empreinte vocale...');
  const fd = new FormData();
  fd.append('audio', fileOrBlob, isRecorded ? 'recording.wav' : fileOrBlob.name);

  const endpoint = isRecorded ? `${API}/record` : `${API}/upload`;
  if (isRecorded) fd.append('prefix', 'analyze');

  try {
    const res  = await fetch(endpoint, { method:'POST', body:fd });
    const data = await res.json();
    if (!res.ok) throw new Error(data.error || 'Erreur serveur');

    const name   = isRecorded ? 'Enregistrement.wav' : fileOrBlob.name;
    const idInfo = data.identified_speaker;

    resultEl.className = 'result-card success';
    resultEl.innerHTML = `
      <div class="result-title">
        <svg viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2" width="20" height="20"><polyline points="20 6 9 17 4 12"/></svg>
        Analyse — ${esc(name)} ${isRecorded ? '<span style="font-size:.75rem;color:var(--accent);font-family:\'Space Mono\',monospace">[WAV 16kHz]</span>' : ''}
      </div>
      <div class="stat-grid">
        <div class="stat-item"><div class="stat-label">DURÉE</div><div class="stat-value neutral">${data.duration?.toFixed(2) ?? 'N/A'}s</div></div>
        <div class="stat-item"><div class="stat-label">FRÉQUENCE</div><div class="stat-value neutral">${data.sample_rate ? (data.sample_rate/1000).toFixed(1)+'kHz' : 'N/A'}</div></div>
        <div class="stat-item"><div class="stat-label">EMBEDDING</div><div class="stat-value">${data.embedding_dim||0}D</div></div>
        <div class="stat-item"><div class="stat-label">STATUT</div><div class="stat-value" style="color:${data.embedding_extracted?'var(--accent)':'var(--danger)'}">${data.embedding_extracted?'✓ Extrait':'✗ Échec'}</div></div>
        ${data.rms_energy!=null?`<div class="stat-item"><div class="stat-label">ÉNERGIE RMS</div><div class="stat-value neutral">${data.rms_energy.toFixed(4)}</div></div>`:''}
        ${data.peak_amplitude!=null?`<div class="stat-item"><div class="stat-label">CRÊTE</div><div class="stat-value neutral">${data.peak_amplitude.toFixed(4)}</div></div>`:''}
      </div>
      ${data.embedding_extracted?`<div class="waveform" style="opacity:.7"><span></span><span></span><span></span><span></span><span></span><span></span><span></span></div>`:''}
      ${idInfo?.speaker?`
      <div style="margin-top:1rem;padding-top:1rem;border-top:1px solid rgba(255,255,255,.06)">
        <div class="stat-label" style="margin-bottom:.5rem">LOCUTEUR IDENTIFIÉ</div>
        ${renderIdentifyResult(idInfo)}
      </div>`:''}
    `;
    resultEl.classList.remove('hidden');
    toast('Empreinte extraite ✓', 'success');
  } catch(err) {
    showResultError(resultEl, err.message);
  } finally { hideLoading(); }
}

// ══════════════════════════════════════════════════════════════
//  COMPARE TAB
// ══════════════════════════════════════════════════════════════
function initCompare() {
  setupCompareSlot('slot1-drop','compare1-input','slot1-name','slot1');
  setupCompareSlot('slot2-drop','compare2-input','slot2-name','slot2');
  document.getElementById('compare-btn')?.addEventListener('click', doCompare);
}

function setupCompareSlot(dropId, inputId, nameId, slotKey) {
  const drop  = document.getElementById(dropId);
  const input = document.getElementById(inputId);
  const nameEl= document.getElementById(nameId);
  if (!drop||!input) return;
  drop.addEventListener('click', () => input.click());
  input.addEventListener('change', e => { if(e.target.files[0]) handleCompareFile(e.target.files[0], slotKey, nameEl, drop); });
  setupDrop(drop, files => handleCompareFile(files[0], slotKey, nameEl, drop));
}

function handleCompareFile(file, slotKey, nameEl, dropEl) {
  // store as File; will be uploaded when Compare is clicked
  state.compareFiles[slotKey] = { file, uploaded: false };
  if (nameEl)  nameEl.textContent = file.name;
  if (dropEl)  dropEl.classList.add('ready');
  updateCompareBtn();
}

async function handleCompareRecord(ctx, blob) {
  // Upload the blob right away and store server filename
  const fd = new FormData();
  fd.append('audio', blob, 'recording.wav');
  fd.append('prefix', ctx);
  const res  = await fetch(`${API}/record`, { method:'POST', body:fd });
  const data = await res.json();
  if (!res.ok || !data.filename) { toast('Erreur enregistrement', 'error'); return; }

  const slotKey = ctx === 'compare1' ? 'slot1' : 'slot2';
  const nameEl  = document.getElementById(`${slotKey}-name`);
  state.compareFiles[slotKey] = { filename: data.filename, uploaded: true };
  if (nameEl) nameEl.textContent = `🎤 ${data.filename}`;
  updateCompareBtn();
  toast('Enregistrement prêt pour comparaison', 'success');
}

function updateCompareBtn() {
  const btn = document.getElementById('compare-btn');
  if (btn) btn.disabled = !(state.compareFiles.slot1 && state.compareFiles.slot2);
}

async function doCompare() {
  const s1 = state.compareFiles.slot1;
  const s2 = state.compareFiles.slot2;
  if (!s1 || !s2) return;
  const resultEl = document.getElementById('compare-result');
  showLoading('Comparaison des empreintes neurales...');

  try {
    // Upload any non-yet-uploaded files
    const upload = async (slot) => {
      if (slot.uploaded) return slot.filename;
      const d = await uploadRaw(slot.file);
      return d?.filename;
    };
    const [f1, f2] = await Promise.all([upload(s1), upload(s2)]);
    if (!f1 || !f2) throw new Error('Impossible d\'uploader les fichiers');

    const res  = await fetch(`${API}/compare`, {
      method:'POST', headers:{'Content-Type':'application/json'},
      body: JSON.stringify({ file1:f1, file2:f2 }),
    });
    const data = await res.json();
    if (!res.ok) throw new Error(data.error);

    const pct      = data.percentage ?? (data.similarity * 100).toFixed(1);
    const same     = data.is_same_speaker;
    const barColor = data.similarity > .80 ? '#00c896' : data.similarity > .65 ? '#f59e0b' : '#ef4444';

    resultEl.className = `result-card ${same?'success':'info'}`;
    resultEl.innerHTML = `
      <div class="result-title">
        <svg viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2" width="20" height="20"><path d="M18 20V10M12 20V4M6 20v-6"/></svg>
        Résultat de comparaison
      </div>
      <div class="sim-bar-wrap">
        <div class="sim-bar-label"><span>Similarité cosinus</span><span style="color:var(--accent);font-family:'Space Mono',monospace;font-weight:700">${pct}%</span></div>
        <div class="sim-bar-track"><div class="sim-bar-fill" style="width:0%;background:${barColor}" id="sim-fill"></div></div>
      </div>
      <div class="${same?'verdict-badge verdict-same':'verdict-badge verdict-diff'}">${same?'✓ Même locuteur':'✗ Locuteurs différents'}</div>
      <div class="stat-grid" style="margin-top:1rem">
        <div class="stat-item"><div class="stat-label">CONFIANCE</div><div class="stat-value">${data.confidence||'N/A'}</div></div>
        <div class="stat-item"><div class="stat-label">SEUIL</div><div class="stat-value neutral">75%</div></div>
      </div>
    `;
    resultEl.classList.remove('hidden');
    setTimeout(() => { const f=document.getElementById('sim-fill'); if(f) f.style.width=pct+'%'; }, 50);
    toast('Comparaison terminée', 'success');
  } catch(err) {
    showResultError(resultEl, err.message);
  } finally { hideLoading(); }
}

// ══════════════════════════════════════════════════════════════
//  TRAIN TAB
// ══════════════════════════════════════════════════════════════
function initTrain() {
  const input     = document.getElementById('train-input');
  const zone      = document.getElementById('train-dropzone');
  const nameInput = document.getElementById('speaker-name');

  if (input) input.addEventListener('change', async e => {
    const files = Array.from(e.target.files);
    if (files.length) await handleTrainUploadFiles(files);
    input.value = '';
  });
  if (zone) setupDrop(zone, files => handleTrainUploadFiles(files));
  nameInput?.addEventListener('input', updateTrainBtn);
  document.getElementById('train-btn')?.addEventListener('click', doTrain);
}

async function handleTrainUploadFiles(files) {
  showLoading(`Upload de ${files.length} fichier(s)...`);
  for (const file of files) {
    const item = addFileItem(file.name, `${(file.size/1024).toFixed(0)} KB`, 'pending', 'En cours...');
    try {
      const data = await uploadRaw(file);
      if (data?.embedding_extracted) {
        markFileItem(item, 'uploaded', '✓ Prêt');
        state.trainFiles.push({ filename:data.filename, path:data.filepath, originalName:file.name, size:file.size, source:'upload' });
        addRemoveHandler(item, data.filename);
      } else {
        markFileItem(item, 'error', '✗ Échec');
        toast(`Impossible de traiter ${file.name}`, 'warning');
        item.remove();
      }
    } catch { markFileItem(item, 'error', '✗ Erreur'); item.remove(); }
  }
  updateTrainBtn();
  hideLoading();
}

// ── Train recording session ───────────────────────────────────────────────────
let trainRecIdx = 0;  // sample counter

async function toggleRecord(ctx) {
  // Delegate train-specific logic
  if (ctx === 'train') {
    await toggleTrainRecord();
    return;
  }
  // Generic
  if (!recorders[ctx]) recorders[ctx] = new RecorderEngine(`${ctx}-canvas`, `${ctx}-timer`);
  const rec    = recorders[ctx];
  const recBtn = document.getElementById(`${ctx}-rec-btn`);
  const sendBtn= document.getElementById(`${ctx}-send-btn`);

  if (!rec.isRecording) {
    try {
      await rec.start();
      recBtn.classList.add('recording');
      recBtn.innerHTML = '<span class="rec-dot"></span> Arrêter';
      sendBtn?.classList.add('hidden');
    } catch(err) { toast('Micro inaccessible: '+err.message,'error'); }
  } else {
    rec.stop();
    recBtn.classList.remove('recording');
    recBtn.innerHTML = '<span class="rec-dot"></span> Réenregistrer';
    const dur = rec.getDurationSec();
    if (dur < 1.5) { toast('Trop court (min 1.5s)','warning'); recBtn.innerHTML='<span class="rec-dot"></span> Commencer'; return; }
    pendingBlobs[ctx] = rec.getWavBlob();
    sendBtn?.classList.remove('hidden');
    toast(`Enregistrement prêt — ${dur.toFixed(1)}s WAV 16kHz`, 'success');
    recorders[ctx] = null;
  }
}

async function toggleTrainRecord() {
  if (!recorders['train']) recorders['train'] = new RecorderEngine('train-canvas', 'train-timer');
  const rec     = recorders['train'];
  const recBtn  = document.getElementById('train-rec-btn');
  const saveBtn = document.getElementById('train-save-btn');
  const label   = document.getElementById('train-rec-label');

  if (!rec.isRecording) {
    try {
      await rec.start();
      recBtn.classList.add('recording');
      if (label) label.textContent = 'Arrêter l\'enregistrement';
      saveBtn?.classList.add('hidden');
    } catch(err) { toast('Micro inaccessible: '+err.message,'error'); }
  } else {
    rec.stop();
    recBtn.classList.remove('recording');
    if (label) label.textContent = 'Enregistrer l\'échantillon';
    const dur = rec.getDurationSec();
    if (dur < 1.5) { toast('Trop court','warning'); recorders['train']=null; return; }
    trainRecBlob = rec.getWavBlob();
    saveBtn?.classList.remove('hidden');
    toast(`Échantillon prêt (${dur.toFixed(1)}s) — cliquez Sauvegarder`, 'info');
    recorders['train'] = null;
  }
}

async function saveTrainRecording() {
  if (!trainRecBlob) return;
  const saveBtn = document.getElementById('train-save-btn');
  saveBtn?.setAttribute('disabled','');

  const fd = new FormData();
  fd.append('audio', trainRecBlob, 'train_sample.wav');
  fd.append('prefix', 'train_rec');

  try {
    showLoading('Upload de l\'échantillon...');
    const res  = await fetch(`${API}/record`, { method:'POST', body:fd });
    const data = await res.json();
    if (!res.ok || !data.embedding_extracted) throw new Error(data.error || 'Embedding échoué');

    trainRecIdx++;
    state.trainFiles.push({ filename:data.filename, path:data.filepath, originalName:`Échantillon ${trainRecIdx}`, size:0, source:'record' });

    // Update count badge
    const badge = document.getElementById('train-rec-count');
    if (badge) badge.textContent = `${trainRecIdx} / 3 min`;

    // Add preview chip
    const previews = document.getElementById('train-rec-previews');
    if (previews) {
      const chip = document.createElement('div');
      chip.className = 'rec-preview-chip';
      chip.dataset.filename = data.filename;
      chip.innerHTML = `🎤 Éch. ${trainRecIdx} <button onclick="removeTrainChip('${esc(data.filename)}',this.parentElement)">×</button>`;
      previews.appendChild(chip);
    }

    // Also add to unified file list
    const item = addFileItem(`🎤 Échantillon ${trainRecIdx}`, 'enregistrement', 'uploaded', '✓ Prêt');
    addRemoveHandler(item, data.filename);

    trainRecBlob = null;
    saveBtn?.removeAttribute('disabled');
    saveBtn?.classList.add('hidden');
    updateTrainBtn();
    toast(`Échantillon ${trainRecIdx} sauvegardé ✓`, 'success');
  } catch(err) {
    toast('Erreur: '+err.message, 'error');
    saveBtn?.removeAttribute('disabled');
  } finally { hideLoading(); }
}
window.saveTrainRecording = saveTrainRecording;

function removeTrainChip(filename, chipEl) {
  state.trainFiles = state.trainFiles.filter(f => f.filename !== filename);
  chipEl?.remove();
  updateTrainBtn();
  // Also remove from file list
  document.querySelector(`.file-item[data-filename="${filename}"]`)?.remove();
  trainRecIdx = Math.max(0, trainRecIdx - 1);
  document.getElementById('train-rec-count').textContent = `${trainRecIdx} / 3 min`;
}
window.removeTrainChip = removeTrainChip;

// ── File list helpers ─────────────────────────────────────────────────────────
function addFileItem(name, size, statusClass, statusText) {
  const listEl = document.getElementById('train-file-list');
  const item   = document.createElement('div');
  item.className = 'file-item';
  item.innerHTML = `
    <svg viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="1.5" width="16" height="16"><path d="M9 19V6l12-3v13M9 19c0 1.1-.9 2-2 2s-2-.9-2-2 .9-2 2-2 2 .9 2 2z"/></svg>
    <span class="file-item-name">${esc(name)}</span>
    <span class="file-item-size">${size}</span>
    <span class="file-item-status ${statusClass}">${statusText}</span>
  `;
  listEl?.appendChild(item);
  return item;
}

function markFileItem(item, statusClass, text) {
  const s = item.querySelector('.file-item-status');
  if (s) { s.className = `file-item-status ${statusClass}`; s.textContent = text; }
}

function addRemoveHandler(item, filename) {
  item.dataset.filename = filename;
  const btn = document.createElement('button');
  btn.className = 'file-item-remove';
  btn.title = 'Supprimer';
  btn.innerHTML = `<svg viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2" width="14" height="14"><path d="M18 6L6 18M6 6l12 12"/></svg>`;
  btn.onclick = () => { state.trainFiles = state.trainFiles.filter(f=>f.filename!==filename); item.remove(); updateTrainBtn(); };
  item.appendChild(btn);
}

function updateTrainBtn() {
  const btn  = document.getElementById('train-btn');
  const name = (document.getElementById('speaker-name')?.value||'').trim();
  if (btn) btn.disabled = !(name && state.trainFiles.length > 0);
}

async function doTrain() {
  const speakerName = document.getElementById('speaker-name').value.trim();
  const resultEl    = document.getElementById('train-result');
  if (!speakerName || !state.trainFiles.length) return;

  showLoading(`Entraînement LSTM pour ${speakerName}...`);
  try {
    const filesInfo = state.trainFiles.map(f => ({ filename:f.filename, path:f.path }));
    const res  = await fetch(`${API}/train`, {
      method:'POST', headers:{'Content-Type':'application/json'},
      body: JSON.stringify({ speaker_name:speakerName, files:filesInfo }),
    });
    const data = await res.json();
    if (!res.ok || data.status !== 'success') throw new Error(data.error || 'Échec entraînement');

    const cons = data.intra_similarity_mean ? (data.intra_similarity_mean*100).toFixed(1) : 'N/A';
    resultEl.className = 'result-card success';
    resultEl.innerHTML = `
      <div class="result-title"><svg viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2" width="20" height="20"><polyline points="20 6 9 17 4 12"/></svg> Modèle entraîné — ${esc(speakerName)}</div>
      <div class="stat-grid">
        <div class="stat-item"><div class="stat-label">ÉCHANTILLONS</div><div class="stat-value">${data.n_samples}</div></div>
        <div class="stat-item"><div class="stat-label">CONSISTANCE</div><div class="stat-value">${cons}%</div></div>
      </div>
      <div class="sim-bar-wrap">
        <div class="sim-bar-label"><span>Similarité intra-classe</span><span style="color:var(--accent);font-family:'Space Mono',monospace">${cons}%</span></div>
        <div class="sim-bar-track"><div class="sim-bar-fill" style="width:0%;background:var(--accent)" id="train-bar"></div></div>
      </div>
      <p style="margin-top:1rem;font-size:.8rem;color:var(--text-muted)">Modèle: <code>${esc(data.model_path||'')}</code></p>
    `;
    resultEl.classList.remove('hidden');
    setTimeout(() => { const f=document.getElementById('train-bar'); if(f) f.style.width=cons+'%'; }, 50);

    // Reset
    document.getElementById('speaker-name').value = '';
    document.getElementById('train-file-list').innerHTML = '';
    document.getElementById('train-rec-previews').innerHTML = '';
    state.trainFiles = [];
    trainRecIdx = 0;
    document.getElementById('train-rec-count').textContent = '0 / 3 min';
    updateTrainBtn();
    loadSpeakers();
    toast(`Modèle créé pour ${speakerName} !`, 'success');
  } catch(err) {
    showResultError(resultEl, err.message);
    toast(err.message, 'error');
  } finally { hideLoading(); }
}

// ══════════════════════════════════════════════════════════════
//  IDENTIFY TAB
// ══════════════════════════════════════════════════════════════
function initIdentify() {
  const zone  = document.getElementById('identify-dropzone');
  const input = document.getElementById('identify-input');
  if (zone)  zone.addEventListener('click', () => input.click());
  if (input) input.addEventListener('change', e => { if(e.target.files[0]) doIdentifyFile(e.target.files[0]); });
  setupDrop(zone, files => doIdentifyFile(files[0]));
}

async function doIdentifyFile(file) {
  const resultEl = document.getElementById('identify-result');
  showLoading('Identification en cours...');
  const fd = new FormData();
  fd.append('audio', file);
  try {
    const res  = await fetch(`${API}/identify`, { method:'POST', body:fd });
    const data = await res.json();
    if (!res.ok) throw new Error(data.error);
    renderIdentify(resultEl, data);
  } catch(err) {
    showResultError(resultEl, err.message);
  } finally { hideLoading(); }
}

async function doIdentifyRecord(blob) {
  const resultEl = document.getElementById('identify-result');
  showLoading('Identification en cours...');
  const fd = new FormData();
  fd.append('audio', blob, 'identify.wav');
  try {
    const res  = await fetch(`${API}/identify_record`, { method:'POST', body:fd });
    const data = await res.json();
    if (!res.ok) throw new Error(data.error);
    renderIdentify(resultEl, data);
  } catch(err) {
    showResultError(resultEl, err.message);
  } finally { hideLoading(); }
}

function renderIdentify(resultEl, data) {
  resultEl.className = `result-card ${data.speaker?'success':'info'}`;
  resultEl.innerHTML = `
    <div class="result-title">
      <svg viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2" width="20" height="20"><circle cx="11" cy="11" r="8"/><path d="m21 21-4.35-4.35"/></svg>
      ${data.speaker ? 'Locuteur identifié' : 'Non reconnu'}
    </div>
    ${renderIdentifyResult(data)}
  `;
  resultEl.classList.remove('hidden');
  animIdBar();
  toast(data.speaker ? `Identifié: ${data.speaker}` : 'Non reconnu', data.speaker?'success':'info');
}

function animIdBar() {
  setTimeout(() => {
    const bar = document.getElementById('id-bar');
    if (bar) {
      const pct = parseFloat(bar.dataset.pct || '50');
      bar.style.width = pct + '%';
    }
  }, 60);
}

function renderIdentifyResult(data) {
  if (data.speaker) {
    const pct  = (data.similarity * 100).toFixed(1);
    const conf = data.confidence || 'medium';
    const confColor = conf==='high'?'var(--accent)':conf==='medium'?'var(--warn)':'var(--text-muted)';
    return `
      <div style="display:flex;align-items:center;gap:1rem;margin-bottom:1rem">
        <div class="speaker-avatar" style="width:56px;height:56px;font-size:1.5rem">${data.speaker.charAt(0).toUpperCase()}</div>
        <div>
          <div style="font-size:1.2rem;font-weight:800">${esc(data.speaker)}</div>
          <div style="color:${confColor};font-size:.85rem;font-family:'Space Mono',monospace">Confiance: ${conf} · ${pct}%</div>
        </div>
      </div>
      <div class="sim-bar-wrap">
        <div class="sim-bar-track"><div class="sim-bar-fill" id="id-bar" data-pct="${pct}" style="width:0%;background:var(--accent)"></div></div>
      </div>
      ${data.all_scores && Object.keys(data.all_scores).length > 1 ? `
      <div style="margin-top:1rem">
        <div class="stat-label" style="margin-bottom:.5rem">TOUS LES SCORES</div>
        <div class="scores-list">
          ${Object.entries(data.all_scores).sort((a,b)=>b[1]-a[1]).map(([n,s]) =>
            `<div class="score-row"><span class="score-name">${esc(n)}</span><span class="score-pct">${(s*100).toFixed(1)}%</span></div>`
          ).join('')}
        </div>
      </div>` : ''}
    `;
  }
  return `
    <div class="verdict-badge verdict-unknown">⚠ Non reconnu</div>
    <p style="margin-top:1rem;font-size:.85rem;color:var(--text-muted)">
      ${esc(data.message || 'Aucun locuteur correspondant.')}
      ${data.similarity ? `Score max: ${(data.similarity*100).toFixed(1)}%` : ''}
    </p>
  `;
}

// ══════════════════════════════════════════════════════════════
//  SPEAKERS TAB
// ══════════════════════════════════════════════════════════════
async function loadSpeakers() {
  const grid  = document.getElementById('speakers-grid');
  const badge = document.getElementById('badge-speakers');
  if (!grid) return;
  try {
    const res  = await fetch(`${API}/speakers`);
    const data = await res.json();
    const sps  = data.speakers || [];
    if (badge) badge.textContent = sps.length || '';
    if (!sps.length) {
      grid.innerHTML = `<div class="empty-state"><svg viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="1" width="48" height="48"><path d="M17 21v-2a4 4 0 0 0-4-4H5a4 4 0 0 0-4 4v2"/><circle cx="9" cy="7" r="4"/></svg><p>Aucun locuteur</p><p class="hint">Allez dans Entraîner</p></div>`;
      return;
    }
    grid.innerHTML = sps.map((sp,i) => {
      const cons = sp.intra_similarity_mean ? (sp.intra_similarity_mean*100).toFixed(0) : 0;
      return `
        <div class="speaker-card" style="animation-delay:${i*.05}s">
          <div class="speaker-avatar">${sp.name.charAt(0).toUpperCase()}</div>
          <div class="speaker-name">${esc(sp.name)}</div>
          <div class="speaker-meta">${sp.n_samples} échantillons · ${new Date(sp.training_date).toLocaleDateString('fr-FR')}</div>
          ${cons?`<div class="speaker-consistency"><span style="font-size:.78rem;color:var(--text-muted)">Consistance</span><div class="mini-bar"><div class="mini-bar-fill" style="width:${cons}%"></div></div><span style="font-family:'Space Mono',monospace;font-size:.72rem;color:var(--accent)">${cons}%</span></div>`:''}
          <div class="speaker-actions">
            <button class="btn btn-danger-ghost btn-sm" onclick="deleteSpeaker('${esc(sp.name)}')">
              <svg viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2" width="14" height="14"><polyline points="3 6 5 6 21 6"/><path d="M19 6v14a2 2 0 0 1-2 2H7a2 2 0 0 1-2-2V6m3 0V4a1 1 0 0 1 1-1h4a1 1 0 0 1 1 1v2"/></svg>
              Supprimer
            </button>
          </div>
        </div>`;
    }).join('');
  } catch(err) {
    grid.innerHTML = `<div class="empty-state"><p style="color:var(--danger)">Erreur: ${esc(err.message)}</p></div>`;
  }
}
window.loadSpeakers = loadSpeakers;

async function deleteSpeaker(name) {
  if (!confirm(`Supprimer "${name}" ?`)) return;
  try {
    const res = await fetch(`${API}/delete_speaker/${encodeURIComponent(name)}`, {method:'DELETE'});
    if (!res.ok) throw new Error((await res.json()).error);
    toast(`${name} supprimé`, 'success');
    loadSpeakers();
  } catch(err) { toast(err.message,'error'); }
}
window.deleteSpeaker = deleteSpeaker;

// ══════════════════════════════════════════════════════════════
//  Health
// ══════════════════════════════════════════════════════════════
async function checkHealth() {
  const dot = document.getElementById('status-dot');
  const txt = document.getElementById('status-text');
  try {
    const res  = await fetch(`${API}/health`);
    if (res.ok) {
      const d = await res.json();
      if (dot) dot.className = 'status-indicator online';
      if (txt) txt.textContent = `Connecté · ${d.models_loaded} modèle(s)`;
    } else throw new Error();
  } catch { if (dot) dot.className='status-indicator offline'; if(txt) txt.textContent='Hors ligne'; }
  setTimeout(checkHealth, 15000);
}

// ══════════════════════════════════════════════════════════════
//  Utilities
// ══════════════════════════════════════════════════════════════
async function uploadRaw(file) {
  const fd = new FormData();
  fd.append('audio', file);
  const res = await fetch(`${API}/upload`, {method:'POST', body:fd});
  return res.ok ? res.json() : null;
}

function setupDrop(el, onDrop) {
  if (!el) return;
  el.addEventListener('dragover', e => { e.preventDefault(); el.classList.add('drag-over'); });
  el.addEventListener('dragleave', () => el.classList.remove('drag-over'));
  el.addEventListener('drop', e => {
    e.preventDefault(); el.classList.remove('drag-over');
    const files = Array.from(e.dataTransfer.files).filter(f => /\.(wav|mp3|flac|m4a|ogg)$/i.test(f.name)||f.type.startsWith('audio/'));
    if (files.length) onDrop(files); else toast('Fichier audio requis', 'error');
  });
}

function showLoading(text='Traitement...') {
  const o=document.getElementById('loading-overlay'), t=document.getElementById('loading-text');
  if(o) o.classList.remove('hidden'); if(t) t.textContent=text;
}
function hideLoading() { document.getElementById('loading-overlay')?.classList.add('hidden'); }

function showResultError(el, msg) {
  el.className='result-card error';
  el.innerHTML=`<div class="result-title" style="color:var(--danger)"><svg viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2" width="20" height="20"><circle cx="12" cy="12" r="10"/><path d="M15 9l-6 6M9 9l6 6"/></svg>Erreur</div><p style="font-size:.9rem;color:var(--text-muted)">${esc(msg)}</p>`;
  el.classList.remove('hidden');
  toast(msg,'error');
}

function toast(message, type='info') {
  const container = document.getElementById('toast-container');
  const icons = { success:'<svg viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2.5" width="16" height="16"><polyline points="20 6 9 17 4 12"/></svg>', error:'<svg viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2.5" width="16" height="16"><path d="M18 6L6 18M6 6l12 12"/></svg>', info:'<svg viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2.5" width="16" height="16"><circle cx="12" cy="12" r="10"/><path d="M12 16v-4M12 8h.01"/></svg>', warning:'<svg viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2.5" width="16" height="16"><path d="M10.29 3.86L1.82 18a2 2 0 0 0 1.71 3h16.94a2 2 0 0 0 1.71-3L13.71 3.86a2 2 0 0 0-3.42 0z"/></svg>' };
  const el = document.createElement('div');
  el.className = `toast ${type}`;
  el.innerHTML = `${icons[type]||''}${esc(message)}`;
  container.appendChild(el);
  setTimeout(() => { el.classList.add('fade-out'); setTimeout(()=>el.remove(),300); }, 3500);
}

function esc(s) { return String(s??'').replace(/&/g,'&amp;').replace(/</g,'&lt;').replace(/>/g,'&gt;').replace(/"/g,'&quot;'); }

// ============================================================================
// AUTHENTIFICATION
// ============================================================================

async function logout() {
    try {
        await fetch(`${API}/auth/logout`, { method: 'POST' });
        localStorage.removeItem('voice_user');
        window.location.href = '/';
    } catch(e) {
        console.error('Logout error', e);
    }
}

// Ajouter un bouton logout dans le sidebar
function addLogoutButton() {
    const sidebar = document.querySelector('.sidebar-status');
    if (sidebar && !document.getElementById('logout-btn-sidebar')) {
        const logoutBtn = document.createElement('button');
        logoutBtn.id = 'logout-btn-sidebar';
        logoutBtn.innerHTML = '🚪 Déconnexion';
        logoutBtn.style.background = 'rgba(255,51,85,0.1)';
        logoutBtn.style.border = '1px solid rgba(255,51,85,0.3)';
        logoutBtn.style.borderRadius = '6px';
        logoutBtn.style.padding = '6px 12px';
        logoutBtn.style.marginTop = '10px';
        logoutBtn.style.width = '100%';
        logoutBtn.style.cursor = 'pointer';
        logoutBtn.style.color = '#ff3355';
        logoutBtn.style.fontFamily = 'inherit';
        logoutBtn.onclick = logout;
        sidebar.appendChild(logoutBtn);
    }
}

// Fonctions pour mot de passe oublié (si présentes dans la page)
// Ces fonctions sont appelées depuis auth.html
window.showForgotPassword = function() {
    const tabs = document.querySelector('.auth-tabs');
    const loginForm = document.getElementById('login-form');
    const signupForm = document.getElementById('signup-form');
    const forgotForm = document.getElementById('forgot-form');
    
    if (tabs) tabs.style.display = 'none';
    if (loginForm) loginForm.classList.remove('active');
    if (signupForm) signupForm.classList.remove('active');
    if (forgotForm) forgotForm.classList.add('active');
};

window.backToLogin = function() {
    const tabs = document.querySelector('.auth-tabs');
    const loginForm = document.getElementById('login-form');
    const forgotForm = document.getElementById('forgot-form');
    
    if (tabs) tabs.style.display = 'flex';
    if (loginForm) loginForm.classList.add('active');
    if (forgotForm) forgotForm.classList.remove('active');
};

window.handleForgotPassword = async function() {
    const email = document.getElementById('forgot-email')?.value;
    const errorEl = document.getElementById('forgot-error');
    const successEl = document.getElementById('forgot-success');
    
    if (!email) {
        if (errorEl) {
            errorEl.textContent = 'Veuillez entrer votre email';
            errorEl.classList.add('show');
        }
        return;
    }
    
    try {
        const res = await fetch(`${API_URL}/auth/forgot-password`, {
            method: 'POST',
            headers: { 'Content-Type': 'application/json' },
            body: JSON.stringify({ email })
        });
        const data = await res.json();
        
        if (data.success) {
            if (errorEl) errorEl.classList.remove('show');
            if (successEl) {
                successEl.textContent = '✅ Un email vous a été envoyé avec les instructions.';
                successEl.classList.add('show');
            }
            document.getElementById('forgot-email').value = '';
            setTimeout(() => window.backToLogin(), 5000);
        } else {
            if (errorEl) {
                errorEl.textContent = data.message || 'Une erreur est survenue';
                errorEl.classList.add('show');
            }
        }
    } catch(e) {
        if (errorEl) {
            errorEl.textContent = 'Erreur réseau';
            errorEl.classList.add('show');
        }
    }
};