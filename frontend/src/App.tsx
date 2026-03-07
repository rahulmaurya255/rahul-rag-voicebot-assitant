import { useState, useRef, useEffect, useCallback } from "react";

type AppState = "idle" | "listening" | "processing" | "speaking";
type TTSItem = { text: string; blob: Promise<Blob | null> };

const API = "/api";

/* Audio feedback chimes */
function playTone(freq: number, dur: number, type: OscillatorType = "sine", vol = 0.08) {
  try {
    const ctx = new AudioContext();
    const osc = ctx.createOscillator();
    const gain = ctx.createGain();
    osc.type = type;
    osc.frequency.value = freq;
    gain.gain.setValueAtTime(vol, ctx.currentTime);
    gain.gain.exponentialRampToValueAtTime(0.001, ctx.currentTime + dur);
    osc.connect(gain);
    gain.connect(ctx.destination);
    osc.start();
    osc.stop(ctx.currentTime + dur);
    setTimeout(() => ctx.close(), (dur + 0.1) * 1000);
  } catch {}
}
const chimeListening = () => { playTone(880, 0.12, "sine", 0.06); setTimeout(() => playTone(1100, 0.15, "sine", 0.06), 100); };
const chimeProcessing = () => playTone(660, 0.2, "triangle", 0.05);
const chimeSpeaking = () => { playTone(520, 0.1, "sine", 0.04); setTimeout(() => playTone(780, 0.15, "sine", 0.04), 80); };

/* WAV encoder */
function encodeWAV(samples: Float32Array, sampleRate: number): Blob {
  const buf = new ArrayBuffer(44 + samples.length * 2);
  const view = new DataView(buf);
  const w = (off: number, s: string) => { for (let i = 0; i < s.length; i++) view.setUint8(off + i, s.charCodeAt(i)); };
  w(0, "RIFF"); view.setUint32(4, 36 + samples.length * 2, true);
  w(8, "WAVE"); w(12, "fmt "); view.setUint32(16, 16, true);
  view.setUint16(20, 1, true); view.setUint16(22, 1, true);
  view.setUint32(24, sampleRate, true); view.setUint32(28, sampleRate * 2, true);
  view.setUint16(32, 2, true); view.setUint16(34, 16, true);
  w(36, "data"); view.setUint32(40, samples.length * 2, true);
  for (let i = 0; i < samples.length; i++) {
    const s = Math.max(-1, Math.min(1, samples[i]));
    view.setInt16(44 + i * 2, s < 0 ? s * 0x8000 : s * 0x7fff, true);
  }
  return new Blob([buf], { type: "audio/wav" });
}

function splitSentences(text: string): string[] {
  const raw = text.match(/[^.!?]+[.!?]+[\s]*/g);
  if (!raw) return text.trim() ? [text.trim()] : [];
  return raw.map((s) => s.trim()).filter(Boolean);
}

function fetchTTSBlob(sentence: string, signal: AbortSignal): Promise<Blob | null> {
  return fetch(API + "/tts/sentence", {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify({ query: sentence }),
    signal,
  }).then(r => r.ok ? r.blob() : null).catch(() => null);
}


function spectralCentroid(fd: Uint8Array, sr: number, fft: number): number {
  const bHz = sr / fft;
  let ws = 0, tm = 0;
  for (let i = 1; i < fd.length; i++) { ws += fd[i] * (i * bHz); tm += fd[i]; }
  return tm > 0 ? ws / tm : 0;
}

function spectralFlatness(fd: Uint8Array): number {
  let ls = 0, li = 0, n = 0;
  for (let i = 1; i < fd.length; i++) { const v = Math.max(fd[i], 1); ls += Math.log(v); li += v; n++; }
  if (n === 0 || li === 0) return 1;
  return Math.exp(ls / n) / (li / n);
}

function speechBandRatio(fd: Uint8Array, sr: number, fft: number): number {
  const bHz = sr / fft;
  const lo = Math.floor(300 / bHz), hi = Math.min(Math.ceil(3500 / bHz), fd.length - 1);
  let sE = 0, tE = 0;
  for (let i = 1; i < fd.length; i++) { const e = fd[i] * fd[i]; tE += e; if (i >= lo && i <= hi) sE += e; }
  return tE > 0 ? sE / tE : 0;
}

function computeRMS(chunks: Float32Array[]): number {
  let sum = 0, count = 0;
  for (const c of chunks) { for (let i = 0; i < c.length; i++) { sum += c[i] * c[i]; count++; } }
  return count > 0 ? Math.sqrt(sum / count) : 0;
}

/* Word-by-word reveal: timing controlled by parent via subtitle updates */
function WordReveal({ text, className, label }: { text: string; className: string; label: string }) {
  const [visibleCount, setVisibleCount] = useState(0);
  const prevLenRef = useRef(0);
  const words = text.split(/\s+/).filter(Boolean);

  useEffect(() => {
    if (words.length > prevLenRef.current) {
      prevLenRef.current = words.length;
      /* Render opacity:0 first, then flip visible next frame for CSS transition */
      requestAnimationFrame(() => setVisibleCount(words.length));
    }
    if (words.length === 0) {
      prevLenRef.current = 0;
      setVisibleCount(0);
    }
  }, [words.length]);

  if (!text) return null;
  return (
    <div className={"transcript " + className}>
      <span className="transcript-label">{label}</span>
      <p>
        {words.map((w, i) => (
          <span key={i} className={"word" + (i < visibleCount ? " visible" : "")}>{w} </span>
        ))}
      </p>
    </div>
  );
}

/* Main App */
export default function App() {
  const [state, setState] = useState<AppState>("idle");
  const [subtitle, setSubtitle] = useState("");
  const [userText, setUserText] = useState("");
  const [volume, setVolume] = useState(0);
  const [error, setError] = useState("");
  const [paused, setPaused] = useState(false);
  const pausedRef = useRef(false);

  const streamRef = useRef<MediaStream | null>(null);
  const audioCtxRef = useRef<AudioContext | null>(null);
  const analyserRef = useRef<AnalyserNode | null>(null);
  const pcmRef = useRef<Float32Array[]>([]);
  const preRollRef = useRef<Float32Array[]>([]);
  const chatHistoryRef = useRef<{role:string;content:string}[]>([]);
  const processorRef = useRef<ScriptProcessorNode | null>(null);
  const audioElRef = useRef<HTMLAudioElement | null>(null);
  const ttsBufferRef = useRef<TTSItem[]>([]);
  const ttsPlayingRef = useRef(false);
  const abortRef = useRef<AbortController | null>(null);
  const displayedTextRef = useRef("");

  const stateRef = useRef(state);
  stateRef.current = state;
  pausedRef.current = paused;
  const silenceCountRef = useRef(0);
  const speechCountRef = useRef(0);
  const vadActiveRef = useRef(false);
  const noiseFloorRef = useRef(0.01);
  const calibrationCountRef = useRef(0);
  const volumeRef = useRef(0);
  volumeRef.current = volume;

  const canvasRef = useRef<HTMLCanvasElement>(null);
  const animFrameRef = useRef(0);

  const SAMPLE_RATE = 16000;
  const FFT_SIZE = 1024;
  const SPK_MULT = 2.5;
  const SIL_MULT = 1.3;
  const MIN_RMS = 0.015;
  const SPEECH_FRAMES_NEEDED = 7;
  const SILENCE_FRAMES_NEEDED = 22;
  const BARGE_IN_FRAMES = 5;
  const CALIBRATION_FRAMES = 20;
  const PRE_ROLL_CHUNKS = 5;
  const MIN_REC_SEC = 0.8;
  const MIN_REC_RMS = 0.01;

  useEffect(() => {
    if (state === "listening") chimeListening();
    else if (state === "processing") chimeProcessing();
    else if (state === "speaking") chimeSpeaking();
  }, [state]);


  const isSpeech = useCallback((rms: number): boolean => {
    const thr = Math.max(MIN_RMS, noiseFloorRef.current * SPK_MULT);
    if (rms < thr) return false;
    const a = analyserRef.current;
    if (!a) return true;
    const fd = new Uint8Array(a.frequencyBinCount); a.getByteFrequencyData(fd);
    const sc = spectralCentroid(fd, SAMPLE_RATE, FFT_SIZE); if (sc < 250 || sc > 3800) return false;
    if (spectralFlatness(fd) > 0.85) return false;
    if (speechBandRatio(fd, SAMPLE_RATE, FFT_SIZE) < 0.25) return false;
    return true;
  }, []);

  const initMic = useCallback(async () => {
    if (streamRef.current) return;
    const stream = await navigator.mediaDevices.getUserMedia({
      audio: { sampleRate: SAMPLE_RATE, channelCount: 1, echoCancellation: true, noiseSuppression: true, autoGainControl: true },
    });
    streamRef.current = stream;
    const ctx = new AudioContext({ sampleRate: SAMPLE_RATE });
    audioCtxRef.current = ctx;
    const source = ctx.createMediaStreamSource(stream);
    const analyser = ctx.createAnalyser();
    analyser.fftSize = FFT_SIZE;
    analyser.smoothingTimeConstant = 0.3;
    source.connect(analyser);
    analyserRef.current = analyser;
    const processor = ctx.createScriptProcessor(4096, 1, 1);
    source.connect(processor);
    processor.connect(ctx.destination);
    processor.onaudioprocess = (e: AudioProcessingEvent) => {
      const data = e.inputBuffer.getChannelData(0);
      let sum = 0;
      for (let i = 0; i < data.length; i++) sum += data[i] * data[i];
      const rms = Math.sqrt(sum / data.length);
      setVolume(rms);
      if (stateRef.current === "idle" || stateRef.current === "listening") {
        const cc = calibrationCountRef.current;
        if (cc < CALIBRATION_FRAMES) {
          noiseFloorRef.current = noiseFloorRef.current * 0.7 + rms * 0.3;
          calibrationCountRef.current = cc + 1;
        } else if (stateRef.current === "idle") {
          noiseFloorRef.current = noiseFloorRef.current * 0.97 + rms * 0.03;
        }
      }
      /* Pre-roll: always keep last N chunks as ring buffer for capturing speech onset */
      preRollRef.current.push(new Float32Array(data));
      if (preRollRef.current.length > 5) preRollRef.current.shift();
      if (stateRef.current === "listening") {
        pcmRef.current.push(new Float32Array(data));
      }
    };
    processorRef.current = processor;
  }, []);

  const startRecording = useCallback(() => {
    /* Prepend pre-roll audio to capture the speech onset that triggered VAD */
    pcmRef.current = [...preRollRef.current];
    preRollRef.current = [];
    setState("listening");
    setError("");
  }, []);

  const stopRecordingAndSend = useCallback(async () => {
    if (stateRef.current !== "listening") return;
    setState("processing");
    setSubtitle("Thinking...");
    setUserText("");
    displayedTextRef.current = "";
    const chunks = pcmRef.current;
    pcmRef.current = [];
    if (chunks.length === 0) { setState("idle"); setSubtitle(""); return; }
    const totalLen = chunks.reduce((a: number, c: Float32Array) => a + c.length, 0);
    const merged = new Float32Array(totalLen);
    let off = 0;
    for (const c of chunks) { merged.set(c, off); off += c.length; }
    if (merged.length < SAMPLE_RATE * MIN_REC_SEC) { setState("idle"); setSubtitle(""); return; }
    if (computeRMS(chunks) < MIN_REC_RMS) { setState("idle"); setSubtitle(""); return; }
    const wavBlob = encodeWAV(merged, SAMPLE_RATE);
    await sendAudioStreaming(wavBlob);
  }, []);

  const sendAudioStreaming = async (blob: Blob) => {
    const formData = new FormData();
    formData.append("audio", blob, "recording.wav");
    if (chatHistoryRef.current.length > 0) {
      formData.append("history", JSON.stringify(chatHistoryRef.current.slice(-6)));
    }
    if (abortRef.current) abortRef.current.abort();
    const controller = new AbortController();
    abortRef.current = controller;
    try {
      const resp = await fetch(API + "/voice-query/stream", {
        method: "POST", body: formData, signal: controller.signal,
      });
      if (!resp.ok) throw new Error("API error: " + resp.status);
      const reader = resp.body!.getReader();
      const decoder = new TextDecoder();
      let sentenceBuffer = "";
      let fullResponse = "";
      ttsBufferRef.current = [];
      ttsPlayingRef.current = false;
      displayedTextRef.current = "";
      while (true) {
        const { done, value } = await reader.read();
        if (done || controller.signal.aborted) break;
        const text = decoder.decode(value, { stream: true });
        const lines = text.split("\n");
        for (const line of lines) {
          if (!line.startsWith("data: ")) continue;
          const jsonStr = line.slice(6).trim();
          if (!jsonStr) continue;
          try {
            const evt = JSON.parse(jsonStr);
            if (evt.type === "transcription") {
              setUserText(evt.text || "(no transcription)");
              if (!evt.text) setSubtitle("I didn't quite catch that...");
            } else if (evt.type === "token") {
              sentenceBuffer += evt.text;
              fullResponse += evt.text;
              /* Do NOT setSubtitle here - text reveals when TTS plays */
              const sentences = splitSentences(sentenceBuffer);
              if (sentences.length > 1) {
                for (let i = 0; i < sentences.length - 1; i++) {
                  const s = sentences[i];
                  ttsBufferRef.current.push({ text: s, blob: fetchTTSBlob(s, controller.signal) });
                }
                sentenceBuffer = sentences[sentences.length - 1];
                if (!ttsPlayingRef.current) playTTSQueue(controller.signal);
              }
            } else if (evt.type === "done") {
              if (fullResponse.trim()) chatHistoryRef.current.push({role:"assistant",content:fullResponse.trim()});
              /* Keep only last 10 turns to limit memory */
              if (chatHistoryRef.current.length > 10) chatHistoryRef.current = chatHistoryRef.current.slice(-10);
              if (sentenceBuffer.trim()) {
                const s = sentenceBuffer.trim();
                ttsBufferRef.current.push({ text: s, blob: fetchTTSBlob(s, controller.signal) });
                if (!ttsPlayingRef.current) playTTSQueue(controller.signal);
              }
            }
          } catch {}
        }
      }
    } catch (e: any) {
      if (e.name !== "AbortError") { setError(e.message); setState("idle"); setSubtitle(""); }
    }
  };

  const playTTSQueue = async (signal: AbortSignal) => {
    if (ttsPlayingRef.current) return;
    ttsPlayingRef.current = true;
    setState("speaking");
    while (ttsBufferRef.current.length > 0) {
      if (signal.aborted) break;
      const item = ttsBufferRef.current.shift()!;
      const words = item.text.split(/\s+/).filter(Boolean);
      try {
        const audioBlob = await item.blob;
        if (!audioBlob || signal.aborted) {
          /* No audio - reveal all words instantly */
          for (const w of words) {
            displayedTextRef.current += (displayedTextRef.current ? " " : "") + w;
            setSubtitle(displayedTextRef.current);
          }
          continue;
        }
        const url = URL.createObjectURL(audioBlob);
        await new Promise<void>((resolve, reject) => {
          const audio = new Audio(url);
          audioElRef.current = audio;
          let revealTimer: ReturnType<typeof setInterval> | null = null;
          let wordIdx = 0;
          const baseText = displayedTextRef.current;

          const revealNextWord = () => {
            if (wordIdx < words.length) {
              wordIdx++;
              const soFar = words.slice(0, wordIdx).join(" ");
              displayedTextRef.current = baseText + (baseText ? " " : "") + soFar;
              setSubtitle(displayedTextRef.current);
            }
            if (wordIdx >= words.length && revealTimer) {
              clearInterval(revealTimer);
              revealTimer = null;
            }
          };

          const cleanup = () => {
            if (revealTimer) { clearInterval(revealTimer); revealTimer = null; }
            /* Ensure all words revealed on cleanup */
            if (wordIdx < words.length) {
              displayedTextRef.current = baseText + (baseText ? " " : "") + item.text;
              setSubtitle(displayedTextRef.current);
            }
            URL.revokeObjectURL(url);
            audioElRef.current = null;
          };

          audio.onloadedmetadata = () => {
            const dur = audio.duration;
            /* Reveal words across 90% of audio duration for natural pacing */
            const msPerWord = Math.max(80, (dur * 900) / Math.max(words.length, 1));
            revealNextWord(); /* Show first word immediately */
            if (words.length > 1) {
              revealTimer = setInterval(revealNextWord, msPerWord);
            }
          };

          audio.onended = () => { cleanup(); resolve(); };
          audio.onerror = () => { cleanup(); resolve(); };
          const onAbort = () => { cleanup(); audio.pause(); reject(new DOMException("Aborted", "AbortError")); };
          signal.addEventListener("abort", onAbort, { once: true });
          audio.play().catch(() => { cleanup(); resolve(); });
        });
      } catch (e: any) { if (e.name === "AbortError") break; }
    }
    ttsPlayingRef.current = false;
    if (!signal.aborted) {
      /* Auto-listen for follow-up (continuous conversation) */
      pcmRef.current = [];
      setState("listening");
      vadActiveRef.current = true;
      silenceCountRef.current = 0;
      speechCountRef.current = 0;
    }
  };

  const interruptAll = useCallback(() => {
    if (audioElRef.current) { audioElRef.current.pause(); audioElRef.current = null; }
    ttsBufferRef.current = [];
    ttsPlayingRef.current = false;
    displayedTextRef.current = "";
    if (abortRef.current) { abortRef.current.abort(); abortRef.current = null; }
    setState("idle");
  }, []);

  /* Multi-feature VAD loop */
  useEffect(() => {
    initMic();
    const interval = setInterval(() => {
      if (pausedRef.current) return;
      const v = volumeRef.current;
      const s = stateRef.current;
      
      const silenceThreshold = Math.max(0.006, noiseFloorRef.current * SIL_MULT);

      const speechDetected = isSpeech(v);

      if (s === "speaking" || s === "processing") {
        if (speechDetected) {
          speechCountRef.current++;
          if (speechCountRef.current >= BARGE_IN_FRAMES) {
            interruptAll();
            speechCountRef.current = 0;
            silenceCountRef.current = 0;
            setTimeout(() => {
              if (stateRef.current === "idle") {
                startRecording();
                vadActiveRef.current = true;
              }
            }, 50);
          }
        } else {
          speechCountRef.current = Math.max(0, speechCountRef.current - 1);
        }
        return;
      }

      if (s === "idle") {
        if (speechDetected) {
          speechCountRef.current++;
          if (speechCountRef.current >= SPEECH_FRAMES_NEEDED) {
            startRecording();
            vadActiveRef.current = true;
            speechCountRef.current = 0;
            silenceCountRef.current = 0;
          }
        } else {
          speechCountRef.current = Math.max(0, speechCountRef.current - 1);
        }
      }

      if (s === "listening" && vadActiveRef.current) {
        if (v < silenceThreshold) {
          silenceCountRef.current++;
          if (silenceCountRef.current >= SILENCE_FRAMES_NEEDED) {
            vadActiveRef.current = false;
            silenceCountRef.current = 0;
            speechCountRef.current = 0;
            stopRecordingAndSend();
          }
        } else if (speechDetected) {
          silenceCountRef.current = 0;
        } else {
          silenceCountRef.current = Math.max(0, silenceCountRef.current - 1);
        }
      }
    }, 80);
    return () => clearInterval(interval);
  }, [initMic, startRecording, stopRecordingAndSend, interruptAll, isSpeech]);

  /* Canvas orb */
  useEffect(() => {
    const canvas = canvasRef.current;
    if (!canvas) return;
    const ctx = canvas.getContext("2d")!;
    const dpr = window.devicePixelRatio || 1;
    canvas.width = 300 * dpr;
    canvas.height = 300 * dpr;
    ctx.scale(dpr, dpr);
    let phase = 0;
    const draw = () => {
      animFrameRef.current = requestAnimationFrame(draw);
      const w = 300, h = 300, cx = w / 2, cy = h / 2;
      ctx.clearRect(0, 0, w, h);
      phase += 0.02;
      const s = stateRef.current;
      const v = Math.min(volumeRef.current * 8, 1);
      let baseR = 70;
      if (s === "listening") baseR = 75 + v * 30;
      else if (s === "processing") baseR = 72 + Math.sin(phase * 3) * 8;
      else if (s === "speaking") baseR = 78 + v * 20;
      const glowCount = s === "idle" ? 2 : s === "listening" ? 4 : 3;
      for (let ring = glowCount; ring >= 1; ring--) {
        const ringR = baseR + ring * (s === "listening" ? 12 + v * 15 : 10);
        const alpha = (0.06 / ring) * (s === "idle" ? 0.5 : 1);
        let color: string;
        if (s === "idle") color = "rgba(99, 102, 241, " + alpha + ")";
        else if (s === "listening") color = "rgba(239, 68, 68, " + (alpha + v * 0.05) + ")";
        else if (s === "processing") color = "rgba(245, 158, 11, " + alpha + ")";
        else color = "rgba(99, 102, 241, " + (alpha + v * 0.03) + ")";
        ctx.beginPath();
        for (let a = 0; a <= Math.PI * 2; a += 0.05) {
          const noise = Math.sin(a * 3 + phase * (1 + ring * 0.5)) * (s === "idle" ? 2 : 4 + v * 8)
                      + Math.sin(a * 5 - phase * 0.7) * (s === "listening" ? 3 + v * 5 : 2);
          const r = ringR + noise;
          const x = cx + Math.cos(a) * r;
          const y = cy + Math.sin(a) * r;
          if (a === 0) ctx.moveTo(x, y); else ctx.lineTo(x, y);
        }
        ctx.closePath();
        ctx.fillStyle = color;
        ctx.fill();
      }
      const grad = ctx.createRadialGradient(cx - 15, cy - 15, 10, cx, cy, baseR);
      if (s === "idle") {
        grad.addColorStop(0, "rgba(139, 92, 246, 0.9)");
        grad.addColorStop(0.6, "rgba(99, 102, 241, 0.8)");
        grad.addColorStop(1, "rgba(79, 70, 229, 0.6)");
      } else if (s === "listening") {
        grad.addColorStop(0, "rgba(248, 113, 113, 0.95)");
        grad.addColorStop(0.5, "rgba(239, 68, 68, 0.85)");
        grad.addColorStop(1, "rgba(185, 28, 28, 0.7)");
      } else if (s === "processing") {
        const pulse = (Math.sin(phase * 4) + 1) / 2;
        grad.addColorStop(0, "rgba(251, 191, 36, " + (0.8 + pulse * 0.15) + ")");
        grad.addColorStop(0.6, "rgba(245, 158, 11, " + (0.7 + pulse * 0.1) + ")");
        grad.addColorStop(1, "rgba(217, 119, 6, 0.6)");
      } else {
        grad.addColorStop(0, "rgba(129, 140, 248, 0.95)");
        grad.addColorStop(0.5, "rgba(99, 102, 241, 0.85)");
        grad.addColorStop(1, "rgba(67, 56, 202, 0.7)");
      }
      ctx.beginPath();
      for (let a = 0; a <= Math.PI * 2; a += 0.04) {
        const wobble = s === "idle"
          ? Math.sin(a * 4 + phase) * 2
          : s === "listening"
          ? Math.sin(a * 3 + phase * 2) * (3 + v * 12) + Math.sin(a * 7 - phase) * v * 6
          : s === "processing"
          ? Math.sin(a * 5 + phase * 3) * 4
          : Math.sin(a * 4 + phase * 1.5) * (2 + v * 8) + Math.cos(a * 6 - phase) * v * 4;
        const r = baseR + wobble;
        const x = cx + Math.cos(a) * r;
        const y = cy + Math.sin(a) * r;
        if (a === 0) ctx.moveTo(x, y); else ctx.lineTo(x, y);
      }
      ctx.closePath();
      ctx.fillStyle = grad;
      ctx.fill();
      const innerGrad = ctx.createRadialGradient(cx - 20, cy - 25, 5, cx, cy, baseR * 0.6);
      innerGrad.addColorStop(0, "rgba(255, 255, 255, 0.25)");
      innerGrad.addColorStop(1, "rgba(255, 255, 255, 0)");
      ctx.beginPath();
      ctx.arc(cx, cy, baseR * 0.6, 0, Math.PI * 2);
      ctx.fillStyle = innerGrad;
      ctx.fill();
    };
    draw();
    return () => cancelAnimationFrame(animFrameRef.current);
  }, []);

  const handleOrbClick = () => {
    if (state === "idle") {
      initMic().then(() => {
        startRecording();
        vadActiveRef.current = true;
        silenceCountRef.current = 0;
        speechCountRef.current = 0;
      });
    }
    else if (state === "listening") { vadActiveRef.current = false; stopRecordingAndSend(); }
    else if (state === "speaking" || state === "processing") { interruptAll(); }
  };

  const stateLabels: Record<AppState, string> = {
    idle: paused ? "Paused" : "Tap or start speaking",
    listening: "Listening...",
    processing: "Thinking...",
    speaking: "Speaking...",
  };

  return (
    <div className="app">
      <header className="header">
        <div className="header-left">
          <div className="logo-icon">M</div>
          <div>
            <h1>Mike</h1>
            <p className="subtitle">Rahul Maurya&apos;s AI Assistant</p>
          </div>
        </div>
        <div className="header-right">
          <div className={"status-badge " + state}>
            <span className="status-dot" />
            <span className="status-text">{state.charAt(0).toUpperCase() + state.slice(1)}</span>
          </div>
        </div>
      </header>

      <div className="orb-area">
        <WordReveal text={userText} className="user-transcript" label="You" />

        <div className="orb-container" onClick={handleOrbClick}>
          <canvas ref={canvasRef} className="orb-canvas" width={300} height={300} />
          <p className="orb-label">{stateLabels[state]}</p>
        </div>

        <WordReveal text={subtitle} className="ai-transcript" label="Mike" />

        {error && <div className="error-banner">{error}</div>}
      </div>

      <footer className="footer-hint">
        <div className="toggle-row">
          <button className={"toggle-btn" + (paused ? " offline" : " online")} onClick={() => {
            if (!paused) { interruptAll(); }
            setPaused(!paused);
          }}>
            <span className="toggle-track"><span className="toggle-thumb" /></span>
            <span className="toggle-label">{paused ? "Offline" : "Online"}</span>
          </button>
        </div>
        <p>{paused ? "Mike is offline" : "Always listening · Speak naturally · Tap orb to interrupt"}</p>
      </footer>
    </div>
  );
}
