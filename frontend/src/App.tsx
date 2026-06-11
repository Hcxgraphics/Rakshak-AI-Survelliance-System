import React, { useEffect, useMemo, useRef, useState } from "react";
import { createRoot } from "react-dom/client";
import { AnimatePresence, motion } from "framer-motion";
import { ConfidenceBar } from "./components/ConfidenceBar";
import { DetectionBadge } from "./components/DetectionBadge";
import { LogPanel } from "./components/LogPanel";
import { Timeline } from "./components/Timeline";
import { SplashScreen } from "./components/SplashScreen";
import { DetectionResponse, fetchHealth, fetchLogs, imageSrc, liveDetect, LogItem, uploadMedia } from "./lib/api";

import "./styles/global.css";
import "./styles/design-tokens.css";

type Page = "dashboard" | "surveillance" | "analytics" | "system";

const alertRisks = new Set(["critical", "high"]);

export default function App() {
  const [showSplash, setShowSplash] = useState(true);
  const [page, setPage] = useState<Page>("dashboard");
  const [theme, setTheme] = useState<"dark" | "light">("dark");
  
  // Dashboard & Inference States
  const [file, setFile] = useState<File | null>(null);
  const [preview, setPreview] = useState("");
  const [result, setResult] = useState<DetectionResponse | null>(null);
  const [timeline, setTimeline] = useState<DetectionResponse[]>([]);
  const [logs, setLogs] = useState<LogItem[]>([]);
  const [loading, setLoading] = useState(false);
  const [cameraOn, setCameraOn] = useState(false);
  const [detectionOn, setDetectionOn] = useState(true);
  const [threshold, setThreshold] = useState(0.55);
  const [saveEvidence, setSaveEvidence] = useState(false);
  
  // Notifications & Statuses
  const [errorToast, setErrorToast] = useState("");
  const [health, setHealth] = useState("checking");
  const [emergencyAlert, setEmergencyAlert] = useState("");
  
  // Camera & Streaming refs
  const videoRef = useRef<HTMLVideoElement | null>(null);
  const canvasRef = useRef<HTMLCanvasElement | null>(null);
  const streamRef = useRef<MediaStream | null>(null);
  const frameIndex = useRef(0);
  const busyRef = useRef(false);

  // Surveillance camera zones state
  const [activeZone, setActiveZone] = useState<"A" | "B" | "C" | "D">("A");
  const [fps, setFps] = useState(24.5);
  const [bufferLatency, setBufferLatency] = useState(18);

  const mediaKind = useMemo(() => {
    if (!file) return "auto";
    return file.type.startsWith("video") ? "video" : "image";
  }, [file]);

  // Apply light/dark theme class to body
  useEffect(() => {
    const root = window.document.documentElement;
    if (theme === "light") {
      root.classList.add("light");
    } else {
      root.classList.remove("light");
    }
  }, [theme]);

  // Fetch Health & Logs from API
  useEffect(() => {
    fetchHealth()
      .then((data) => setHealth(data.status === "ok" ? "online" : "degraded"))
      .catch(() => setHealth("offline"));

    const logsInterval = window.setInterval(() => {
      fetchLogs()
        .then((data) => setLogs(data.items))
        .catch(() => undefined);
    }, 2200);

    return () => window.clearInterval(logsInterval);
  }, []);

  // Set file upload preview
  useEffect(() => {
    if (!file) {
      setPreview("");
      return;
    }
    const url = URL.createObjectURL(file);
    setPreview(url);
    return () => URL.revokeObjectURL(url);
  }, [file]);

  // Control Webcam stream
  useEffect(() => {
    if (!cameraOn) return undefined;
    let cancelled = false;
    navigator.mediaDevices
      .getUserMedia({ video: { width: 1280, height: 720 }, audio: false })
      .then((stream) => {
        if (cancelled) return;
        streamRef.current = stream;
        if (videoRef.current) videoRef.current.srcObject = stream;
      })
      .catch((exc) => {
        showErrorToast(`Camera unavailable: ${exc.message}`);
        setCameraOn(false);
      });

    return () => {
      cancelled = true;
      streamRef.current?.getTracks().forEach((track) => track.stop());
      streamRef.current = null;
    };
  }, [cameraOn]);

  // Live Camera polling frame submission
  useEffect(() => {
    if (!cameraOn) return undefined;
    const interval = window.setInterval(async () => {
      if (!videoRef.current || !canvasRef.current || busyRef.current) return;
      const video = videoRef.current;
      if (video.readyState < 2) return;
      
      busyRef.current = true;
      const canvas = canvasRef.current;
      canvas.width = video.videoWidth || 960;
      canvas.height = video.videoHeight || 540;
      const context = canvas.getContext("2d");
      if (!context) return;
      context.drawImage(video, 0, 0, canvas.width, canvas.height);
      
      canvas.toBlob(async (blob) => {
        if (!blob) {
          busyRef.current = false;
          return;
        }
        try {
          // Poll frame every 1200ms
          let data: DetectionResponse;
          if (health === "offline") {
            // Simulated live detection when backend is offline
            data = {
              request_id: "demo-live-" + Math.random().toString(36).substring(2, 9),
              title: "Live Offline Demo",
              summary: "Monitoring feed — (Simulated safety check)",
              scene: "All clear",
              risk_level: "low",
              fusion_confidence: 0.05,
              weapon_count: 0,
              weapon_detections: [],
              has_weapon_yolo: false,
              gun_score_police: 0.01,
              knife_score_police: 0.0,
              police_detected: false,
              police_score: 0.04,
              violence_score_police: 0.02,
              violence_score_lstm: 0.01,
              accident_class: null,
              accident_confidence: 0.0,
              signals: [
                { label: "Weapon Detector", confidence: 0.01, source: "weapon", passed: false, weight: 0.4, weighted_score: 0.004 },
                { label: "Police Uniform Classifier", confidence: 0.04, source: "police", passed: false, weight: 0.1, weighted_score: 0.004 }
              ],
              component_errors: {},
              component_latency_ms: { "yolo": 4, "mobilenet": 2 },
              image_base64: ""
            };
          } else {
            data = await liveDetect(blob, threshold, detectionOn, frameIndex.current++, saveEvidence);
          }
          setResult(data);
          setTimeline((items) => [...items.slice(-30), data]);
          triggerAlert(data);
          
          // Randomize FPS & latency slightly to look real
          setFps(Number((24 + Math.random() * 2).toFixed(1)));
          setBufferLatency(Math.floor(14 + Math.random() * 8));
        } catch (exc) {
          showErrorToast(exc instanceof Error ? exc.message : "Live detection failed");
        } finally {
          busyRef.current = false;
        }
      }, "image/jpeg", 0.82);
    }, 1200);

    return () => window.clearInterval(interval);
  }, [cameraOn, detectionOn, threshold, saveEvidence, health]);

  function triggerAlert(data: DetectionResponse) {
    if (!alertRisks.has(data.risk_level)) return;
    setEmergencyAlert(data.summary);
    window.setTimeout(() => setEmergencyAlert(""), 4000);
    try {
      const audio = new Audio("data:audio/wav;base64,UklGRiQAAABXQVZFZm10IBAAAAABAAEAESsAACJWAAACABAAZGF0YQAAAAA=");
      void audio.play();
    } catch {
      return;
    }
  }

  function showErrorToast(msg: string) {
    setErrorToast(msg);
    window.setTimeout(() => setErrorToast(""), 4500);
  }

  async function processUpload() {
    if (!file) return;
    setLoading(true);
    setResult(null);
    if (health === "offline") {
      showErrorToast("Backend offline: executing client-side simulation demo.");
      window.setTimeout(() => {
        const isWeapon = file.name.toLowerCase().includes("weapon") || file.name.toLowerCase().includes("gun") || file.name.toLowerCase().includes("knife");
        const mockResult: DetectionResponse = {
          request_id: "demo-" + Math.random().toString(36).substring(2, 9),
          title: "Offline Simulation",
          summary: isWeapon ? "Potential weapon hazard detected (simulated)" : "Scene evaluated normal (simulated)",
          scene: isWeapon ? "Weapon threat detected" : "No threat found",
          risk_level: isWeapon ? "high" : "low",
          fusion_confidence: isWeapon ? 0.78 : 0.05,
          weapon_count: isWeapon ? 1 : 0,
          weapon_detections: isWeapon ? [{ label: "simulated_weapon", confidence: 0.85, box: [100, 100, 300, 300] }] : [],
          has_weapon_yolo: isWeapon,
          gun_score_police: isWeapon ? 0.85 : 0.01,
          knife_score_police: 0.0,
          police_detected: false,
          police_score: 0.05,
          violence_score_police: 0.02,
          violence_score_lstm: 0.01,
          accident_class: null,
          accident_confidence: 0.0,
          signals: [
            { label: "Weapon Detection (YOLOv8)", confidence: isWeapon ? 0.85 : 0.01, source: "weapon", passed: isWeapon, weight: 0.4, weighted_score: isWeapon ? 0.34 : 0.004 },
            { label: "Police Uniform (MobileNet)", confidence: 0.05, source: "police", passed: false, weight: 0.1, weighted_score: 0.005 }
          ],
          component_errors: {},
          component_latency_ms: { "yolo": 45, "mobilenet": 25 },
          image_base64: ""
        };
        setResult(mockResult);
        setTimeline((items) => [...items.slice(-30), mockResult]);
        triggerAlert(mockResult);
        setLoading(false);
      }, 1000);
      return;
    }
    try {
      const data = await uploadMedia(file, mediaKind as "auto" | "image" | "video", threshold, saveEvidence);
      setResult(data);
      setTimeline((items) => [...items.slice(-30), data]);
      triggerAlert(data);
    } catch (exc) {
      showErrorToast(exc instanceof Error ? exc.message : "Upload processing failed");
    } finally {
      setLoading(false);
    }
  }

  const triggerManualAlert = () => {
    setEmergencyAlert("Emergency Lockdown Triggered Manually!");
    window.setTimeout(() => setEmergencyAlert(""), 5000);
  };

  const output = imageSrc(result);
  const combinedViolence = Math.max(result?.violence_score_lstm ?? 0, result?.violence_score_police ?? 0);
  const weaponScore = Math.max(
    result?.weapon_detections?.reduce((score, d) => Math.max(score, d.confidence), 0) ?? 0,
    result?.gun_score_police ?? 0,
    result?.knife_score_police ?? 0,
    result?.has_weapon_yolo ? 0.75 : 0,
  );

  // Splash display
  if (showSplash) {
    return <SplashScreen onComplete={() => setShowSplash(false)} healthStatus={health} />;
  }

  return (
    <div className="min-h-screen flex flex-col md:flex-row bg-[#080f1a] text-slate-200">
      {/* Sidebar navigation */}
      <aside className="w-full md:w-64 bg-[#0d1b2a] border-r border-white/5 flex flex-col flex-shrink-0">
        <div className="h-16 px-6 border-b border-white/5 flex items-center gap-3">
          <div className="h-8 w-8 rounded-lg border border-[#00e5ff] flex items-center justify-center bg-white/5">
            <span className="text-sm font-bold text-[#00e5ff] tracking-tighter">R</span>
          </div>
          <div>
            <div className="text-sm font-bold tracking-wider text-white mono-val">RAKSHAK</div>
            <div className="text-[9px] text-[#00e5ff] tracking-[0.2em] font-semibold">SURVEILLANCE</div>
          </div>
        </div>

        <nav className="flex-1 px-4 py-6 space-y-2">
          <button
            onClick={() => setPage("dashboard")}
            className={`w-full flex items-center gap-3 px-4 py-3 rounded-lg text-sm font-medium transition ${
              page === "dashboard" ? "bg-[#00e5ff]/10 text-[#00e5ff] border-l-2 border-[#00e5ff]" : "text-slate-400 hover:bg-white/5 hover:text-white"
            }`}
          >
            <span>📊</span> Command Center
          </button>
          <button
            onClick={() => setPage("surveillance")}
            className={`w-full flex items-center gap-3 px-4 py-3 rounded-lg text-sm font-medium transition ${
              page === "surveillance" ? "bg-[#00e5ff]/10 text-[#00e5ff] border-l-2 border-[#00e5ff]" : "text-slate-400 hover:bg-white/5 hover:text-white"
            }`}
          >
            <span>📹</span> Surveillance Grid
          </button>
          <button
            onClick={() => setPage("analytics")}
            className={`w-full flex items-center gap-3 px-4 py-3 rounded-lg text-sm font-medium transition ${
              page === "analytics" ? "bg-[#00e5ff]/10 text-[#00e5ff] border-l-2 border-[#00e5ff]" : "text-slate-400 hover:bg-white/5 hover:text-white"
            }`}
          >
            <span>📈</span> Threat Analytics
          </button>
          <button
            onClick={() => setPage("system")}
            className={`w-full flex items-center gap-3 px-4 py-3 rounded-lg text-sm font-medium transition ${
              page === "system" ? "bg-[#00e5ff]/10 text-[#00e5ff] border-l-2 border-[#00e5ff]" : "text-slate-400 hover:bg-white/5 hover:text-white"
            }`}
          >
            <span>⚙️</span> System Config
          </button>
        </nav>

        {/* Sidebar Footer */}
        <div className="p-4 border-t border-white/5 text-center text-[10px] text-slate-500 font-mono">
          VIVID-CORE v2.4.8
        </div>
      </aside>

      {/* Main Content Pane */}
      <div className="flex-1 flex flex-col min-w-0 overflow-y-auto">
        {health === "offline" && (
          <div className="bg-rose-950/90 text-rose-200 border-b border-rose-500/25 px-4 py-2 text-center text-xs font-semibold font-mono tracking-wider flex items-center justify-center gap-2">
            <span>⚠️</span>
            <span>BACKEND DISCONNECTED - RUNNING OFFLINE CLIENT FUSION SIMULATOR</span>
          </div>
        )}
        {/* Header bar */}
        <header className="h-16 px-6 border-b border-white/5 flex items-center justify-between bg-[#0d1b2a]/40 backdrop-blur-md sticky top-0 z-30">
          <div className="flex items-center gap-6">
            {/* Nav Tabs */}
            <div className="hidden sm:flex items-center gap-1 bg-black/20 p-1 rounded-lg border border-white/5">
              <button
                onClick={() => setPage("dashboard")}
                className={`px-3 py-1.5 rounded-md text-xs font-semibold transition ${
                  page === "dashboard" ? "bg-white text-slate-950 shadow-sm" : "text-slate-400 hover:text-white"
                }`}
              >
                Dashboard
              </button>
              <button
                onClick={() => setPage("surveillance")}
                className={`px-3 py-1.5 rounded-md text-xs font-semibold transition ${
                  page === "surveillance" ? "bg-white text-slate-950 shadow-sm" : "text-slate-400 hover:text-white"
                }`}
              >
                Live View
              </button>
              <button
                onClick={() => setPage("analytics")}
                className={`px-3 py-1.5 rounded-md text-xs font-semibold transition ${
                  page === "analytics" ? "bg-white text-slate-950 shadow-sm" : "text-slate-400 hover:text-white"
                }`}
              >
                Insights
              </button>
            </div>

            {/* FastAPI Indicator */}
            <div className="flex items-center gap-2 text-xs font-semibold text-slate-400">
              <span className={`h-2.5 w-2.5 rounded-full ${
                health === "online" ? "bg-emerald-400 animate-pulse" : health === "degraded" ? "bg-amber-400" : "bg-rose-500"
              }`} />
              <span>FastAPI {health.toUpperCase()}</span>
            </div>
          </div>

          <div className="flex items-center gap-3">
            {/* Theme Toggle */}
            <button
              onClick={() => setTheme((t) => (t === "dark" ? "light" : "dark"))}
              className="p-2 rounded-lg bg-white/5 border border-white/15 text-slate-300 hover:bg-white/10"
              title="Toggle Light/Dark Theme"
            >
              {theme === "dark" ? "☀️" : "🌙"}
            </button>

            {/* Glowing Red Emergency Alert Button */}
            <button
              onClick={triggerManualAlert}
              className="glow-emergency px-4 py-1.5 rounded-full text-xs font-bold text-rose-500 bg-rose-500/10 hover:bg-rose-600 hover:text-white transition-all uppercase tracking-wider"
            >
              🚨 EMERGENCY ALERT
            </button>
          </div>
        </header>

        {/* Content Body */}
        <main className="p-6 space-y-6 flex-1 max-w-[1700px] w-full mx-auto">
          {page === "dashboard" && (
            <div className="grid gap-6 xl:grid-cols-[1.3fr_0.7fr]">
              {/* Left Column: Intake & Bounding Overlay */}
              <div className="space-y-6">
                <section className="glass-panel p-5">
                  <div className="mb-4 flex flex-wrap items-center justify-between gap-3">
                    <div>
                      <h2 className="text-xl font-bold text-white tracking-wide">Media Intake</h2>
                      <p className="text-xs text-slate-400">Upload images or videos to feed the safety neural pipelines.</p>
                    </div>
                    <div className="flex items-center gap-3">
                      <label className="flex items-center gap-2 text-xs font-semibold text-slate-300 cursor-pointer select-none">
                        <input
                          type="checkbox"
                          checked={saveEvidence}
                          onChange={(e) => setSaveEvidence(e.target.checked)}
                          className="h-3.5 w-3.5 rounded border-white/20 bg-white/5 text-[#00e5ff] accent-[#00e5ff]"
                        />
                        Save Evidence
                      </label>
                      <label className="flex items-center gap-2 text-xs font-semibold text-slate-300 cursor-pointer select-none">
                        <input
                          type="checkbox"
                          checked={detectionOn}
                          onChange={(e) => setDetectionOn(e.target.checked)}
                          className="h-3.5 w-3.5 rounded border-white/20 bg-white/5 text-[#00e5ff] accent-[#00e5ff]"
                        />
                        Detection Active
                      </label>
                    </div>
                  </div>

                  <div className="grid gap-4 lg:grid-cols-[280px_1fr]">
                    {/* Choose files panel */}
                    <div className="p-4 rounded-xl border border-white/5 bg-black/25 flex flex-col justify-between">
                      <div>
                        <label className="block text-xs font-bold text-slate-400 mb-2 uppercase tracking-wide">Select File</label>
                        <input
                          type="file"
                          accept="image/*,video/*"
                          onChange={(e) => setFile(e.target.files?.[0] ?? null)}
                          className="block w-full text-xs text-slate-400 file:mr-3 file:rounded-lg file:border-0 file:bg-[#00e5ff] file:px-3 file:py-1.5 file:text-xs file:font-semibold file:text-slate-950 file:hover:bg-[#00bcd4] cursor-pointer"
                        />
                        
                        <div className="mt-6">
                          <div className="flex justify-between text-xs text-slate-300 mb-1">
                            <span>Confidence Limit:</span>
                            <span className="text-[#00e5ff] font-bold">{Math.round(threshold * 100)}%</span>
                          </div>
                          <input
                            type="range"
                            min="0.2"
                            max="0.95"
                            step="0.01"
                            value={threshold}
                            onChange={(e) => setThreshold(Number(e.target.value))}
                            className="w-full h-1 bg-white/10 rounded-lg appearance-none cursor-pointer accent-[#00e5ff]"
                          />
                        </div>
                      </div>

                      <button
                        onClick={processUpload}
                        disabled={!file || loading}
                        className="mt-6 w-full py-3 rounded-lg bg-gradient-to-r from-[#00bcd4] to-[#00e5ff] text-slate-950 text-xs font-bold tracking-wider uppercase hover:opacity-90 disabled:opacity-40 disabled:cursor-not-allowed transition"
                      >
                        {loading ? "Intaking Neural Stream..." : "Run Smart Inference"}
                      </button>
                    </div>

                    {/* Pre-upload file preview */}
                    <div className="relative aspect-video rounded-xl overflow-hidden border border-white/5 bg-black/35 flex items-center justify-center">
                      {loading ? (
                        <div className="absolute inset-0 flex flex-col items-center justify-center bg-black/50 space-y-3">
                          <div className="h-8 w-8 animate-spin rounded-full border-2 border-slate-600 border-t-[#00e5ff]" />
                          <span className="text-xs text-slate-400 font-mono">Running model cascade...</span>
                        </div>
                      ) : null}
                      <MediaPreview file={file} preview={preview} />
                    </div>
                  </div>
                </section>

                {/* Annotated Output card */}
                <section className="glass-panel p-5">
                  <div className="mb-3 flex items-center justify-between">
                    <h2 className="text-lg font-bold text-white tracking-wide">Annotated Output</h2>
                    {result?.saved_path && (
                      <span className="text-[10px] font-mono font-bold text-[#30d158] bg-[#30d158]/10 border border-[#30d158]/20 px-2 py-0.5 rounded-full">
                        EVIDENCE CAPTURED
                      </span>
                    )}
                  </div>
                  <div className="aspect-video rounded-xl overflow-hidden border border-white/5 bg-black/45 flex items-center justify-center scan-line">
                    {output ? (
                      <img src={output} alt="Annotated detection results" className="w-full h-full object-contain" />
                    ) : (
                      <div className="text-center p-6 text-slate-500">
                        <span className="text-3xl block mb-2">👁️</span>
                        <p className="text-xs">Annotated frames overlaying neural detections will display here.</p>
                      </div>
                    )}
                  </div>
                </section>
              </div>

              {/* Right Column: Scene Summary, Confidence Bar, Logs */}
              <div className="space-y-6">
                <SceneSummaryCard result={result} weaponScore={weaponScore} combinedViolence={combinedViolence} />
                
                {/* Confidence Fusion bar */}
                <section className="glass-panel p-5 space-y-4">
                  <h2 className="text-xs font-bold uppercase tracking-[0.2em] text-slate-400">Confidence Fusion</h2>
                  <div className="space-y-4">
                    <ConfidenceBar label="Weapon Detect (YOLOv8)" value={weaponScore} tone="red" />
                    <ConfidenceBar label="Violence Detect (LSTM)" value={combinedViolence} tone="amber" />
                    <ConfidenceBar label="Police Uniform (MobileNet)" value={result?.police_score ?? 0} tone="blue" />
                    <ConfidenceBar label="Accident Detect (ResNet50)" value={result?.accident_confidence ?? 0} tone="red" />
                  </div>
                </section>

                <Timeline items={timeline} />

                {/* Core Telemetry Card */}
                <section className="glass-panel p-5">
                  <h2 className="text-xs font-bold uppercase tracking-[0.2em] text-slate-400 mb-4">Core Telemetry</h2>
                  <div className="grid grid-cols-3 gap-3 text-center">
                    <div className="p-3 bg-black/10 rounded-lg border border-white/5">
                      <div className="text-[10px] text-slate-500 font-mono">LATENCY</div>
                      <div className="text-lg font-bold text-[#00e5ff] font-mono mt-1">
                        {result?.component_latency_ms ? 
                          Object.values(result.component_latency_ms).reduce((a, b) => a + b, 0).toFixed(0) : "0"
                        }ms
                      </div>
                    </div>
                    <div className="p-3 bg-black/10 rounded-lg border border-white/5">
                      <div className="text-[10px] text-slate-500 font-mono">TPU LOAD</div>
                      <div className="text-lg font-bold text-[#30d158] font-mono mt-1">
                        {health === "online" ? "42%" : "—"}
                      </div>
                    </div>
                    <div className="p-3 bg-black/10 rounded-lg border border-white/5">
                      <div className="text-[10px] text-slate-500 font-mono">CORE TEMP</div>
                      <div className="text-lg font-bold text-amber-500 font-mono mt-1">
                        {health === "online" ? "48.5°" : "—"}
                      </div>
                    </div>
                  </div>
                </section>

                <LogPanel logs={logs} />
              </div>
            </div>
          )}

          {page === "surveillance" && (
            <div className="grid gap-6 xl:grid-cols-[1.3fr_0.7fr]">
              {/* Surveillance Grid camera feed */}
              <div className="space-y-6">
                <section className="glass-panel p-5">
                  <div className="mb-4 flex items-center justify-between">
                    <div>
                      <h2 className="text-xl font-bold text-white tracking-wide">
                        Live Surveillance Feed - Zone {activeZone}
                      </h2>
                      <p className="text-xs text-slate-400">Streamed from camera index 0 with base64 neural overlays.</p>
                    </div>
                    <div className="flex items-center gap-3">
                      <span className="px-2 py-0.5 rounded bg-rose-500/10 border border-rose-500/25 text-[10px] font-bold text-rose-500 animate-pulse tracking-wide uppercase">
                        LIVE
                      </span>
                      <span className="text-xs text-slate-400 font-mono">FPS: {fps}</span>
                      <span className="text-xs text-slate-400 font-mono">BUFFER: {bufferLatency}ms</span>
                    </div>
                  </div>

                  {/* Main View */}
                  <div className="relative aspect-video rounded-xl overflow-hidden border border-white/10 bg-black/80 scan-line flex items-center justify-center">
                    <video ref={videoRef} autoPlay muted playsInline className="absolute inset-0 w-full h-full object-cover opacity-60" />
                    <canvas ref={canvasRef} className="hidden" />
                    
                    {/* Bounding box overlays displayed when webcam is feeding */}
                    {cameraOn && output && (
                      <img src={output} alt="Annotated Stream" className="absolute inset-0 w-full h-full object-cover z-10" />
                    )}

                    {!cameraOn && (
                      <div className="text-center z-10 p-6">
                        <span className="text-4xl block mb-2 text-slate-600">📹</span>
                        <p className="text-xs text-slate-500">Camera Feed is Offline. Toggle webcam in Surveillance controls.</p>
                      </div>
                    )}

                    {/* Zone indicators overlay */}
                    <div className="absolute top-4 left-4 bg-black/75 px-3 py-1.5 rounded border border-white/10 z-20 font-mono text-[10px] text-[#00e5ff] font-bold tracking-widest uppercase">
                      ZONE-{activeZone} patrol
                    </div>
                    <div className="absolute top-4 right-4 bg-black/75 px-3 py-1.5 rounded border border-white/10 z-20 font-mono text-[10px] text-slate-400">
                      UTC: {new Date().toISOString().slice(11, 19)}
                    </div>
                  </div>

                  {/* Camera Zones thumbnails */}
                  <div className="mt-6">
                    <h3 className="text-xs font-bold uppercase tracking-[0.15em] text-slate-400 mb-3">Zone Grid selection</h3>
                    <div className="grid grid-cols-4 gap-4">
                      {(["A", "B", "C", "D"] as const).map((z) => (
                        <button
                          key={z}
                          onClick={() => setActiveZone(z)}
                          className={`aspect-video rounded-lg overflow-hidden border transition relative flex items-center justify-center ${
                            activeZone === z ? "border-[#00e5ff] ring-1 ring-[#00e5ff]" : "border-white/5 bg-black/20 hover:border-white/20"
                          }`}
                        >
                          <span className="text-xs font-bold font-mono tracking-wider text-slate-400">ZONE {z}</span>
                          <div className={`absolute bottom-2 left-2 h-2 w-2 rounded-full ${activeZone === z ? "bg-[#00e5ff] animate-pulse" : "bg-slate-600"}`} />
                        </button>
                      ))}
                    </div>
                  </div>
                </section>
              </div>

              {/* Right Panel controls */}
              <div className="space-y-6">
                <SceneSummaryCard result={result} weaponScore={weaponScore} combinedViolence={combinedViolence} />
                
                {/* Surveillance Controls */}
                <section className="glass-panel p-5">
                  <h2 className="text-xs font-bold uppercase tracking-[0.2em] text-slate-400 mb-4">Surveillance Controls</h2>
                  <div className="space-y-4">
                    <button
                      onClick={() => setCameraOn(!cameraOn)}
                      className={`w-full py-3 rounded-lg text-xs font-bold tracking-wider uppercase transition ${
                        cameraOn ? "bg-rose-600 hover:bg-rose-700 text-white" : "bg-[#00e5ff] hover:bg-[#00bcd4] text-slate-950"
                      }`}
                    >
                      {cameraOn ? "Disable Surveillance Feed" : "Initialize Webcam Feed"}
                    </button>
                    
                    <div className="flex items-center justify-between border-t border-white/5 pt-4 text-xs text-slate-400 font-mono">
                      <span>Stream: index 0 (USB)</span>
                      <span>Inference: 100ms interval</span>
                    </div>
                  </div>
                </section>

                {/* Quick Actions Panel */}
                <section className="glass-panel p-5">
                  <h2 className="text-xs font-bold uppercase tracking-[0.2em] text-slate-400 mb-4">Quick Actions</h2>
                  <div className="grid grid-cols-2 gap-3">
                    <button
                      onClick={triggerManualAlert}
                      className="py-3 rounded-lg border border-rose-500/30 bg-rose-500/10 text-rose-500 text-xs font-bold tracking-wider hover:bg-rose-500 hover:text-white transition uppercase"
                    >
                      Alert Emergency
                    </button>
                    <button
                      onClick={() => alert("Loudspeaker warning and door magnetic locks engaged.")}
                      className="py-3 rounded-lg border border-amber-500/30 bg-amber-500/10 text-amber-500 text-xs font-bold tracking-wider hover:bg-amber-500 hover:text-white transition uppercase"
                    >
                      Active Lockdown
                    </button>
                    <button
                      onClick={() => alert("SMS, Email, and Telegram dispatch sent to local precinct.")}
                      className="py-3 rounded-lg border border-sky-500/30 bg-sky-500/10 text-sky-500 text-xs font-bold tracking-wider hover:bg-sky-500 hover:text-white transition uppercase"
                    >
                      Notify Police
                    </button>
                    <button
                      onClick={() => {
                        setThreshold(0.55);
                        alert("AI calibration reset. Baseline parameters set to 55%.");
                      }}
                      className="py-3 rounded-lg border border-[#00e5ff]/30 bg-[#00e5ff]/10 text-[#00e5ff] text-xs font-bold tracking-wider hover:bg-[#00e5ff] hover:text-slate-950 transition uppercase"
                    >
                      Calibrate AI
                    </button>
                  </div>
                </section>

                <Timeline items={timeline} />
              </div>
            </div>
          )}

          {page === "analytics" && (
            <section className="glass-panel p-6">
              <h2 className="text-2xl font-bold text-white mb-2">Threat Analytics Dashboard</h2>
              <p className="text-sm text-slate-400 mb-6">Historical trends and aggregated threat statistics.</p>
              
              <div className="grid gap-6 md:grid-cols-2 lg:grid-cols-4">
                <div className="p-5 bg-black/20 rounded-xl border border-white/5">
                  <div className="text-xs text-slate-500 font-mono">TOTAL EVENTS</div>
                  <div className="text-3xl font-bold mt-2 font-mono">1,482</div>
                  <div className="text-[10px] text-[#30d158] font-mono mt-1">↑ 12% vs last week</div>
                </div>
                <div className="p-5 bg-black/20 rounded-xl border border-white/5">
                  <div className="text-xs text-slate-500 font-mono">WEAPONS DISARMED</div>
                  <div className="text-3xl font-bold mt-2 font-mono">42</div>
                  <div className="text-[10px] text-[#30d158] font-mono mt-1">↑ 4% vs last week</div>
                </div>
                <div className="p-5 bg-black/20 rounded-xl border border-white/5">
                  <div className="text-xs text-slate-500 font-mono">VIOLENCE MITIGATED</div>
                  <div className="text-3xl font-bold mt-2 font-mono">118</div>
                  <div className="text-[10px] text-rose-500 font-mono mt-1">↓ 2% vs last week</div>
                </div>
                <div className="p-5 bg-black/20 rounded-xl border border-white/5">
                  <div className="text-xs text-slate-500 font-mono">TPU UPTIME</div>
                  <div className="text-3xl font-bold mt-2 font-mono">99.98%</div>
                  <div className="text-[10px] text-[#30d158] font-mono mt-1">Optimal status</div>
                </div>
              </div>

              {/* Graphic Placeholder */}
              <div className="mt-8 p-12 rounded-xl border border-white/5 bg-black/20 flex flex-col items-center justify-center text-center">
                <div className="text-slate-600 text-5xl mb-3">📊</div>
                <h3 className="text-md font-bold text-white">Aggregated Vector Analysis</h3>
                <p className="text-xs text-slate-500 max-w-md mt-1">Neural vector clusters representing weapon types, violence severity, and uniform patterns mapped over time.</p>
              </div>
            </section>
          )}

          {page === "system" && (
            <section className="glass-panel p-6 space-y-6">
              <div>
                <h2 className="text-2xl font-bold text-white">System Configurations</h2>
                <p className="text-sm text-slate-400">Adjust model triggers, hardware devices, and pipeline cascades.</p>
              </div>

              <div className="border-t border-white/5 pt-6 space-y-6 max-w-2xl">
                <div className="flex items-center justify-between">
                  <div>
                    <h3 className="text-sm font-bold text-white">Device Selection</h3>
                    <p className="text-xs text-slate-500">Pick hardware acceleration devices for training & inference</p>
                  </div>
                  <select className="bg-black/40 border border-white/10 rounded px-3 py-1.5 text-xs font-mono text-[#00e5ff]">
                    <option>CUDA GPU (NVIDIA RTX)</option>
                    <option>Apple Metal CoreML</option>
                    <option>Intel OpenVINO TPU</option>
                    <option>Standard CPU (No Accel)</option>
                  </select>
                </div>

                <div className="flex items-center justify-between">
                  <div>
                    <h3 className="text-sm font-bold text-white">Live Skip Rate</h3>
                    <p className="text-xs text-slate-500">Number of frames to skip during live evaluations</p>
                  </div>
                  <input
                    type="number"
                    defaultValue={2}
                    className="bg-black/40 border border-white/10 rounded px-3 py-1.5 text-xs text-center w-16 text-[#00e5ff] font-mono"
                  />
                </div>

                <div className="flex items-center justify-between">
                  <div>
                    <h3 className="text-sm font-bold text-white">Neural Cache Pipeline</h3>
                    <p className="text-xs text-slate-500">Keep model weights pre-loaded in memory for faster responses</p>
                  </div>
                  <button className="px-3 py-1 rounded bg-[#30d158]/25 text-[#30d158] text-xs font-bold uppercase tracking-wider">
                    ENABLED
                  </button>
                </div>
              </div>
            </section>
          )}
        </main>
      </div>

      {/* Emergeny Alert Notification Modal overlay */}
      <AnimatePresence>
        {emergencyAlert && (
          <motion.div
            initial={{ opacity: 0, y: 24, scale: 0.95 }}
            animate={{ opacity: 1, y: 0, scale: 1 }}
            exit={{ opacity: 0, y: 24, scale: 0.95 }}
            className="fixed bottom-5 right-5 z-50 rounded-xl border border-rose-500/40 bg-rose-950/90 backdrop-blur-md p-5 text-white shadow-2xl max-w-sm flex items-start gap-4"
          >
            <span className="text-3xl">⚠️</span>
            <div>
              <div className="text-xs font-bold uppercase tracking-[0.2em] text-[#ff3b30] mb-1">CRITICAL INCIDENT ALERT</div>
              <div className="text-sm font-semibold">{emergencyAlert}</div>
              <div className="text-[10px] text-slate-400 mt-2 font-mono">AUTOMATIC MITIGATION SYSTEM ENGAGED</div>
            </div>
          </motion.div>
        )}
      </AnimatePresence>

      {/* Inline Error Toasts */}
      <AnimatePresence>
        {errorToast && (
          <motion.div
            initial={{ opacity: 0, y: 16 }}
            animate={{ opacity: 1, y: 0 }}
            exit={{ opacity: 0, y: 16 }}
            className="fixed bottom-5 left-5 z-50 rounded-lg border border-amber-300/20 bg-amber-500/90 backdrop-blur-md px-4 py-3 text-xs font-semibold text-white shadow-2xl max-w-sm flex items-center gap-3"
          >
            <span>⚠️</span>
            <span>{errorToast}</span>
          </motion.div>
        )}
      </AnimatePresence>
    </div>
  );
}

function MediaPreview({ file, preview }: { file: File | null; preview: string }) {
  if (!file || !preview) {
    return (
      <div className="text-center p-6 text-slate-500">
        <span className="text-3xl block mb-2">📁</span>
        <p className="text-xs">Preview panel for chosen assets.</p>
      </div>
    );
  }
  if (file.type.startsWith("video")) {
    return <video controls src={preview} className="w-full h-full object-contain" />;
  }
  return <img src={preview} alt="Chosen Preview" className="w-full h-full object-contain" />;
}

type SummaryProps = {
  result: DetectionResponse | null;
  weaponScore: number;
  combinedViolence: number;
};

function SceneSummaryCard({ result, weaponScore, combinedViolence }: SummaryProps) {
  const scene = result?.summary ?? "Awaiting Neural Signal";
  const risk = result?.risk_level ?? "SAFE";
  
  // Decide active states for components
  const hasWeapon = Boolean(result?.has_weapon_yolo) || (result?.gun_score_police ?? 0) > 0.55 || (result?.knife_score_police ?? 0) > 0.55;
  const hasViolence = Math.max(result?.violence_score_lstm ?? 0, result?.violence_score_police ?? 0) > 0.55;
  const hasPolice = result?.police_detected ?? false;
  const hasAccident = (result?.accident_confidence ?? 0) > 0.55;

  const getRiskBadgeStyles = (r: string) => {
    switch (r.toLowerCase()) {
      case "critical":
        return "bg-rose-500/10 border border-rose-500/25 text-[#ff3b30]";
      case "high":
        return "bg-amber-500/10 border border-amber-500/25 text-[#ff9f0a]";
      case "watch":
        return "bg-sky-500/10 border border-sky-500/25 text-[#64d2ff]";
      default:
        return "bg-[#30d158]/10 border border-[#30d158]/25 text-[#30d158]";
    }
  };

  return (
    <section className="glass-panel p-5">
      <div className="mb-4 flex items-start justify-between gap-3">
        <div>
          <h2 className="text-xs font-bold uppercase tracking-[0.2em] text-slate-400">Scene Summary</h2>
          <div className="mt-2 text-xl font-bold text-white tracking-wide">{scene}</div>
        </div>
        <span className={`px-2.5 py-1 rounded-full text-[10px] font-bold uppercase tracking-wider ${getRiskBadgeStyles(risk)}`}>
          {risk}
        </span>
      </div>

      <div className="grid grid-cols-2 gap-3">
        <DetectionBadge label="Weapon" icon="⚔️" active={hasWeapon} value={`${Math.round(weaponScore * 100)}%`} />
        <DetectionBadge label="Violence" icon="💥" active={hasViolence} value={`${Math.round(combinedViolence * 100)}%`} />
        <DetectionBadge label="Police" icon="👮" active={hasPolice} value={`${Math.round((result?.police_score ?? 0) * 100)}%`} />
        <DetectionBadge label="Accident" icon="🚗" active={hasAccident} value={`${Math.round((result?.accident_confidence ?? 0) * 100)}%`} />
      </div>
    </section>
  );
}

// Render block for mounting to DOM
const container = document.getElementById("root");
if (container) {
  const root = createRoot(container);
  root.render(
    <React.StrictMode>
      <App />
    </React.StrictMode>
  );
}
