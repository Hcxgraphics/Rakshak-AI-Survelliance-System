import React, { useEffect, useMemo, useRef, useState } from "react";
import { createRoot } from "react-dom/client";
import { AnimatePresence, motion } from "framer-motion";
import { ConfidenceBar } from "./components/ConfidenceBar";
import { DetectionBadge } from "./components/DetectionBadge";
import { LogPanel } from "./components/LogPanel";
import { Timeline } from "./components/Timeline";
import { DetectionResponse, fetchHealth, fetchLogs, imageSrc, liveDetect, LogItem, uploadMedia } from "./lib/api";
import "./styles/global.css";

type Mode = "upload" | "camera";

const alertRisks = new Set(["critical", "high"]);

function App() {
  const [mode, setMode] = useState<Mode>("upload");
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
  const [error, setError] = useState("");
  const [health, setHealth] = useState("checking");
  const [alertMessage, setAlertMessage] = useState("");
  const videoRef = useRef<HTMLVideoElement | null>(null);
  const canvasRef = useRef<HTMLCanvasElement | null>(null);
  const streamRef = useRef<MediaStream | null>(null);
  const frameIndex = useRef(0);
  const busyRef = useRef(false);

  const mediaKind = useMemo(() => {
    if (!file) return "auto";
    return file.type.startsWith("video") ? "video" : "image";
  }, [file]);

  useEffect(() => {
    fetchHealth()
      .then((data) => setHealth(data.status === "ok" ? "online" : "degraded"))
      .catch(() => setHealth("offline"));
    const id = window.setInterval(() => {
      fetchLogs().then((data) => setLogs(data.items)).catch(() => undefined);
    }, 2200);
    return () => window.clearInterval(id);
  }, []);

  useEffect(() => {
    if (!file) {
      setPreview("");
      return;
    }
    const url = URL.createObjectURL(file);
    setPreview(url);
    return () => URL.revokeObjectURL(url);
  }, [file]);

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
      .catch((exc) => setError(`Camera unavailable: ${exc.message}`));

    return () => {
      cancelled = true;
      streamRef.current?.getTracks().forEach((track) => track.stop());
      streamRef.current = null;
    };
  }, [cameraOn]);

  useEffect(() => {
    if (!cameraOn) return undefined;
    const id = window.setInterval(async () => {
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
          const data = await liveDetect(blob, threshold, detectionOn, frameIndex.current++, saveEvidence);
          setResult(data);
          setTimeline((items) => [...items.slice(-30), data]);
          triggerAlert(data);
        } catch (exc) {
          setError(exc instanceof Error ? exc.message : "Live detection failed");
        } finally {
          busyRef.current = false;
        }
      }, "image/jpeg", 0.82);
    }, 1200);
    return () => window.clearInterval(id);
  }, [cameraOn, detectionOn, threshold, saveEvidence]);

  function triggerAlert(data: DetectionResponse) {
    if (!alertRisks.has(data.risk_level)) return;
    setAlertMessage(data.summary);
    window.setTimeout(() => setAlertMessage(""), 3400);
    try {
      const audio = new Audio("data:audio/wav;base64,UklGRiQAAABXQVZFZm10IBAAAAABAAEAESsAACJWAAACABAAZGF0YQAAAAA=");
      void audio.play();
    } catch {
      return;
    }
  }

  async function processUpload() {
    if (!file) return;
    setLoading(true);
    setError("");
    try {
      const data = await uploadMedia(file, mediaKind as "auto" | "image" | "video", threshold, saveEvidence);
      setResult(data);
      setTimeline((items) => [...items.slice(-30), data]);
      triggerAlert(data);
    } catch (exc) {
      setError(exc instanceof Error ? exc.message : "Upload failed");
    } finally {
      setLoading(false);
    }
  }

  const output = imageSrc(result);
  const combinedViolence = Math.max(result?.violence_score_lstm ?? 0, result?.violence_score_police ?? 0);
  const weaponScore = Math.max(
    result?.weapon_detections?.reduce((score, detection) => Math.max(score, detection.confidence), 0) ?? 0,
    result?.gun_score_police ?? 0,
    result?.knife_score_police ?? 0,
    result?.has_weapon_yolo ? 0.75 : 0,
  );

  return (
    <main className="min-h-screen px-4 py-5 text-slate-100 sm:px-6 lg:px-8">
      <div className="mx-auto max-w-[1500px]">
        <header className="mb-5 flex flex-col gap-4 lg:flex-row lg:items-center lg:justify-between">
          <div>
            <div className="mb-2 flex items-center gap-3 text-sm text-slate-400">
              <span className={`h-2.5 w-2.5 rounded-full ${health === "online" ? "bg-emerald-400" : health === "degraded" ? "bg-amber-300" : "bg-rose-400"}`} />
              <span>FastAPI {health}</span>
            </div>
            <h1 className="text-3xl font-semibold tracking-normal text-white md:text-5xl">AI Public Safety Command Center</h1>
            <p className="mt-2 max-w-2xl text-sm leading-6 text-slate-400">
              Local multi-model surveillance dashboard with adaptive routing, evidence capture, and live inference.
            </p>
          </div>
          <div className="glass flex flex-wrap items-center gap-3 rounded-lg p-2">
            <button onClick={() => setMode("upload")} className={`rounded-md px-4 py-2 text-sm font-medium ${mode === "upload" ? "bg-white text-slate-950" : "text-slate-300 hover:bg-white/10"}`}>
              Upload
            </button>
            <button onClick={() => setMode("camera")} className={`rounded-md px-4 py-2 text-sm font-medium ${mode === "camera" ? "bg-white text-slate-950" : "text-slate-300 hover:bg-white/10"}`}>
              Camera
            </button>
          </div>
        </header>

        <div className="grid gap-5 xl:grid-cols-[1.4fr_0.8fr]">
          <section className="glass rounded-lg p-4">
            <div className="mb-4 flex flex-wrap items-center justify-between gap-3">
              <div>
                <h2 className="text-lg font-semibold text-white">{mode === "upload" ? "Media Intake" : "Live Camera"}</h2>
                <p className="text-sm text-slate-400">{mode === "upload" ? "Preview before processing image or video uploads." : "Frame-by-frame inference from the local webcam."}</p>
              </div>
              <div className="flex flex-wrap items-center gap-3">
                <label className="flex items-center gap-2 text-sm text-slate-300">
                  <input type="checkbox" checked={saveEvidence} onChange={(event) => setSaveEvidence(event.target.checked)} className="h-4 w-4 accent-cyan-300" />
                  Save Evidence
                </label>
                <label className="flex items-center gap-2 text-sm text-slate-300">
                  <input type="checkbox" checked={detectionOn} onChange={(event) => setDetectionOn(event.target.checked)} className="h-4 w-4 accent-cyan-300" />
                  Detection
                </label>
              </div>
            </div>

            {mode === "upload" ? (
              <div className="grid gap-4 lg:grid-cols-[320px_1fr]">
                <div className="rounded-lg border border-dashed border-white/15 bg-white/[0.03] p-4">
                  <input
                    type="file"
                    accept="image/*,video/*"
                    onChange={(event) => setFile(event.target.files?.[0] ?? null)}
                    className="block w-full text-sm text-slate-300 file:mr-3 file:rounded-md file:border-0 file:bg-cyan-300 file:px-4 file:py-2 file:text-sm file:font-semibold file:text-slate-950"
                  />
                  <button
                    onClick={processUpload}
                    disabled={!file || loading}
                    className="mt-4 w-full rounded-md bg-white px-4 py-3 text-sm font-semibold text-slate-950 transition hover:bg-cyan-100 disabled:cursor-not-allowed disabled:opacity-45"
                  >
                    {loading ? "Processing..." : "Run Smart Inference"}
                  </button>
                  <div className="mt-5">
                    <label className="text-sm text-slate-300">Confidence Threshold {Math.round(threshold * 100)}%</label>
                    <input type="range" min="0.2" max="0.95" step="0.01" value={threshold} onChange={(event) => setThreshold(Number(event.target.value))} className="mt-2 w-full accent-cyan-300" />
                  </div>
                </div>
                <MediaPreview file={file} preview={preview} />
              </div>
            ) : (
              <div className="grid gap-4 lg:grid-cols-[1fr_280px]">
                <div className="relative overflow-hidden rounded-lg border border-white/10 bg-black">
                  <video ref={videoRef} autoPlay muted playsInline className="aspect-video w-full object-cover" />
                  <canvas ref={canvasRef} className="hidden" />
                  {!cameraOn ? <div className="absolute inset-0 grid place-items-center text-slate-500">Camera is off</div> : null}
                </div>
                <div className="rounded-lg border border-white/10 bg-white/[0.03] p-4">
                  <button onClick={() => setCameraOn((value) => !value)} className={`w-full rounded-md px-4 py-3 text-sm font-semibold ${cameraOn ? "bg-rose-400 text-white" : "bg-cyan-300 text-slate-950"}`}>
                    {cameraOn ? "Camera OFF" : "Camera ON"}
                  </button>
                  <div className="mt-5">
                    <label className="text-sm text-slate-300">Confidence Threshold {Math.round(threshold * 100)}%</label>
                    <input type="range" min="0.2" max="0.95" step="0.01" value={threshold} onChange={(event) => setThreshold(Number(event.target.value))} className="mt-2 w-full accent-cyan-300" />
                  </div>
                  <p className="mt-4 text-xs leading-5 text-slate-500">Live mode sends compressed frames to `/live-detect` and overlays the annotated response below.</p>
                </div>
              </div>
            )}
          </section>

          <aside className="space-y-5">
            <SceneSummary result={result} />
            <Timeline items={timeline} />
          </aside>
        </div>

        <div className="mt-5 grid gap-5 xl:grid-cols-[1.4fr_0.8fr]">
          <section className="glass rounded-lg p-4">
            <div className="mb-3 flex items-center justify-between">
              <h2 className="text-lg font-semibold text-white">Annotated Output</h2>
              {result?.saved_path ? <span className="rounded-full bg-emerald-400/15 px-3 py-1 text-xs text-emerald-200">Saved</span> : null}
            </div>
            <div className="overflow-hidden rounded-lg border border-white/10 bg-black">
              {output ? <img src={output} alt="Annotated detection result" className="aspect-video w-full object-contain" /> : <div className="grid aspect-video place-items-center text-slate-500">Output appears here</div>}
            </div>
          </section>
          <section className="space-y-5">
            <div className="glass rounded-lg p-4">
              <h2 className="mb-3 text-sm font-semibold uppercase tracking-wider text-slate-300">Confidence Fusion</h2>
              <div className="space-y-4">
                <ConfidenceBar label="Weapon" value={weaponScore} tone="red" />
                <ConfidenceBar label="Violence" value={combinedViolence} tone="amber" />
                <ConfidenceBar label="Police" value={result?.police_score ?? 0} tone="blue" />
                <ConfidenceBar label="Accident" value={result?.accident_confidence ?? 0} tone="red" />
              </div>
            </div>
            <LogPanel logs={logs} />
          </section>
        </div>
      </div>

      <AnimatePresence>
        {alertMessage ? (
          <motion.div initial={{ opacity: 0, y: 24 }} animate={{ opacity: 1, y: 0 }} exit={{ opacity: 0, y: 24 }} className="fixed bottom-5 right-5 z-20 rounded-lg border border-rose-300/40 bg-rose-500/90 px-5 py-4 text-white shadow-2xl">
            <div className="text-sm font-semibold">Safety Alert</div>
            <div className="text-sm">{alertMessage}</div>
          </motion.div>
        ) : null}
      </AnimatePresence>

      {error ? <div className="fixed left-5 bottom-5 max-w-md rounded-lg border border-amber-300/40 bg-amber-500/15 px-4 py-3 text-sm text-amber-100">{error}</div> : null}
    </main>
  );
}

function MediaPreview({ file, preview }: { file: File | null; preview: string }) {
  if (!file || !preview) {
    return <div className="grid min-h-[320px] place-items-center rounded-lg border border-white/10 bg-black/30 text-slate-500">Choose an image or video</div>;
  }
  if (file.type.startsWith("video")) {
    return <video controls src={preview} className="aspect-video w-full rounded-lg border border-white/10 bg-black object-contain" />;
  }
  return <img src={preview} alt="Upload preview" className="aspect-video w-full rounded-lg border border-white/10 bg-black object-contain" />;
}

function SceneSummary({ result }: { result: DetectionResponse | null }) {
  const scene = result?.summary ?? "Awaiting signal";
  const risk = result?.risk_level ?? "low";
  return (
    <div className="glass rounded-lg p-4">
      <div className="mb-4 flex items-start justify-between gap-3">
        <div>
          <h2 className="text-sm font-semibold uppercase tracking-wider text-slate-300">Scene Summary</h2>
          <div className="mt-2 text-2xl font-semibold text-white">{scene}</div>
        </div>
        <span className={`rounded-full px-3 py-1 text-xs font-semibold ${risk === "critical" ? "bg-rose-400 text-white" : risk === "high" ? "bg-amber-300 text-slate-950" : "bg-emerald-300 text-slate-950"}`}>
          {risk}
        </span>
      </div>
      <div className="grid grid-cols-2 gap-3">
        <DetectionBadge label="Weapon" icon="W" active={Boolean(result?.has_weapon_yolo) || (result?.gun_score_police ?? 0) > 0.55 || (result?.knife_score_police ?? 0) > 0.55} value={`${Math.round(weaponScore * 100)}%`} />
        <DetectionBadge label="Violence" icon="!" active={Math.max(result?.violence_score_lstm ?? 0, result?.violence_score_police ?? 0) > 0.55} value={`${Math.round(Math.max(result?.violence_score_lstm ?? 0, result?.violence_score_police ?? 0) * 100)}%`} />
        <DetectionBadge label="Police" icon="P" active={result?.police_detected ?? false} value={`${Math.round((result?.police_score ?? 0) * 100)}%`} />
        <DetectionBadge label="Accident" icon="A" active={(result?.accident_confidence ?? 0) > 0.55} value={`${Math.round((result?.accident_confidence ?? 0) * 100)}%`} />
      </div>
    </div>
  );
}

createRoot(document.getElementById("root") as HTMLElement).render(
  <React.StrictMode>
    <App />
  </React.StrictMode>,
);
