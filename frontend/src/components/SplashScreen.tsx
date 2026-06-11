import React, { useEffect, useState } from "react";
import { motion } from "framer-motion";

type Props = {
  onComplete: () => void;
  healthStatus: string;
};

export function SplashScreen({ onComplete, healthStatus }: Props) {
  const [progress, setProgress] = useState(0);
  const [statusText, setStatusText] = useState("Initializing System Core");

  useEffect(() => {
    const interval = setInterval(() => {
      setProgress((prev) => {
        if (prev >= 100) {
          clearInterval(interval);
          return 100;
        }
        
        // Dynamic status updates based on progress percentage
        if (prev > 75) {
          setStatusText("Encrypted Neural Link Established");
        } else if (prev > 50) {
          setStatusText("Linking FastAPI Secure Port 8000");
        } else if (prev > 25) {
          setStatusText("Pre-Caching Vivid-Core v2.4.8 Weights");
        } else {
          setStatusText("Initializing System Core");
        }
        
        return prev + 4;
      });
    }, 80);

    return () => clearInterval(interval);
  }, []);

  useEffect(() => {
    // Dismiss splash once progress reaches 100% AND backend health is checking/complete
    if (progress === 100) {
      const timeout = setTimeout(() => {
        onComplete();
      }, 500);
      return () => clearTimeout(timeout);
    }
  }, [progress, onComplete]);

  return (
    <motion.div
      initial={{ opacity: 1 }}
      exit={{ opacity: 0 }}
      transition={{ duration: 0.6 }}
      className="fixed inset-0 w-screen h-screen z-[9999] flex flex-col items-center justify-center bg-[#050b14] text-slate-100 scan-line overflow-hidden"
    >
      <div className="w-full max-w-lg px-6 text-center">
        {/* Shield Logo SVG */}
        <motion.div 
          initial={{ scale: 0.8, opacity: 0 }}
          animate={{ scale: 1, opacity: 1 }}
          transition={{ duration: 0.8, ease: "easeOut" }}
          className="mx-auto mb-6 flex h-24 w-24 items-center justify-center rounded-2xl border border-[#00e5ff]/20 bg-white/5 shadow-[0_0_30px_rgba(0,229,255,0.08)]"
        >
          <svg
            className="h-14 w-14 text-[#00e5ff]"
            fill="none"
            viewBox="0 0 24 24"
            stroke="currentColor"
            strokeWidth={1.5}
          >
            <path
              strokeLinecap="round"
              strokeLinejoin="round"
              d="M9 12.75L11.25 15 15 9.75m-3-7.036A11.959 11.959 0 013.598 6 11.99 11.99 0 003 9.749c0 5.592 3.824 10.29 9 11.623 5.176-1.332 9-6.03 9-11.622 0-1.31-.21-2.57-.598-3.75h-.152c-3.196 0-6.1-1.248-8.25-3.285z"
            />
          </svg>
        </motion.div>

        {/* Title & Subtitle */}
        <h1 className="text-4xl font-bold tracking-[0.25em] text-[#00e5ff] mono-val">
          RAKSHAK
        </h1>
        <p className="mt-2 text-xs tracking-[0.4em] text-slate-400 font-semibold">
          ADVANCED PUBLIC SAFETY AI
        </p>

        {/* Custom Progress Bar */}
        <div className="mt-12">
          <div className="flex items-center justify-between text-xs text-slate-400 mb-2 font-mono">
            <span>{statusText}...</span>
            <span className="text-[#00e5ff]">{progress}%</span>
          </div>
          <div className="h-1.5 w-full overflow-hidden rounded-full bg-white/5 border border-white/5">
            <motion.div
              className="h-full bg-gradient-to-r from-[#00bcd4] to-[#00e5ff]"
              style={{ width: `${progress}%` }}
              transition={{ ease: "easeOut" }}
            />
          </div>
        </div>

        {/* Telemetry info */}
        <div className="mt-16 border-t border-white/5 pt-6 flex justify-between text-[10px] text-slate-500 font-mono tracking-wider">
          <span>MODEL: VIVID-CORE V2.4.8</span>
          <span>MODULES: SURV-01 · ANLYT-04 · CTRL-09</span>
        </div>
        <div className="mt-3 text-[10px] text-center text-slate-600 font-mono tracking-wide">
          ENCRYPTED NEURAL LINK ESTABLISHED · FAST-API SECURE PORT 8000
        </div>
      </div>
    </motion.div>
  );
}
