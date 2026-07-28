import React, { useEffect, useState } from "react";
import { checkHealth } from "../api";
import { Server, AlertTriangle } from "lucide-react";

export default function HealthIndicator({ onHealthChange }) {
  const [status, setStatus] = useState("checking");
  const [errorMessage, setErrorMessage] = useState("");

  const verifyHealth = async () => {
    const res = await checkHealth();
    if (res.ok) {
      setStatus("connected");
      setErrorMessage("");
      if (onHealthChange) onHealthChange(true);
    } else {
      setStatus("unreachable");
      setErrorMessage(res.error || "Connection failed");
      if (onHealthChange) onHealthChange(false);
    }
  };

  useEffect(() => {
    verifyHealth();
    const interval = setInterval(verifyHealth, 10000);
    return () => clearInterval(interval);
  }, []);

  return (
    <div className="flex flex-col sm:flex-row sm:items-center justify-between gap-2 bg-white px-4 py-2.5 rounded-lg border border-slate-200 shadow-sm text-sm">
      <div className="flex items-center gap-2">
        <Server className="w-4 h-4 text-slate-500" />
        <span className="font-medium text-slate-700">API Status:</span>
        {status === "checking" && (
          <span className="inline-flex items-center gap-1.5 text-slate-500 font-medium">
            <span className="w-2.5 h-2.5 rounded-full bg-slate-400 animate-pulse"></span>
            Checking connection...
          </span>
        )}
        {status === "connected" && (
          <span className="inline-flex items-center gap-1.5 text-emerald-700 font-semibold bg-emerald-50 px-2.5 py-0.5 rounded-full border border-emerald-200">
            <span className="w-2 h-2 rounded-full bg-emerald-500 animate-ping"></span>
            <span className="w-2 h-2 rounded-full bg-emerald-600 -ml-3.5"></span>
            Backend Connected (127.0.0.1:8000)
          </span>
        )}
        {status === "unreachable" && (
          <span className="inline-flex items-center gap-1.5 text-rose-700 font-semibold bg-rose-50 px-2.5 py-0.5 rounded-full border border-rose-200">
            <span className="w-2 h-2 rounded-full bg-rose-600"></span>
            Backend Unreachable
          </span>
        )}
      </div>

      {status === "unreachable" && (
        <div className="flex items-center gap-1.5 text-xs text-rose-600 font-medium">
          <AlertTriangle className="w-3.5 h-3.5" />
          <span>Run <code className="bg-rose-100 px-1 py-0.5 rounded font-mono">uvicorn backend.main:app --reload --port 8000</code></span>
        </div>
      )}
    </div>
  );
}
