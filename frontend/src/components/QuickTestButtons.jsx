import React from "react";
import { LEGITIMATE_PRESET, FRAUD_PRESET } from "../presets";
import { CheckCircle2, ShieldAlert, Zap } from "lucide-react";

export default function QuickTestButtons({ onSelectPreset, disabled }) {
  return (
    <div className="bg-slate-100/70 p-4 rounded-xl border border-slate-200/80 mb-6">
      <div className="flex items-center gap-2 mb-3">
        <Zap className="w-4 h-4 text-amber-500" />
        <h3 className="text-sm font-semibold text-slate-800">Evaluator Quick-Test Bench</h3>
        <span className="text-xs text-slate-500 font-normal">
          (Populates all 30 features from real transaction records)
        </span>
      </div>

      <div className="grid grid-cols-1 sm:grid-cols-2 gap-3">
        <button
          type="button"
          disabled={disabled}
          onClick={() => onSelectPreset(LEGITIMATE_PRESET)}
          className="flex items-center justify-center gap-2 px-4 py-2.5 bg-white hover:bg-emerald-50 text-emerald-800 border border-emerald-300 hover:border-emerald-400 font-medium text-sm rounded-lg shadow-sm transition-all disabled:opacity-50 disabled:cursor-not-allowed group"
        >
          <CheckCircle2 className="w-4 h-4 text-emerald-600 group-hover:scale-110 transition-transform" />
          <span>Simulate Legitimate Purchase</span>
        </button>

        <button
          type="button"
          disabled={disabled}
          onClick={() => onSelectPreset(FRAUD_PRESET)}
          className="flex items-center justify-center gap-2 px-4 py-2.5 bg-white hover:bg-rose-50 text-rose-800 border border-rose-300 hover:border-rose-400 font-medium text-sm rounded-lg shadow-sm transition-all disabled:opacity-50 disabled:cursor-not-allowed group"
        >
          <ShieldAlert className="w-4 h-4 text-rose-600 group-hover:scale-110 transition-transform" />
          <span>Simulate Stolen Card / Fraud</span>
        </button>
      </div>
    </div>
  );
}
