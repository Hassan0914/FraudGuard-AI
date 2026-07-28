import React from "react";
import { ShieldCheck, ShieldAlert, AlertTriangle, Activity, BarChart3 } from "lucide-react";

export default function RiskResult({ result }) {
  if (!result) return null;

  const {
    fraud_prediction,
    fraud_probability_score,
    risk_level,
    threshold_used,
    top_contributing_features = [],
  } = result;

  const probPercentage = (fraud_probability_score * 100).toFixed(2);

  // Semantic color mapping for risk tiers
  const tierStyles = {
    Low: {
      banner: "bg-emerald-50 border-emerald-300 text-emerald-950",
      badge: "bg-emerald-100 text-emerald-800 border-emerald-300",
      bar: "bg-emerald-500",
      icon: ShieldCheck,
      iconColor: "text-emerald-600",
      border: "border-emerald-200",
    },
    Medium: {
      banner: "bg-amber-50 border-amber-300 text-amber-950",
      badge: "bg-amber-100 text-amber-800 border-amber-300",
      bar: "bg-amber-500",
      icon: AlertTriangle,
      iconColor: "text-amber-600",
      border: "border-amber-200",
    },
    High: {
      banner: "bg-rose-50 border-rose-300 text-rose-950",
      badge: "bg-rose-100 text-rose-800 border-rose-300",
      bar: "bg-rose-600",
      icon: ShieldAlert,
      iconColor: "text-rose-600",
      border: "border-rose-200",
    },
  };

  const style = tierStyles[risk_level] || tierStyles.Low;
  const StatusIcon = style.icon;

  // Maximum SHAP magnitude for scaling horizontal bar chart
  const maxAbsShap = Math.max(
    ...top_contributing_features.map((f) => Math.abs(f.shap_value)),
    0.0001
  );

  return (
    <div className={`p-6 rounded-2xl border-2 shadow-sm ${style.banner} transition-all duration-300`}>
      {/* Header Banner */}
      <div className="flex flex-col sm:flex-row sm:items-center justify-between gap-4 pb-5 border-b border-slate-200/60">
        <div className="flex items-center gap-3">
          <div className={`p-2.5 rounded-xl bg-white shadow-sm border ${style.border}`}>
            <StatusIcon className={`w-6 h-6 ${style.iconColor}`} />
          </div>
          <div>
            <div className="flex items-center gap-2">
              <span className={`px-2.5 py-0.5 rounded-full text-xs font-bold uppercase tracking-wider border ${style.badge}`}>
                {risk_level} Risk Tier
              </span>
              <span className="text-xs text-slate-500 font-mono">
                Threshold: {threshold_used.toFixed(2)}
              </span>
            </div>
            <h3 className="text-xl font-bold mt-1 text-slate-900">
              {fraud_prediction === 1 ? "FLAGGED AS FRAUD" : "LEGITIMATE TRANSACTION"}
            </h3>
          </div>
        </div>

        {/* Big Percentage Score Display */}
        <div className="text-left sm:text-right bg-white/80 px-4 py-2 rounded-xl border border-slate-200 shadow-sm">
          <div className="text-[10px] uppercase font-bold text-slate-500 tracking-wider">Fraud Probability</div>
          <div className="text-3xl font-extrabold font-mono text-slate-900">
            {probPercentage}%
          </div>
        </div>
      </div>

      {/* Progress Bar Confidence Meter */}
      <div className="py-4">
        <div className="flex justify-between text-xs font-semibold text-slate-700 mb-1.5">
          <span>Fraud Probability Score</span>
          <span className="font-mono">{fraud_probability_score.toFixed(4)}</span>
        </div>
        <div className="w-full h-3 bg-slate-200/80 rounded-full overflow-hidden p-0.5">
          <div
            className={`h-full rounded-full ${style.bar} transition-all duration-500 shadow-inner`}
            style={{ width: `${Math.max(Math.min(fraud_probability_score * 100, 100), 1.5)}%` }}
          ></div>
        </div>
        <div className="flex justify-between text-[10px] text-slate-500 font-mono mt-1">
          <span>0.0 (Safe)</span>
          <span className="font-bold text-slate-700">Optimal Threshold ({threshold_used})</span>
          <span>1.0 (Certain Fraud)</span>
        </div>
      </div>

      {/* SHAP Feature Contribution Bar Chart */}
      {top_contributing_features.length > 0 && (
        <div className="mt-4 pt-4 border-t border-slate-200/60">
          <div className="flex items-center gap-2 mb-3">
            <BarChart3 className="w-4 h-4 text-slate-700" />
            <h4 className="text-sm font-bold text-slate-900">
              Top 5 SHAP Feature Contributions
            </h4>
          </div>

          <div className="space-y-2.5 bg-white/70 p-3.5 rounded-xl border border-slate-200">
            {top_contributing_features.map((item, idx) => {
              const absVal = Math.abs(item.shap_value);
              const pct = (absVal / maxAbsShap) * 100;
              const isFraudPush = item.shap_value > 0;

              return (
                <div key={idx} className="flex items-center text-xs gap-3">
                  <div className="w-20 font-mono font-bold text-slate-700 shrink-0 truncate">
                    {item.feature}
                  </div>
                  <div className="flex-1 bg-slate-100 h-5 rounded overflow-hidden relative flex items-center">
                    <div
                      className={`h-full ${isFraudPush ? "bg-rose-500" : "bg-sky-500"} transition-all duration-500`}
                      style={{ width: `${Math.max(pct, 2)}%` }}
                    ></div>
                  </div>
                  <div
                    className={`w-16 text-right font-mono font-bold shrink-0 ${
                      isFraudPush ? "text-rose-600" : "text-sky-600"
                    }`}
                  >
                    {isFraudPush ? `+${item.shap_value.toFixed(2)}` : item.shap_value.toFixed(2)}
                  </div>
                </div>
              );
            })}
          </div>

          <div className="flex justify-between items-center text-[10px] text-slate-500 mt-2 px-1">
            <span className="flex items-center gap-1 text-rose-600 font-semibold">
              <span className="w-2 h-2 rounded-full bg-rose-500 inline-block"></span>
              Red (+): Pushes risk HIGHER
            </span>
            <span className="flex items-center gap-1 text-sky-600 font-semibold">
              <span className="w-2 h-2 rounded-full bg-sky-500 inline-block"></span>
              Blue (-): Pushes risk LOWER
            </span>
          </div>
        </div>
      )}
    </div>
  );
}
