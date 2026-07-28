import React, { useState } from "react";
import { DollarSign, Clock, ChevronDown, ChevronUp, Sliders, ShieldCheck } from "lucide-react";

export default function TransactionForm({ formState, setFormState, onSubmit, isLoading, disabled }) {
  const [isAdvancedOpen, setIsAdvancedOpen] = useState(false);

  // Derive hour of day from Time (Time in seconds % 86400 // 3600)
  const timeVal = parseFloat(formState.Time) || 0;
  const hourOfDay = Math.floor((timeVal % 86400) / 3600);

  const handleAmountChange = (e) => {
    const val = e.target.value;
    setFormState((prev) => ({ ...prev, Amount: val === "" ? "" : parseFloat(val) }));
  };

  const handleTimeChange = (e) => {
    const val = e.target.value;
    setFormState((prev) => ({ ...prev, Time: val === "" ? "" : parseFloat(val) }));
  };

  const handleHourSliderChange = (e) => {
    const newHour = parseInt(e.target.value, 10);
    // Convert new hour to seconds preserving remaining minutes/seconds offset
    const currentOffset = timeVal % 3600;
    const newTime = newHour * 3600 + currentOffset;
    setFormState((prev) => ({ ...prev, Time: newTime }));
  };

  const handleVFieldChange = (vName, val) => {
    setFormState((prev) => ({ ...prev, [vName]: val === "" ? 0.0 : parseFloat(val) }));
  };

  const handleSubmit = (e) => {
    e.preventDefault();
    onSubmit();
  };

  return (
    <form onSubmit={handleSubmit} className="bg-white p-6 rounded-2xl border border-slate-200 shadow-sm">
      <div className="flex items-center justify-between pb-4 mb-5 border-b border-slate-100">
        <div>
          <h2 className="text-lg font-bold text-slate-900">Transaction Input Parameters</h2>
          <p className="text-xs text-slate-500">Submit transaction features for real-time risk assessment</p>
        </div>
      </div>

      {/* Primary Visible Inputs */}
      <div className="grid grid-cols-1 md:grid-cols-2 gap-5 mb-6">
        {/* Amount Input */}
        <div>
          <label className="block text-xs font-semibold text-slate-700 uppercase tracking-wider mb-2">
            Transaction Amount ($)
          </label>
          <div className="relative rounded-lg shadow-sm">
            <div className="absolute inset-y-0 left-0 pl-3 flex items-center pointer-events-none text-slate-400 font-medium">
              <DollarSign className="w-4 h-4" />
            </div>
            <input
              type="number"
              step="any"
              min="0"
              required
              value={formState.Amount}
              onChange={handleAmountChange}
              placeholder="e.g. 149.62"
              className="w-full pl-9 pr-4 py-2.5 bg-slate-50 border border-slate-300 rounded-lg text-slate-900 font-semibold focus:bg-white focus:ring-2 focus:ring-slate-900 focus:border-slate-900 transition-all text-base"
            />
          </div>
        </div>

        {/* Time / Hour of Day Input */}
        <div>
          <div className="flex items-center justify-between mb-2">
            <label className="text-xs font-semibold text-slate-700 uppercase tracking-wider">
              Time of Day
            </label>
            <span className="text-xs font-mono font-medium text-slate-600 bg-slate-100 px-2 py-0.5 rounded border border-slate-200">
              {String(hourOfDay).padStart(2, '0')}:00 ({timeVal.toLocaleString()} sec)
            </span>
          </div>
          
          <div className="space-y-2">
            <div className="flex items-center gap-3 bg-slate-50 p-2.5 rounded-lg border border-slate-300">
              <Clock className="w-4 h-4 text-slate-400 shrink-0" />
              <input
                type="range"
                min="0"
                max="23"
                value={hourOfDay}
                onChange={handleHourSliderChange}
                className="w-full accent-slate-900 cursor-pointer"
              />
            </div>
            
            <div className="flex items-center justify-between text-[11px] text-slate-400 font-mono px-1">
              <span>00:00 (Midnight)</span>
              <span>12:00 (Noon)</span>
              <span>23:00 (Night)</span>
            </div>
          </div>
        </div>
      </div>

      {/* Advanced Section: Collapsible V1-V28 Grid */}
      <div className="border border-slate-200 rounded-xl overflow-hidden mb-6 bg-slate-50/50">
        <button
          type="button"
          onClick={() => setIsAdvancedOpen(!isAdvancedOpen)}
          className="w-full px-4 py-3 bg-slate-100/80 hover:bg-slate-200/60 flex items-center justify-between text-left transition-colors"
        >
          <div className="flex items-center gap-2">
            <Sliders className="w-4 h-4 text-slate-600" />
            <span className="text-sm font-semibold text-slate-800">
              Advanced: Raw Feature Vector (V1 – V28)
            </span>
            <span className="text-xs text-slate-500 font-normal">
              ({isAdvancedOpen ? "Hide grid" : "Show 28 PCA features"})
            </span>
          </div>
          {isAdvancedOpen ? (
            <ChevronUp className="w-4 h-4 text-slate-600" />
          ) : (
            <ChevronDown className="w-4 h-4 text-slate-600" />
          )}
        </button>

        {isAdvancedOpen && (
          <div className="p-4 bg-white border-t border-slate-200">
            <p className="text-xs text-slate-500 mb-3">
              PCA-anonymized features. Modify values directly to test synthetic edge cases:
            </p>
            <div className="grid grid-cols-2 sm:grid-cols-4 md:grid-cols-7 gap-2.5 max-h-72 overflow-y-auto pr-1">
              {Array.from({ length: 28 }, (_, i) => i + 1).map((num) => {
                const vKey = `V${num}`;
                const val = formState[vKey] ?? 0.0;
                return (
                  <div key={vKey} className="bg-slate-50 p-1.5 rounded border border-slate-200">
                    <label className="block text-[10px] font-mono font-bold text-slate-500 uppercase">
                      {vKey}
                    </label>
                    <input
                      type="number"
                      step="any"
                      value={val}
                      onChange={(e) => handleVFieldChange(vKey, e.target.value)}
                      className="w-full bg-white border border-slate-300 rounded px-1.5 py-0.5 text-xs font-mono text-slate-800 focus:ring-1 focus:ring-slate-900 focus:border-slate-900"
                    />
                  </div>
                );
              })}
            </div>
          </div>
        )}
      </div>

      {/* Submit Button */}
      <button
        type="submit"
        disabled={disabled || isLoading}
        className="w-full py-3.5 px-6 bg-slate-900 hover:bg-slate-800 text-white font-semibold text-sm rounded-xl shadow-md hover:shadow-lg transition-all flex items-center justify-center gap-2.5 disabled:opacity-50 disabled:cursor-not-allowed"
      >
        {isLoading ? (
          <>
            <svg className="animate-spin h-4 w-4 text-white" viewBox="0 0 24 24" fill="none">
              <circle className="opacity-25" cx="12" cy="12" r="10" stroke="currentColor" strokeWidth="4"></circle>
              <path className="opacity-75" fill="currentColor" d="M4 12a8 8 0 018-8V0C5.373 0 0 5.373 0 12h4zm2 5.291A7.962 7.962 0 014 12H0c0 3.042 1.135 5.824 3 7.938l3-2.647z"></path>
            </svg>
            <span>Computing Risk Score & SHAP Values...</span>
          </>
        ) : (
          <>
            <ShieldCheck className="w-4 h-4 text-emerald-400" />
            <span>Submit for Fraud Risk Assessment</span>
          </>
        )}
      </button>
    </form>
  );
}
