import React, { useState } from "react";
import HealthIndicator from "./components/HealthIndicator";
import QuickTestButtons from "./components/QuickTestButtons";
import TransactionForm from "./components/TransactionForm";
import RiskResult from "./components/RiskResult";
import { LEGITIMATE_PRESET } from "./presets";
import { predictTransaction } from "./api";
import { ShieldCheck, AlertCircle } from "lucide-react";

export default function App() {
  const [formState, setFormState] = useState(LEGITIMATE_PRESET);
  const [isBackendHealthy, setIsBackendHealthy] = useState(true);
  const [isLoading, setIsLoading] = useState(false);
  const [error, setError] = useState(null);
  const [result, setResult] = useState(null);

  const handleSelectPreset = (preset) => {
    setFormState(preset);
    setError(null);
  };

  const handleSubmit = async () => {
    setIsLoading(true);
    setError(null);
    try {
      const res = await predictTransaction(formState);
      setResult(res);
    } catch (err) {
      setError(err.message || "An unexpected error occurred during prediction.");
    } finally {
      setIsLoading(false);
    }
  };

  return (
    <div className="min-h-screen bg-slate-50 text-slate-900 pb-16">
      {/* Header Bar */}
      <header className="bg-white border-b border-slate-200 shadow-sm sticky top-0 z-10">
        <div className="max-w-6xl mx-auto px-4 sm:px-6 py-4">
          <div className="flex flex-col md:flex-row md:items-center justify-between gap-4">
            <div className="flex items-center gap-3">
              <div className="p-2.5 bg-slate-900 text-white rounded-xl shadow">
                <ShieldCheck className="w-6 h-6 text-emerald-400" />
              </div>
              <div>
                <h1 className="text-xl font-bold tracking-tight text-slate-900 flex items-center gap-2">
                  FraudGuard<span className="text-emerald-600 font-extrabold">.AI</span>
                  <span className="text-xs bg-slate-100 text-slate-600 border border-slate-200 px-2 py-0.5 rounded font-mono font-medium">
                    Merchant Risk Operations
                  </span>
                </h1>
                <p className="text-xs text-slate-500">
                  Real-time transaction risk scoring with cost-optimal thresholds & SHAP explainability
                </p>
              </div>
            </div>

            <HealthIndicator onHealthChange={setIsBackendHealthy} />
          </div>
        </div>
      </header>

      {/* Main Content Layout */}
      <main className="max-w-6xl mx-auto px-4 sm:px-6 mt-8">
        <div className="grid grid-cols-1 lg:grid-cols-12 gap-8 items-start">
          {/* Left / Main Column: Form & Presets */}
          <div className="lg:col-span-7 space-y-6">
            <QuickTestButtons
              onSelectPreset={handleSelectPreset}
              disabled={!isBackendHealthy || isLoading}
            />

            <TransactionForm
              formState={formState}
              setFormState={setFormState}
              onSubmit={handleSubmit}
              isLoading={isLoading}
              disabled={!isBackendHealthy}
            />
          </div>

          {/* Right Column: Results & Explanations */}
          <div className="lg:col-span-5 space-y-6">
            {error && (
              <div className="p-4 rounded-xl bg-rose-50 border border-rose-200 text-rose-800 text-sm flex items-start gap-3 shadow-sm">
                <AlertCircle className="w-5 h-5 text-rose-600 shrink-0 mt-0.5" />
                <div>
                  <h4 className="font-bold text-rose-900">Prediction Failed</h4>
                  <p className="text-xs text-rose-700 mt-0.5">{error}</p>
                </div>
              </div>
            )}

            {result ? (
              <RiskResult result={result} />
            ) : (
              <div className="bg-white p-8 rounded-2xl border border-slate-200 text-center shadow-sm">
                <div className="w-12 h-12 rounded-full bg-slate-100 mx-auto flex items-center justify-center mb-3">
                  <ShieldCheck className="w-6 h-6 text-slate-400" />
                </div>
                <h3 className="text-base font-bold text-slate-800">Awaiting Submission</h3>
                <p className="text-xs text-slate-500 mt-1 max-w-xs mx-auto">
                  Click one of the Quick-Test presets or submit the transaction form to see real-time risk assessment and SHAP feature explainability.
                </p>
              </div>
            )}
          </div>
        </div>
      </main>
    </div>
  );
}
