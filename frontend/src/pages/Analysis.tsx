import { Eye, Zap, SmilePlus, AudioLines, Pause, TrendingUp, RefreshCw, Lightbulb, AlertTriangle, ArrowLeft, ShieldAlert, Mic } from "lucide-react";
import { Card, CardContent, CardHeader, CardTitle } from "@/components/ui/card";
import { Button } from "@/components/ui/button";
import CircularProgress from "@/components/CircularProgress";
import { motion, AnimatePresence } from "framer-motion";
import { useState, useMemo } from "react";
import { useNavigate, useLocation } from "react-router-dom";
import type { AnalysisResult } from "@/lib/api";

const fadeUp = (delay: number) => ({
  initial: { opacity: 0, y: 16 },
  animate: { opacity: 1, y: 0 },
  transition: { delay, duration: 0.4 },
});

const Analysis = () => {
  const navigate = useNavigate();
  const location = useLocation();
  const result = (location.state as { result?: AnalysisResult })?.result;

  if (!result) {
    return (
      <div className="container max-w-4xl py-16 text-center space-y-6">
        <AlertTriangle className="h-12 w-12 text-muted-foreground mx-auto" />
        <h2 className="font-heading text-2xl font-bold">No Analysis Data</h2>
        <p className="text-muted-foreground">Record a video first to see your analysis results.</p>
        <Button onClick={() => navigate("/")} className="gap-2">
          <ArrowLeft className="h-4 w-4" /> Go to Recording
        </Button>
      </div>
    );
  }

  const audio = result.audio_module;
  const facial = result.facial_module;
  const fusion = result.fusion_module;
  const analysisMode = result.analysis_mode ?? fusion?.analysis_mode ?? "unknown";

  // Threshold-based label: 0-39 Low, 40-69 Medium, 70-100 High
  const scoreToClass = (s: number) => s >= 70 ? "High" : s >= 40 ? "Medium" : "Low";

  // Whether we have a valid overall score
  const isComplete = analysisMode === "full_multimodal" || analysisMode === "voice_only";

  const confidenceScore = isComplete ? (fusion?.final_confidence_score ?? 0) : null;
  const confidenceClass = confidenceScore != null ? scoreToClass(Math.round(confidenceScore)) : "Incomplete";

  const audioScore = (audio?.status === "success" && audio?.score != null) ? audio.score : null;
  const isFacialValid = facial?.status === "success" && facial?.score != null;
  const facialScore = isFacialValid ? facial.score : null;

  // Derive stress level from confidence class (only meaningful when complete)
  const stressLevel = !isComplete ? "N/A"
    : confidenceClass === "High" ? "Low"
    : confidenceClass === "Low" ? "High"
    : "Medium";

  // Mode label for display
  const modeLabel = analysisMode === "full_multimodal" ? "Full Multimodal Analysis"
    : analysisMode === "voice_only" ? "Voice-only Analysis"
    : analysisMode === "facial_only_incomplete" ? "Incomplete — Facial Only"
    : analysisMode === "invalid_recording" ? "Invalid Recording"
    : "Unknown";

  // Build metrics from real data
  const metrics = useMemo(() => {
    const items = [];
    if (isFacialValid && facial?.features) {
      items.push({ label: "Eye Contact", value: `${facial.features.eye_contact_percentage?.toFixed(1)}%`, icon: Eye });
      items.push({ label: "Blink Rate", value: `${facial.features.blink_rate?.toFixed(2)}/s`, icon: Zap });
      items.push({ label: "Emotion Stability", value: `${((facial.features.emotion_stability ?? 0) * 100).toFixed(0)}%`, icon: SmilePlus });
    }
    if (audio?.features) {
      items.push({ label: "Speech Rate", value: `${audio.features.speech_rate?.toFixed(1)} syl/s`, icon: AudioLines });
      items.push({ label: "Pause Ratio", value: `${audio.features.pause_ratio?.toFixed(3)}`, icon: Pause });
      items.push({ label: "Pitch Std", value: `${audio.features.pitch_std?.toFixed(1)} Hz`, icon: TrendingUp });
    }
    // Fallback if no features available
    if (items.length === 0) {
      items.push({ label: "Audio Score", value: audioScore != null ? `${audioScore}` : "N/A", icon: AudioLines });
      items.push({ label: "Facial Score", value: facialScore != null ? `${facialScore}` : "N/A", icon: Eye });
    }
    return items;
  }, [audio, facial, audioScore, facialScore, isFacialValid]);

  // Generate insights from real data
  const insights = useMemo(() => {
    const ins: string[] = [];

    // Mode-specific messages first
    if (analysisMode === "facial_only_incomplete") {
      ins.push("🔇 Speech is required for confidence evaluation. Facial cues alone are not enough.");
    }
    if (analysisMode === "invalid_recording") {
      ins.push("⚠️ No sufficient speech or face detected. Please record again.");
    }
    if (analysisMode === "voice_only") {
      ins.push("📷 No face detected. Result is based on voice only.");
    }

    if (isFacialValid && facial?.features) {
      const ec = facial.features.eye_contact_percentage ?? 0;
      if (ec > 70) ins.push("Good eye contact maintained throughout the session");
      else if (ec > 40) ins.push("Moderate eye contact — try to look at the camera more");
      else ins.push("Low eye contact detected — practice facing the camera directly");

      const es = facial.features.emotion_stability ?? 0;
      if (es > 0.8) ins.push("Facial expressions show consistent engagement");
      else ins.push("Some facial expression instability detected");
    } else if (facial?.status === "facial_no_face_detected" && analysisMode !== "voice_only" && analysisMode !== "invalid_recording") {
      ins.push(`📷 ${facial.message ?? "No face detected. Please keep your face visible and try again."}`);
    }

    if (audio?.features) {
      const pr = audio.features.pause_ratio ?? 0;
      if (pr < 0.3) ins.push("Natural speech flow with appropriate pauses");
      else ins.push("Higher than average pause ratio — try to reduce hesitations");

      if (audio.model_prediction === "confident") ins.push("Voice analysis indicates confident speech patterns");
      else if (audio.model_prediction === "nervous") ins.push("Voice analysis suggests some nervousness — practice speaking slowly");
      else ins.push("Neutral vocal tone detected");
    }
    if (audio?.status === "audio_insufficient_speech" && analysisMode !== "facial_only_incomplete" && analysisMode !== "invalid_recording") {
      ins.push(`🔇 ${audio.message ?? "No sufficient speech detected. Please speak clearly and try again."}`);
    }
    if (result.warnings?.length) {
      result.warnings.forEach(w => ins.push(`⚠️ ${w}`));
    }
    if (ins.length === 0) ins.push("Analysis complete — review your scores above");
    return ins;
  }, [audio, facial, result.warnings, analysisMode, isFacialValid]);

  // Helper: mode badge color
  const modeBadgeColor =
    analysisMode === "full_multimodal" ? "bg-accent/10 text-accent border border-accent/20" :
    analysisMode === "voice_only" ? "bg-primary/10 text-primary border border-primary/20" :
    "bg-yellow-500/10 text-yellow-400 border border-yellow-500/20";

  return (
    <div className="container max-w-6xl py-10 space-y-8">
      <motion.div key="results" initial={{ opacity: 0 }} animate={{ opacity: 1 }} transition={{ duration: 0.5 }} className="space-y-8">
        <motion.div {...fadeUp(0)} className="flex items-center gap-3 flex-wrap">
          <h1 className="font-heading text-3xl font-bold">Analysis Results</h1>
          <span className={`text-xs font-medium px-3 py-1 rounded-full ${modeBadgeColor}`}>
            {modeLabel}
          </span>
        </motion.div>

        <div className="grid md:grid-cols-3 gap-6">
          {/* Main score card */}
          <motion.div {...fadeUp(0.1)}>
            <motion.div whileHover={{ y: -4, boxShadow: "0 8px 30px -8px rgba(0,0,0,0.1)" }} transition={{ duration: 0.2 }}>
              <Card className="shadow-card h-full">
                <CardContent className="p-6 flex flex-col items-center gap-4">
                  {isComplete ? (
                    <>
                      <CircularProgress value={Math.round(confidenceScore!)} />
                      <div className="text-center space-y-1">
                        <p className="font-heading font-semibold text-lg">
                          Confidence: <span className="text-primary">{confidenceClass}</span>
                        </p>
                        <p className="text-sm text-muted-foreground">
                          Stress Level: <span className="text-accent font-medium">{stressLevel}</span>
                        </p>
                      </div>
                    </>
                  ) : (
                    <div className="flex flex-col items-center gap-3 py-4">
                      <ShieldAlert className="h-12 w-12 text-yellow-400" />
                      <p className="font-heading font-semibold text-lg text-yellow-400">Incomplete</p>
                      <p className="text-sm text-muted-foreground text-center max-w-[220px]">
                        {fusion?.message ?? "Insufficient data for confidence evaluation."}
                      </p>
                    </div>
                  )}

                  {/* Module scores */}
                  <div className="w-full space-y-2 pt-2 border-t">
                    <div className="flex justify-between text-sm">
                      <span className="text-muted-foreground">Voice Score</span>
                      <span className="font-semibold">{audioScore != null ? audioScore : "N/A"}</span>
                    </div>
                    <div className="flex justify-between text-sm">
                      <span className="text-muted-foreground">Facial Score</span>
                      <span className="font-semibold">
                        {facialScore != null ? facialScore : "N/A"}
                        {/* In facial-only incomplete mode, show the value as a behavioral indicator */}
                        {analysisMode === "facial_only_incomplete" && facial?.score != null && (
                          <span className="text-xs text-muted-foreground ml-1">(behavioral only)</span>
                        )}
                      </span>
                    </div>
                    {audioScore != null && audio?.model_prediction && (
                      <div className="flex justify-between text-sm">
                        <span className="text-muted-foreground">Voice Class</span>
                        <span className="font-semibold capitalize">{audio.model_prediction}</span>
                      </div>
                    )}
                    <div className="flex justify-between text-sm">
                      <span className="text-muted-foreground">Facial Class</span>
                      <span className="font-semibold">
                        {isFacialValid && facial?.class ? facial.class : "N/A"}
                      </span>
                    </div>
                  </div>

                  {/* Status badges */}
                  <div className="flex gap-2 flex-wrap">
                    <AnalysisModeBadge mode={analysisMode} />
                    <StatusBadge label="Audio" status={audio?.status} />
                    <StatusBadge label="Facial" status={facial?.status} />
                  </div>
                </CardContent>
              </Card>
            </motion.div>
          </motion.div>

          {/* Metrics card */}
          <motion.div {...fadeUp(0.2)} className="md:col-span-2">
            <motion.div whileHover={{ y: -4, boxShadow: "0 8px 30px -8px rgba(0,0,0,0.1)" }} transition={{ duration: 0.2 }}>
              <Card className="shadow-card h-full">
                <CardHeader className="pb-3">
                  <CardTitle className="font-heading text-lg">
                    {analysisMode === "facial_only_incomplete" ? "Facial Behavioral Indicators (Not Confidence)" : "Feature Metrics"}
                  </CardTitle>
                </CardHeader>
                <CardContent className="grid grid-cols-2 sm:grid-cols-3 gap-4">
                  {metrics.map((m, i) => (
                    <motion.div
                      key={m.label}
                      initial={{ opacity: 0, y: 10 }}
                      animate={{ opacity: 1, y: 0 }}
                      transition={{ delay: 0.3 + i * 0.06 }}
                      whileHover={{ scale: 1.03 }}
                      className="flex items-center gap-3 p-3 rounded-lg bg-muted/40 cursor-default"
                    >
                      <motion.div
                        className="h-9 w-9 rounded-lg bg-primary/10 flex items-center justify-center"
                        whileHover={{ rotate: 8, scale: 1.1 }}
                        transition={{ type: "spring", stiffness: 300 }}
                      >
                        <m.icon className="h-4 w-4 text-primary" />
                      </motion.div>
                      <div>
                        <p className="font-semibold text-sm">{m.value}</p>
                        <p className="text-xs text-muted-foreground">{m.label}</p>
                      </div>
                    </motion.div>
                  ))}
                </CardContent>
              </Card>
            </motion.div>
          </motion.div>
        </div>

        {/* Insights */}
        <motion.div {...fadeUp(0.4)}>
          <motion.div whileHover={{ y: -4, boxShadow: "0 8px 30px -8px rgba(0,0,0,0.1)" }} transition={{ duration: 0.2 }}>
            <Card className="shadow-card">
              <CardHeader className="pb-2">
                <CardTitle className="font-heading text-lg flex items-center gap-2">
                  <motion.div whileHover={{ rotate: 15 }} transition={{ type: "spring", stiffness: 300 }}>
                    <Lightbulb className="h-5 w-5 text-accent" />
                  </motion.div>
                  Insights
                </CardTitle>
              </CardHeader>
              <CardContent className="space-y-2">
                {insights.map((text, i) => (
                  <motion.p
                    key={i}
                    initial={{ opacity: 0, x: -10 }}
                    animate={{ opacity: 1, x: 0 }}
                    transition={{ delay: 0.5 + i * 0.08 }}
                    className="text-sm text-muted-foreground flex items-start gap-2"
                  >
                    <span className="text-accent mt-0.5">•</span> {text}
                  </motion.p>
                ))}
              </CardContent>
            </Card>
          </motion.div>
        </motion.div>

        {/* Warnings */}
        {result.warnings && result.warnings.length > 0 && (
          <motion.div {...fadeUp(0.5)}>
            <Card className="shadow-card border-yellow-500/30 bg-yellow-950/10">
              <CardContent className="p-4 space-y-1">
                {result.warnings.map((w, i) => (
                  <p key={i} className="text-sm text-yellow-400 flex items-start gap-2">
                    <AlertTriangle className="h-4 w-4 mt-0.5 shrink-0" /> {w}
                  </p>
                ))}
              </CardContent>
            </Card>
          </motion.div>
        )}

        {/* Actions */}
        <motion.div {...fadeUp(0.6)} className="flex gap-3">
          <motion.div whileHover={{ scale: 1.05 }} whileTap={{ scale: 0.95 }}>
            <Button className="gap-2" onClick={() => navigate("/")}>
              <RefreshCw className="h-4 w-4" /> {isComplete ? "Analyze Again" : "Record Again"}
            </Button>
          </motion.div>
          <motion.div whileHover={{ scale: 1.05 }} whileTap={{ scale: 0.95 }}>
            <Button variant="outline" className="gap-2" onClick={() => navigate("/history")}>
              View History
            </Button>
          </motion.div>
        </motion.div>
      </motion.div>
    </div>
  );
};

/** Badge showing the analysis mode (replaces old "Fusion: success" badge) */
const AnalysisModeBadge = ({ mode }: { mode: string }) => {
  const config: Record<string, { label: string; color: string }> = {
    full_multimodal: { label: "Fusion: success", color: "bg-accent/10 text-accent" },
    voice_only: { label: "Voice-only", color: "bg-primary/10 text-primary" },
    facial_only_incomplete: { label: "Incomplete", color: "bg-yellow-500/10 text-yellow-400" },
    invalid_recording: { label: "Incomplete", color: "bg-destructive/10 text-destructive" },
  };
  const c = config[mode] ?? { label: mode, color: "bg-muted text-muted-foreground" };
  return <span className={`text-[10px] font-medium px-2 py-0.5 rounded-full ${c.color}`}>{c.label}</span>;
};

const StatusBadge = ({ label, status }: { label: string; status?: string }) => {
  const color =
    status === "success" ? "bg-accent/10 text-accent" :
    status === "error" ? "bg-destructive/10 text-destructive" :
    status === "audio_insufficient_speech" ? "bg-yellow-500/10 text-yellow-400" :
    status === "facial_no_face_detected" ? "bg-yellow-500/10 text-yellow-400" :
    status === "voice_only" ? "bg-primary/10 text-primary" :
    status === "incomplete" ? "bg-yellow-500/10 text-yellow-400" :
    "bg-muted text-muted-foreground";
  const displayStatus =
    status === "audio_insufficient_speech" ? "no speech" :
    status === "facial_no_face_detected" ? "no face" :
    (status ?? "N/A");
  return <span className={`text-[10px] font-medium px-2 py-0.5 rounded-full ${color}`}>{label}: {displayStatus}</span>;
};

export default Analysis;
