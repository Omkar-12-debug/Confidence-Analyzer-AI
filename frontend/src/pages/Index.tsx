import { Camera, Mic, Square, Play, CheckCircle2, XCircle, MonitorSmartphone, Loader2, AlertCircle } from "lucide-react";
import { Button } from "@/components/ui/button";
import { Card, CardContent } from "@/components/ui/card";
import { useNavigate } from "react-router-dom";
import { useState, useRef, useEffect, useCallback } from "react";
import { motion, AnimatePresence } from "framer-motion";
import { analyzeRecording, checkHealth } from "@/lib/api";
import { toast } from "sonner";

/** Pick the best video+audio MIME type the browser supports */
function pickMimeType(): string {
  const candidates = [
    "video/webm;codecs=vp9,opus",
    "video/webm;codecs=vp8,opus",
    "video/webm;codecs=vp9",
    "video/webm;codecs=vp8",
    "video/webm",
    "video/mp4",
  ];
  for (const t of candidates) {
    if (MediaRecorder.isTypeSupported(t)) {
      console.log(`[Recorder] Selected MIME type: ${t}`);
      return t;
    }
  }
  console.warn("[Recorder] No preferred MIME type supported, using browser default");
  return "";
}

type RecordingState = "idle" | "requesting" | "ready" | "recording" | "uploading" | "analyzing" | "error";

const LONG_RECORDING_WARN = 300; // 5 minutes — soft warning, no auto-stop

const Index = () => {
  const navigate = useNavigate();
  const [state, setState] = useState<RecordingState>("idle");
  const [cameraOk, setCameraOk] = useState(false);
  const [micOk, setMicOk] = useState(false);
  const [backendOk, setBackendOk] = useState(false);
  const [timer, setTimer] = useState(0);
  const [errorMsg, setErrorMsg] = useState("");

  const videoRef = useRef<HTMLVideoElement>(null);
  const streamRef = useRef<MediaStream | null>(null);
  const recorderRef = useRef<MediaRecorder | null>(null);
  const chunksRef = useRef<Blob[]>([]);
  const timerRef = useRef<ReturnType<typeof setInterval> | null>(null);

  // Poll backend health on mount (gives sidecar time to start in desktop mode)
  useEffect(() => {
    let cancelled = false;
    const pollBackend = async () => {
      for (let i = 0; i < 30 && !cancelled; i++) {
        const ok = await checkHealth();
        if (ok) {
          setBackendOk(true);
          return;
        }
        await new Promise((r) => setTimeout(r, 1000));
      }
    };
    pollBackend();
    return () => { cancelled = true; };
  }, []);

  // Attach / detach the media stream to the <video> element whenever it changes
  useEffect(() => {
    const video = videoRef.current;
    if (!video) return;
    const stream = streamRef.current;
    if (stream && (state === "ready" || state === "recording")) {
      video.srcObject = stream;
      video.play().catch((e) => console.warn("[Preview] play() rejected:", e));
    } else {
      video.srcObject = null;
    }
  }, [state]);  // re-run whenever state changes (stream is set before state transitions)

  const stopStream = useCallback(() => {
    streamRef.current?.getTracks().forEach(t => t.stop());
    streamRef.current = null;
    if (videoRef.current) videoRef.current.srcObject = null;
    setCameraOk(false);
    setMicOk(false);
  }, []);

  const requestPermissions = async () => {
    setState("requesting");
    setErrorMsg("");
    try {
      const stream = await navigator.mediaDevices.getUserMedia({ video: true, audio: true });
      streamRef.current = stream;

      const videoTracks = stream.getVideoTracks().length;
      const audioTracks = stream.getAudioTracks().length;
      console.log(`[Stream] Acquired — video tracks: ${videoTracks}, audio tracks: ${audioTracks}`);

      setCameraOk(videoTracks > 0);
      setMicOk(audioTracks > 0);
      setState("ready");  // triggers the useEffect above which sets srcObject
    } catch (err: any) {
      setErrorMsg(err?.message || "Could not access camera/microphone");
      setState("error");
    }
  };

  const startRecording = () => {
    if (!streamRef.current) return;
    chunksRef.current = [];
    setTimer(0);

    const selectedMime = pickMimeType();

    // Verify the stream still has video tracks
    const vtCount = streamRef.current.getVideoTracks().length;
    const atCount = streamRef.current.getAudioTracks().length;
    console.log(`[Recorder] Starting — stream has ${vtCount} video, ${atCount} audio tracks`);
    if (vtCount === 0) {
      console.error("[Recorder] No video tracks on stream — recording will be audio-only!");
    }

    const recorder = new MediaRecorder(
      streamRef.current,
      selectedMime ? { mimeType: selectedMime } : undefined
    );
    console.log(`[Recorder] Actual mimeType: ${recorder.mimeType}`);

    recorder.ondataavailable = (e) => {
      if (e.data.size > 0) chunksRef.current.push(e.data);
    };
    recorder.onstop = handleRecordingDone;
    recorder.start(500); // collect chunks every 500ms
    recorderRef.current = recorder;
    setState("recording");

    // Elapsed-time counter (no auto-stop)
    timerRef.current = setInterval(() => {
      setTimer(prev => prev + 1);
    }, 1000);
  };

  const stopRecording = () => {
    if (timerRef.current) { clearInterval(timerRef.current); timerRef.current = null; }
    if (recorderRef.current && recorderRef.current.state !== "inactive") {
      recorderRef.current.stop();
    }
  };

  const handleRecordingDone = async () => {
    setState("uploading");
    stopStream();

    const actualMime = recorderRef.current?.mimeType || "video/webm";
    const blob = new Blob(chunksRef.current, { type: actualMime });

    // Determine correct file extension from actual MIME
    let ext = ".webm";
    if (actualMime.startsWith("video/mp4")) ext = ".mp4";
    else if (actualMime.startsWith("video/webm")) ext = ".webm";
    const filename = `recording${ext}`;

    console.log(`[Upload] Blob size: ${blob.size} bytes, type: ${actualMime}, filename: ${filename}`);

    if (blob.size === 0) {
      setErrorMsg("Recording produced empty file");
      setState("error");
      return;
    }

    setState("analyzing");
    try {
      const result = await analyzeRecording(blob, filename);
      toast.success("Analysis complete!");
      navigate("/analysis", { state: { result } });
    } catch (err: any) {
      setErrorMsg(err?.message || "Analysis failed");
      setState("error");
    }
  };

  const reset = () => {
    stopStream();
    if (timerRef.current) clearInterval(timerRef.current);
    setState("idle");
    setTimer(0);
    setErrorMsg("");
  };

  // Cleanup on unmount
  useEffect(() => {
    return () => {
      stopStream();
      if (timerRef.current) clearInterval(timerRef.current);
    };
  }, [stopStream]);

  const formatTime = (s: number) => `${Math.floor(s / 60)}:${(s % 60).toString().padStart(2, "0")}`;

  return (
    <div className="container max-w-4xl py-16 space-y-10">
      <motion.div initial={{ opacity: 0, y: 20 }} animate={{ opacity: 1, y: 0 }} transition={{ duration: 0.5 }} className="text-center space-y-4">
        <h1 className="font-heading text-4xl md:text-5xl font-extrabold tracking-tight">
          Confidence & Stress
          <br />
          <motion.span className="text-primary inline-block" initial={{ opacity: 0, y: 10 }} animate={{ opacity: 1, y: 0 }} transition={{ delay: 0.3, duration: 0.5 }}>
            Analysis System
          </motion.span>
        </h1>
        <p className="text-muted-foreground text-lg max-w-xl mx-auto">
          Record a short video answer and get AI-powered confidence analysis using voice and facial behavior.
        </p>
      </motion.div>

      {/* Video preview — always mounted so the ref is stable; visibility controlled by CSS */}
      <div
        className="flex justify-center"
        style={{
          display: (state === "ready" || state === "recording") ? "flex" : "none",
        }}
      >
        <div className="relative rounded-2xl overflow-hidden border-2 border-border shadow-elevated" style={{ width: 480, height: 360 }}>
          <video
            ref={videoRef}
            autoPlay
            muted
            playsInline
            className="w-full h-full object-cover bg-black"
            style={{ transform: "scaleX(-1)" }}   /* mirror for natural selfie feel */
          />
          {state === "recording" && (
            <>
              <div className="absolute top-3 left-3 flex items-center gap-2 bg-destructive/90 text-white px-3 py-1 rounded-full text-sm font-medium">
                <span className="w-2 h-2 bg-white rounded-full animate-pulse" />
                REC {formatTime(timer)}
              </div>
              {timer >= LONG_RECORDING_WARN && (
                <div className="absolute bottom-3 left-3 right-3 text-center bg-yellow-500/90 text-black px-3 py-1 rounded-full text-xs font-medium">
                  Long recording detected. You can stop when ready.
                </div>
              )}
            </>
          )}
        </div>
      </div>

      {/* Loading states */}
      <AnimatePresence>
        {(state === "uploading" || state === "analyzing") && (
          <motion.div initial={{ opacity: 0 }} animate={{ opacity: 1 }} exit={{ opacity: 0 }} className="flex flex-col items-center gap-4 py-10">
            <Loader2 className="h-10 w-10 text-primary animate-spin" />
            <p className="text-lg font-medium">{state === "uploading" ? "Uploading recording..." : "Analyzing with AI models..."}</p>
            <p className="text-sm text-muted-foreground">This may take a few seconds</p>
          </motion.div>
        )}
      </AnimatePresence>

      {/* Error state */}
      <AnimatePresence>
        {state === "error" && (
          <motion.div initial={{ opacity: 0 }} animate={{ opacity: 1 }} exit={{ opacity: 0 }} className="flex flex-col items-center gap-4 py-6">
            <AlertCircle className="h-10 w-10 text-destructive" />
            <p className="text-destructive font-medium">{errorMsg}</p>
            <Button variant="outline" onClick={reset}>Try Again</Button>
          </motion.div>
        )}
      </AnimatePresence>

      {/* Action buttons */}
      <motion.div initial={{ opacity: 0, scale: 0.9 }} animate={{ opacity: 1, scale: 1 }} transition={{ delay: 0.2, duration: 0.5, type: "spring", stiffness: 200 }} className="flex justify-center gap-4">
        {state === "idle" && (
          <div className="relative">
            <motion.div className="absolute -inset-2 rounded-3xl bg-primary/20 blur-lg" animate={{ opacity: [0.4, 0.7, 0.4], scale: [1, 1.05, 1] }} transition={{ duration: 2.5, repeat: Infinity, ease: "easeInOut" }} />
            <motion.div whileHover={{ scale: 1.06 }} whileTap={{ scale: 0.97 }} transition={{ type: "spring", stiffness: 400, damping: 15 }}>
              <Button size="lg" className="relative h-16 px-10 text-lg font-semibold rounded-2xl shadow-elevated gap-3" onClick={requestPermissions}>
                <Camera className="h-5 w-5" /> Start Recording
              </Button>
            </motion.div>
          </div>
        )}

        {state === "requesting" && (
          <Button size="lg" disabled className="h-16 px-10 text-lg rounded-2xl gap-3">
            <Loader2 className="h-5 w-5 animate-spin" /> Requesting permissions...
          </Button>
        )}

        {state === "ready" && (
          <motion.div whileHover={{ scale: 1.06 }} whileTap={{ scale: 0.97 }}>
            <Button size="lg" className="h-16 px-10 text-lg font-semibold rounded-2xl shadow-elevated gap-3" onClick={startRecording}>
              <Play className="h-5 w-5" /> Begin Recording
            </Button>
          </motion.div>
        )}

        {state === "recording" && (
          <motion.div whileHover={{ scale: 1.06 }} whileTap={{ scale: 0.97 }}>
            <Button size="lg" variant="destructive" className="h-16 px-10 text-lg font-semibold rounded-2xl shadow-elevated gap-3" onClick={stopRecording}>
              <Square className="h-5 w-5" /> Stop Recording
            </Button>
          </motion.div>
        )}
      </motion.div>

      {/* System Check Card */}
      {(state === "idle" || state === "ready") && (
        <motion.div initial={{ opacity: 0, y: 20 }} animate={{ opacity: 1, y: 0 }} transition={{ delay: 0.35, duration: 0.5 }}>
          <Card className="shadow-card border-border/50">
            <CardContent className="p-6 space-y-4">
              <div className="flex items-center gap-2 mb-2">
                <MonitorSmartphone className="h-5 w-5 text-primary" />
                <h2 className="font-heading font-semibold text-lg">System Check</h2>
              </div>
              <StatusIndicator label="Camera" active={cameraOk} />
              <StatusIndicator label="Microphone" active={micOk} />
              <StatusIndicator label="Backend Server" active={backendOk} />
              {!backendOk && (
                <p className="text-xs text-destructive">
                  Backend not reachable at 127.0.0.1:5000. Start it with: <code className="bg-muted px-1 rounded">python run_backend.py</code>
                </p>
              )}
              <p className="text-xs text-muted-foreground pt-1">
                Recording continues until you click Stop Recording. Ensure your camera and microphone are connected.
              </p>
            </CardContent>
          </Card>
        </motion.div>
      )}
    </div>
  );
};

const StatusIndicator = ({ label, active }: { label: string; active: boolean }) => (
  <motion.div layout className="flex items-center gap-3 p-3 rounded-lg bg-muted/50">
    <motion.div key={String(active)} initial={{ scale: 0.5, opacity: 0 }} animate={{ scale: 1, opacity: 1 }} transition={{ type: "spring", stiffness: 400, damping: 15 }}>
      {active ? <CheckCircle2 className="h-5 w-5 text-accent" /> : <XCircle className="h-5 w-5 text-destructive" />}
    </motion.div>
    <span className="text-sm font-medium">{label}</span>
    <span className={`ml-auto text-xs font-medium px-2 py-0.5 rounded-full ${active ? "bg-accent/10 text-accent" : "bg-destructive/10 text-destructive"}`}>
      {active ? "Active" : "Not Detected"}
    </span>
  </motion.div>
);

export default Index;
