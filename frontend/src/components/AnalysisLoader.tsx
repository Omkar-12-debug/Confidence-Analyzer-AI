import { motion, AnimatePresence } from "framer-motion";
import { Brain, Scan, AudioLines } from "lucide-react";
import { useEffect, useState } from "react";

const steps = [
  { icon: Scan, text: "Analyzing Facial Features..." },
  { icon: AudioLines, text: "Processing Audio Signals..." },
  { icon: Brain, text: "Running Multimodal Fusion..." },
];

const AnalysisLoader = ({ onComplete }: { onComplete: () => void }) => {
  const [step, setStep] = useState(0);
  const [progress, setProgress] = useState(0);

  useEffect(() => {
    const interval = setInterval(() => {
      setProgress((p) => {
        if (p >= 100) {
          clearInterval(interval);
          setTimeout(onComplete, 400);
          return 100;
        }
        return p + 1;
      });
    }, 30);
    return () => clearInterval(interval);
  }, [onComplete]);

  useEffect(() => {
    if (progress < 35) setStep(0);
    else if (progress < 70) setStep(1);
    else setStep(2);
  }, [progress]);

  const StepIcon = steps[step].icon;

  return (
    <motion.div
      initial={{ opacity: 0 }}
      animate={{ opacity: 1 }}
      exit={{ opacity: 0, scale: 0.95 }}
      transition={{ duration: 0.4 }}
      className="flex flex-col items-center justify-center py-32 space-y-8"
    >
      {/* Animated icon */}
      <div className="relative">
        <motion.div
          className="absolute inset-0 rounded-full bg-primary/20"
          animate={{ scale: [1, 1.4, 1], opacity: [0.5, 0, 0.5] }}
          transition={{ duration: 2, repeat: Infinity, ease: "easeInOut" }}
        />
        <motion.div
          className="relative h-20 w-20 rounded-full bg-primary/10 flex items-center justify-center"
          animate={{ rotate: [0, 5, -5, 0] }}
          transition={{ duration: 2, repeat: Infinity, ease: "easeInOut" }}
        >
          <AnimatePresence mode="wait">
            <motion.div
              key={step}
              initial={{ opacity: 0, scale: 0.5 }}
              animate={{ opacity: 1, scale: 1 }}
              exit={{ opacity: 0, scale: 0.5 }}
              transition={{ duration: 0.3 }}
            >
              <StepIcon className="h-9 w-9 text-primary" />
            </motion.div>
          </AnimatePresence>
        </motion.div>
      </div>

      {/* Step text */}
      <AnimatePresence mode="wait">
        <motion.p
          key={step}
          initial={{ opacity: 0, y: 8 }}
          animate={{ opacity: 1, y: 0 }}
          exit={{ opacity: 0, y: -8 }}
          transition={{ duration: 0.3 }}
          className="font-heading font-semibold text-lg text-foreground"
        >
          {steps[step].text}
        </motion.p>
      </AnimatePresence>

      {/* Progress bar */}
      <div className="w-64 h-2 bg-muted rounded-full overflow-hidden">
        <motion.div
          className="h-full bg-primary rounded-full"
          style={{ width: `${progress}%` }}
          transition={{ duration: 0.1 }}
        />
      </div>

      <p className="text-sm text-muted-foreground">{progress}%</p>
    </motion.div>
  );
};

export default AnalysisLoader;
