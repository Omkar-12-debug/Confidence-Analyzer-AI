import { useEffect, useState, useRef } from "react";
import { motion } from "framer-motion";

const CircularProgress = ({ value, size = 140, strokeWidth = 10, animate = true }: { value: number; size?: number; strokeWidth?: number; animate?: boolean }) => {
  const radius = (size - strokeWidth) / 2;
  const circumference = 2 * Math.PI * radius;
  const [displayValue, setDisplayValue] = useState(0);
  const [currentOffset, setCurrentOffset] = useState(circumference);
  const animationRef = useRef<number>();
  const color = value >= 70 ? "hsl(var(--accent))" : value >= 40 ? "hsl(var(--primary))" : "hsl(var(--destructive))";

  useEffect(() => {
    if (!animate) {
      setDisplayValue(value);
      setCurrentOffset(circumference - (value / 100) * circumference);
      return;
    }

    const duration = 1500;
    const startTime = performance.now();
    const startVal = 0;

    const tick = (now: number) => {
      const elapsed = now - startTime;
      const progress = Math.min(elapsed / duration, 1);
      // ease-out cubic
      const eased = 1 - Math.pow(1 - progress, 3);
      const current = Math.round(startVal + (value - startVal) * eased);
      setDisplayValue(current);
      setCurrentOffset(circumference - (current / 100) * circumference);
      if (progress < 1) {
        animationRef.current = requestAnimationFrame(tick);
      }
    };

    animationRef.current = requestAnimationFrame(tick);
    return () => { if (animationRef.current) cancelAnimationFrame(animationRef.current); };
  }, [value, animate, circumference]);

  return (
    <motion.div
      initial={{ scale: 0.8, opacity: 0 }}
      animate={{ scale: 1, opacity: 1 }}
      transition={{ duration: 0.5, ease: "easeOut" }}
      className="relative inline-flex items-center justify-center"
    >
      <svg width={size} height={size} className="-rotate-90">
        <circle cx={size / 2} cy={size / 2} r={radius} fill="none" stroke="hsl(var(--muted))" strokeWidth={strokeWidth} />
        <circle
          cx={size / 2} cy={size / 2} r={radius} fill="none"
          stroke={color} strokeWidth={strokeWidth}
          strokeDasharray={circumference} strokeDashoffset={currentOffset}
          strokeLinecap="round"
        />
      </svg>
      <div className="absolute flex flex-col items-center">
        <span className="font-heading text-3xl font-bold">{displayValue}</span>
        <span className="text-xs text-muted-foreground">/ 100</span>
      </div>
    </motion.div>
  );
};

export default CircularProgress;
