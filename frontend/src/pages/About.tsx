import { Card, CardContent, CardHeader, CardTitle } from "@/components/ui/card";
import { Brain, Cpu, AudioLines, Eye, Layers, AlertTriangle } from "lucide-react";
import { motion } from "framer-motion";

const fadeUp = (delay: number) => ({
  initial: { opacity: 0, y: 16 },
  animate: { opacity: 1, y: 0 },
  transition: { delay, duration: 0.4 },
});

const techs = ["TensorFlow", "MediaPipe", "OpenCV", "Librosa", "Scikit-learn"];

const methods = [
  { icon: AudioLines, title: "Audio Feature Extraction", desc: "Extracts pitch, speech rate, pauses, and vocal patterns using Librosa." },
  { icon: Eye, title: "Facial Feature Analysis", desc: "Tracks eye contact, blink rate, and facial expressions via MediaPipe and OpenCV." },
  { icon: Layers, title: "Multimodal Fusion Model", desc: "Combines audio and visual features using a trained ML model for final scoring." },
];

const About = () => (
  <div className="container max-w-4xl py-10 space-y-8">
    <motion.h1 {...fadeUp(0)} className="font-heading text-3xl font-bold">About</motion.h1>

    <motion.div {...fadeUp(0.1)}>
      <Card className="shadow-card">
        <CardHeader>
          <CardTitle className="font-heading flex items-center gap-2"><Brain className="h-5 w-5 text-primary" /> Project Overview</CardTitle>
        </CardHeader>
        <CardContent>
          <p className="text-sm text-muted-foreground leading-relaxed">
            This system uses multimodal AI to analyze a user's confidence and stress levels in real-time. By combining facial behavior analysis and audio feature extraction, it provides an objective assessment suitable for interview preparation, public speaking practice, and self-improvement.
          </p>
        </CardContent>
      </Card>
    </motion.div>

    <motion.div {...fadeUp(0.2)}>
      <Card className="shadow-card">
        <CardHeader>
          <CardTitle className="font-heading flex items-center gap-2"><Cpu className="h-5 w-5 text-primary" /> Technologies Used</CardTitle>
        </CardHeader>
        <CardContent className="flex flex-wrap gap-2">
          {techs.map((t) => (
            <span key={t} className="px-3 py-1.5 text-sm font-medium rounded-lg bg-primary/10 text-primary">{t}</span>
          ))}
        </CardContent>
      </Card>
    </motion.div>

    <motion.div {...fadeUp(0.3)}>
      <Card className="shadow-card">
        <CardHeader>
          <CardTitle className="font-heading">Methodology</CardTitle>
        </CardHeader>
        <CardContent className="space-y-4">
          {methods.map((m) => (
            <div key={m.title} className="flex items-start gap-4 p-3 rounded-lg bg-muted/40">
              <div className="h-10 w-10 rounded-lg bg-accent/10 flex items-center justify-center shrink-0">
                <m.icon className="h-5 w-5 text-accent" />
              </div>
              <div>
                <p className="font-semibold text-sm">{m.title}</p>
                <p className="text-xs text-muted-foreground">{m.desc}</p>
              </div>
            </div>
          ))}
        </CardContent>
      </Card>
    </motion.div>

    <motion.div {...fadeUp(0.4)}>
      <Card className="shadow-card border-destructive/20">
        <CardHeader>
          <CardTitle className="font-heading flex items-center gap-2 text-destructive"><AlertTriangle className="h-5 w-5" /> Disclaimer</CardTitle>
        </CardHeader>
        <CardContent className="space-y-2 text-sm text-muted-foreground">
          <p>• This system is <strong>not</strong> a lie detection tool.</p>
          <p>• This system is <strong>not</strong> a medical diagnostic system.</p>
          <p>• Results are for educational and research purposes only.</p>
        </CardContent>
      </Card>
    </motion.div>
  </div>
);

export default About;
