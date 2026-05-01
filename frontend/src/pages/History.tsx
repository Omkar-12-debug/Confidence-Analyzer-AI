import { useState, useEffect } from "react";
import { Card, CardContent } from "@/components/ui/card";
import { Input } from "@/components/ui/input";
import { Badge } from "@/components/ui/badge";
import { Search, Loader2 } from "lucide-react";
import { motion } from "framer-motion";
import { fetchHistory, type HistoryEntry } from "@/lib/api";

// Threshold-based label: 0-39 Low, 40-69 Medium, 70-100 High
const scoreToClass = (s: number | null) =>
  s == null ? "Incomplete" : s >= 70 ? "High" : s >= 40 ? "Medium" : "Low";

const levelColor = (level: string) =>
  level === "High" ? "bg-accent/10 text-accent border-accent/20" :
  level === "Medium" ? "bg-primary/10 text-primary border-primary/20" :
  level === "Incomplete" ? "bg-yellow-500/10 text-yellow-400 border-yellow-500/20" :
  "bg-destructive/10 text-destructive border-destructive/20";

const modeLabel = (mode: string) =>
  mode === "full_multimodal" ? "Multimodal" :
  mode === "voice_only" ? "Voice-only" :
  mode === "facial_only_incomplete" ? "Incomplete" :
  mode === "invalid_recording" ? "Invalid" :
  "—";

const modeBadgeColor = (mode: string) =>
  mode === "full_multimodal" ? "bg-accent/10 text-accent border-accent/20" :
  mode === "voice_only" ? "bg-primary/10 text-primary border-primary/20" :
  "bg-yellow-500/10 text-yellow-400 border-yellow-500/20";

const History = () => {
  const [search, setSearch] = useState("");
  const [filter, setFilter] = useState<string | null>(null);
  const [entries, setEntries] = useState<HistoryEntry[]>([]);
  const [loading, setLoading] = useState(true);

  useEffect(() => {
    fetchHistory().then(h => {
      setEntries(h);
      setLoading(false);
    });
  }, []);

  const filtered = entries.filter((e) => {
    const cls = scoreToClass(e.confidence_score);
    if (filter && cls !== filter) return false;
    if (search && !e.timestamp.includes(search) && !e.id.includes(search)) return false;
    return true;
  });

  return (
    <div className="container max-w-4xl py-10 space-y-6">
      <motion.h1 initial={{ opacity: 0 }} animate={{ opacity: 1 }} className="font-heading text-3xl font-bold">
        History
      </motion.h1>

      <div className="flex flex-col sm:flex-row gap-3">
        <div className="relative flex-1">
          <Search className="absolute left-3 top-1/2 -translate-y-1/2 h-4 w-4 text-muted-foreground" />
          <Input placeholder="Search by date..." className="pl-9" value={search} onChange={(e) => setSearch(e.target.value)} />
        </div>
        <div className="flex gap-2">
          {["High", "Medium", "Low", "Incomplete"].map((l) => (
            <button
              key={l}
              onClick={() => setFilter(filter === l ? null : l)}
              className={`px-3 py-1.5 text-xs font-medium rounded-lg border transition-colors ${filter === l ? "bg-primary text-primary-foreground border-primary" : "bg-card text-muted-foreground border-border hover:bg-muted"}`}
            >
              {l}
            </button>
          ))}
        </div>
      </div>

      {loading ? (
        <div className="flex justify-center py-12">
          <Loader2 className="h-8 w-8 text-primary animate-spin" />
        </div>
      ) : (
        <div className="space-y-3">
          {filtered.map((entry, i) => {
            const cls = scoreToClass(entry.confidence_score);
            const isIncomplete = entry.confidence_score == null;
            const mode = entry.analysis_mode ?? "unknown";
            return (
              <motion.div key={entry.id} initial={{ opacity: 0, y: 10 }} animate={{ opacity: 1, y: 0 }} transition={{ delay: i * 0.05 }}>
                <Card className="shadow-card hover:shadow-elevated transition-shadow">
                  <CardContent className="p-4 flex items-center justify-between flex-wrap gap-3">
                    <div>
                      <p className="font-medium text-sm">
                        {entry.timestamp ? new Date(entry.timestamp).toLocaleString() : entry.id}
                      </p>
                      <p className="text-xs text-muted-foreground">
                        Audio: {entry.audio_status === "audio_insufficient_speech" ? "no speech" : entry.audio_status} · Facial: {entry.facial_status === "facial_no_face_detected" ? "no face" : entry.facial_status}
                      </p>
                    </div>
                    <div className="flex items-center gap-4">
                      <div className="text-center">
                        <p className="font-heading font-bold text-lg">
                          {isIncomplete ? "—" : Math.round(entry.confidence_score!)}
                        </p>
                        <p className="text-[10px] text-muted-foreground uppercase tracking-wide">Score</p>
                      </div>
                      <div className="text-center">
                        <p className="font-heading font-bold text-sm">{entry.audio_score != null ? Math.round(entry.audio_score) : "N/A"}</p>
                        <p className="text-[10px] text-muted-foreground uppercase tracking-wide">Audio</p>
                      </div>
                      <div className="text-center">
                        <p className="font-heading font-bold text-sm">{entry.facial_score != null ? Math.round(entry.facial_score) : "N/A"}</p>
                        <p className="text-[10px] text-muted-foreground uppercase tracking-wide">Facial</p>
                      </div>
                      <Badge variant="outline" className={`${modeBadgeColor(mode)} text-[10px]`}>{modeLabel(mode)}</Badge>
                      <Badge variant="outline" className={levelColor(cls)}>{cls}</Badge>
                    </div>
                  </CardContent>
                </Card>
              </motion.div>
            );
          })}
          {filtered.length === 0 && (
            <p className="text-center text-muted-foreground py-8">
              {entries.length === 0 ? "No analysis history yet. Record a video to get started!" : "No results found."}
            </p>
          )}
        </div>
      )}
    </div>
  );
};

export default History;
