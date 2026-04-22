"use client";

import { useCallback, useState } from "react";
import { uploadDocument } from "@/lib/api";
import type { ProcessingEvent } from "@/lib/types";

interface Props {
  onUploadComplete: () => void;
  onToast?: (message: string) => void;
}

const ACCEPTED = [".pdf", ".docx", ".html", ".txt", ".md", ".csv", ".py", ".js", ".ts", ".json"];

const STEP_LABELS: Record<string, string> = {
  parsing: "Parsing",
  chunking: "Chunking",
  embedding: "Embedding",
  storing: "Storing",
  complete: "Complete",
};

export default function UploadZone({ onUploadComplete, onToast }: Props) {
  const [dragging, setDragging] = useState(false);
  const [uploading, setUploading] = useState(false);
  const [events, setEvents] = useState<ProcessingEvent[]>([]);
  const [error, setError] = useState<string | null>(null);

  const handleFile = useCallback(
    async (file: File) => {
      setUploading(true);
      setEvents([]);
      setError(null);

      try {
        await uploadDocument(file, (event) => {
          setEvents((prev) => {
            const idx = prev.findIndex((e) => e.step === event.step);
            if (idx !== -1) {
              const next = [...prev];
              next[idx] = event;
              return next;
            }
            return [...prev, event];
          });
        });
        onUploadComplete();
      } catch (e) {
        const msg = e instanceof Error ? e.message : "Upload failed";
        if (msg.toLowerCase().includes("already uploaded") && onToast) {
          onToast(msg);
        } else {
          setError(msg);
        }
      } finally {
        setTimeout(() => {
          setUploading(false);
          setEvents([]);
        }, 3000);
      }
    },
    [onUploadComplete]
  );

  const onDrop = useCallback(
    (e: React.DragEvent) => {
      e.preventDefault();
      setDragging(false);
      const file = e.dataTransfer.files[0];
      if (file) handleFile(file);
    },
    [handleFile]
  );

  const onInputChange = useCallback(
    (e: React.ChangeEvent<HTMLInputElement>) => {
      const file = e.target.files?.[0];
      if (file) handleFile(file);
      e.target.value = "";
    },
    [handleFile]
  );

  const lastEvent = events[events.length - 1];
  const progress = lastEvent?.progress ?? 0;

  return (
    <div className="space-y-2">
      <label
        onDragOver={(e) => { e.preventDefault(); setDragging(true); }}
        onDragLeave={() => setDragging(false)}
        onDrop={onDrop}
        className={`
          relative flex flex-col items-center justify-center gap-0.5 cursor-pointer
          rounded-lg border border-dashed p-3 text-center transition-colors
          ${dragging ? "drop-active bg-coral-light" : "border-border hover:border-coral/40"}
          ${uploading ? "pointer-events-none opacity-60" : ""}
        `}
      >
        <input
          type="file"
          className="sr-only"
          accept={ACCEPTED.join(",")}
          onChange={onInputChange}
          disabled={uploading}
        />
        <svg className="h-5 w-5 text-charcoal/30 mb-0.5" fill="none" viewBox="0 0 24 24" stroke="currentColor">
          <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={1.5}
            d="M3 16.5v2.25A2.25 2.25 0 005.25 21h13.5A2.25 2.25 0 0021 18.75V16.5m-13.5-9L12 3m0 0l4.5 4.5M12 3v13.5" />
        </svg>
        <span className="text-xs text-charcoal/50">
          {uploading ? "Uploading…" : "Drag & drop or click to upload"}
        </span>
        <span className="text-[10px] text-charcoal/30">PDF, DOCX, TXT, CSV, MD</span>
      </label>

      {uploading && (
        <div className="space-y-1.5 card p-3">
          <div className="h-1 rounded-full bg-border overflow-hidden">
            <div
              className="h-full bg-coral transition-all duration-300 rounded-full"
              style={{ width: `${Math.round(progress * 100)}%` }}
            />
          </div>
          <div className="space-y-0.5">
            {events.map((ev) => (
              <div key={ev.step} className="flex items-center gap-2 text-xs">
                {ev.status === "done" || ev.status === "complete" ? (
                  <span className="text-teal">✓</span>
                ) : ev.status === "error" ? (
                  <span className="text-coral">✗</span>
                ) : (
                  <span className="h-3 w-3 rounded-full border-2 border-coral border-t-transparent animate-spin inline-block" />
                )}
                <span className={ev.status === "error" ? "text-coral" : "text-charcoal/70"}>
                  {STEP_LABELS[ev.step] ?? ev.step}
                  {ev.detail ? ` — ${ev.detail}` : ""}
                </span>
              </div>
            ))}
          </div>
        </div>
      )}

      {error && (
        <p className="text-xs text-coral px-1">{error}</p>
      )}
    </div>
  );
}
