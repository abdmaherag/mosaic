"use client";

import { useState } from "react";
import type { DocumentSource } from "@/lib/types";

interface Props {
  sources: DocumentSource[];
}

function formatPages(pages: number[]): string {
  if (pages.length === 0) return "";
  if (pages.length === 1) return `Page ${pages[0]}`;

  const ranges: string[] = [];
  let start = pages[0];
  let end = pages[0];

  for (let i = 1; i < pages.length; i++) {
    if (pages[i] === end + 1) {
      end = pages[i];
    } else {
      ranges.push(start === end ? `${start}` : `${start}-${end}`);
      start = pages[i];
      end = pages[i];
    }
  }
  ranges.push(start === end ? `${start}` : `${start}-${end}`);

  return `Pages ${ranges.join(", ")}`;
}

export default function SourceCitations({ sources }: Props) {
  const [expanded, setExpanded] = useState<number | null>(null);

  if (sources.length === 0) return null;

  const maxScore = Math.max(...sources.map((s) => s.score), 0.001);
  const pct = (score: number) => Math.round((score / maxScore) * 100);

  return (
    <div className="mt-2 space-y-1.5">
      <p className="text-[11px] font-semibold text-charcoal/40 uppercase tracking-[0.12em] mb-1.5">
        Sources ({sources.length})
      </p>
      {sources.map((s, i) => (
        <div
          key={i}
          className="rounded-lg border border-border bg-card overflow-hidden transition-shadow hover:shadow-sm"
        >
          {/* Document header — clickable to expand */}
          <button
            onClick={() => setExpanded(expanded === i ? null : i)}
            className="w-full flex items-center gap-2.5 px-3 py-2.5 text-left hover:bg-coral-light/50 transition-colors
              focus-visible:outline-none focus-visible:ring-2 focus-visible:ring-coral/50 focus-visible:ring-inset"
          >
            {/* Document icon */}
            <div className="shrink-0 w-7 h-7 rounded-md bg-coral-light flex items-center justify-center">
              <svg className="w-3.5 h-3.5 text-coral" fill="none" viewBox="0 0 24 24" stroke="currentColor" strokeWidth={2}>
                <path strokeLinecap="round" strokeLinejoin="round"
                  d="M19.5 14.25v-2.625a3.375 3.375 0 00-3.375-3.375h-1.5A1.125 1.125 0 0113.5 7.125v-1.5a3.375 3.375 0 00-3.375-3.375H8.25m2.25 0H5.625c-.621 0-1.125.504-1.125 1.125v17.25c0 .621.504 1.125 1.125 1.125h12.75c.621 0 1.125-.504 1.125-1.125V11.25a9 9 0 00-9-9z" />
              </svg>
            </div>

            {/* Document name + pages */}
            <span className="flex-1 min-w-0">
              <span className="text-xs font-medium text-charcoal truncate block">
                {s.document}
              </span>
              {s.pages.length > 0 && (
                <span className="text-[11px] text-charcoal/50">
                  {formatPages(s.pages)}
                </span>
              )}
            </span>

            {/* Score + chunk count badge */}
            <span className="text-[10px] font-mono px-1.5 py-0.5 rounded-full bg-warm-white text-charcoal/50 shrink-0">
              {pct(s.score)}%
            </span>
            <span className="text-[10px] text-charcoal/40 shrink-0">
              {(s.chunks?.length ?? 0)} chunk{(s.chunks?.length ?? 0) !== 1 ? "s" : ""}
            </span>

            {/* Chevron */}
            <svg
              className={`h-3.5 w-3.5 text-charcoal/40 shrink-0 transition-transform ${expanded === i ? "rotate-180" : ""}`}
              fill="none" viewBox="0 0 24 24" stroke="currentColor"
            >
              <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M19 9l-7 7-7-7" />
            </svg>
          </button>

          {/* Expanded: chunk sections sorted by score */}
          {expanded === i && s.chunks?.length > 0 && (
            <div className="border-t border-border divide-y divide-border">
              {s.chunks.map((chunk, j) => (
                <div key={j} className="px-3 py-2.5 flex gap-2.5">
                  {/* Rank indicator */}
                  <div className="shrink-0 w-5 h-5 rounded-full bg-warm-white flex items-center justify-center mt-0.5">
                    <span className="text-[10px] font-mono font-semibold text-charcoal/50">
                      {j + 1}
                    </span>
                  </div>

                  {/* Chunk content */}
                  <div className="min-w-0 flex-1">
                    {(chunk.page != null || chunk.section) && (
                      <p className="text-[11px] text-charcoal/50 mb-0.5">
                        {chunk.page != null && `p.${chunk.page}`}
                        {chunk.page != null && chunk.section && " · "}
                        {chunk.section}
                      </p>
                    )}
                    <p className="text-xs text-charcoal/60 leading-relaxed">
                      &ldquo;{chunk.text}&rdquo;
                    </p>
                  </div>

                  {/* Chunk score */}
                  <span className="text-[10px] font-mono text-charcoal/40 shrink-0 mt-0.5">
                    {pct(chunk.score)}%
                  </span>
                </div>
              ))}
            </div>
          )}
        </div>
      ))}
    </div>
  );
}
