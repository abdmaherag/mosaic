"use client";

import type { DocumentInfo } from "@/lib/types";

interface DocumentStatsProps {
  documents: DocumentInfo[];
}

const FILE_TYPE_COLORS: Record<string, string> = {
  pdf: "bg-coral text-white",
  html: "bg-teal text-white",
  docx: "bg-amber text-deep",
  txt: "bg-charcoal/10 text-charcoal",
  md: "bg-deep text-white",
  csv: "bg-teal/70 text-white",
};

function getExtension(filename: string): string {
  const ext = filename.split(".").pop()?.toLowerCase() ?? "";
  return ext;
}

function formatTokens(chunks: number): string {
  const tokens = chunks * 100;
  if (tokens >= 1000) return `${(tokens / 1000).toFixed(1)}k`;
  return String(tokens);
}

export default function DocumentStats({ documents }: DocumentStatsProps) {
  const totalChunks = documents.reduce((s, d) => s + (d.chunk_count || 0), 0);
  const maxChunks = Math.max(...documents.map((d) => d.chunk_count || 0), 1);

  // Group by file type
  const byType: Record<string, { count: number; chunks: number }> = {};
  for (const doc of documents) {
    const ext = getExtension(doc.filename) || "other";
    if (!byType[ext]) byType[ext] = { count: 0, chunks: 0 };
    byType[ext].count++;
    byType[ext].chunks += doc.chunk_count || 0;
  }
  const typeEntries = Object.entries(byType).sort((a, b) => b[1].chunks - a[1].chunks);

  if (documents.length === 0) {
    return (
      <div className="flex-1 flex items-center justify-center">
        <div className="text-center">
          <div className="text-4xl mb-3 text-charcoal/20">
            <svg className="h-12 w-12 mx-auto" fill="none" viewBox="0 0 24 24" stroke="currentColor" strokeWidth={1}>
              <path strokeLinecap="round" strokeLinejoin="round" d="M3 13.125C3 12.504 3.504 12 4.125 12h2.25c.621 0 1.125.504 1.125 1.125v6.75C7.5 20.496 6.996 21 6.375 21h-2.25A1.125 1.125 0 013 19.875v-6.75zM9.75 8.625c0-.621.504-1.125 1.125-1.125h2.25c.621 0 1.125.504 1.125 1.125v11.25c0 .621-.504 1.125-1.125 1.125h-2.25a1.125 1.125 0 01-1.125-1.125V8.625zM16.5 4.125c0-.621.504-1.125 1.125-1.125h2.25C20.496 3 21 3.504 21 4.125v15.75c0 .621-.504 1.125-1.125 1.125h-2.25a1.125 1.125 0 01-1.125-1.125V4.125z" />
            </svg>
          </div>
          <p className="text-sm text-charcoal/40">Upload documents to see stats</p>
        </div>
      </div>
    );
  }

  return (
    <div className="flex-1 overflow-y-auto p-6">
      <div className="max-w-2xl mx-auto space-y-6">
        {/* Summary cards */}
        <div className="grid grid-cols-3 gap-4">
          <div className="card p-4 text-center">
            <p className="text-2xl font-mono font-semibold text-deep">{documents.length}</p>
            <p className="text-xs text-charcoal/50 mt-1">Documents</p>
          </div>
          <div className="card p-4 text-center">
            <p className="text-2xl font-mono font-semibold text-deep">{totalChunks}</p>
            <p className="text-xs text-charcoal/50 mt-1">Total Chunks</p>
          </div>
          <div className="card p-4 text-center">
            <p className="text-2xl font-mono font-semibold text-deep">{formatTokens(totalChunks)}</p>
            <p className="text-xs text-charcoal/50 mt-1">Est. Tokens</p>
          </div>
        </div>

        {/* Chunks per document bar chart */}
        <div className="card p-5">
          <h3 className="font-heading text-deep text-base mb-4">Chunks per Document</h3>
          <div className="space-y-3">
            {documents
              .sort((a, b) => (b.chunk_count || 0) - (a.chunk_count || 0))
              .map((doc) => {
                const pct = ((doc.chunk_count || 0) / maxChunks) * 100;
                const ext = getExtension(doc.filename);
                return (
                  <div key={doc.id}>
                    <div className="flex items-center justify-between mb-1">
                      <div className="flex items-center gap-2 min-w-0">
                        <span className={`text-[10px] font-mono font-semibold px-1.5 py-0.5 rounded ${FILE_TYPE_COLORS[ext] || "bg-charcoal/10 text-charcoal"}`}>
                          .{ext}
                        </span>
                        <span className="text-sm text-charcoal truncate">{doc.filename}</span>
                      </div>
                      <span className="text-xs font-mono text-charcoal/50 ml-2 shrink-0">{doc.chunk_count}</span>
                    </div>
                    <div className="h-2 bg-warm-white rounded-full overflow-hidden">
                      <div
                        className="h-full bg-coral rounded-full transition-all duration-500"
                        style={{ width: `${pct}%` }}
                      />
                    </div>
                  </div>
                );
              })}
          </div>
        </div>

        {/* File type breakdown */}
        <div className="card p-5">
          <h3 className="font-heading text-deep text-base mb-4">File Types</h3>
          <div className="space-y-2">
            {typeEntries.map(([ext, data]) => {
              const pct = totalChunks > 0 ? ((data.chunks / totalChunks) * 100).toFixed(1) : "0";
              return (
                <div key={ext} className="flex items-center gap-3">
                  <span className={`text-[10px] font-mono font-semibold px-1.5 py-0.5 rounded w-12 text-center ${FILE_TYPE_COLORS[ext] || "bg-charcoal/10 text-charcoal"}`}>
                    .{ext}
                  </span>
                  <div className="flex-1 h-2 bg-warm-white rounded-full overflow-hidden">
                    <div
                      className="h-full bg-teal rounded-full transition-all duration-500"
                      style={{ width: `${pct}%` }}
                    />
                  </div>
                  <span className="text-xs text-charcoal/50 w-20 text-right">
                    {data.count} file{data.count !== 1 ? "s" : ""} &middot; {pct}%
                  </span>
                </div>
              );
            })}
          </div>
        </div>

        {/* Document details table */}
        <div className="card overflow-hidden">
          <table className="w-full text-sm">
            <thead>
              <tr className="border-b border-border bg-warm-white/50">
                <th className="text-left px-4 py-2.5 text-xs font-semibold text-charcoal/50 uppercase tracking-wider">File</th>
                <th className="text-right px-4 py-2.5 text-xs font-semibold text-charcoal/50 uppercase tracking-wider">Type</th>
                <th className="text-right px-4 py-2.5 text-xs font-semibold text-charcoal/50 uppercase tracking-wider">Chunks</th>
                <th className="text-right px-4 py-2.5 text-xs font-semibold text-charcoal/50 uppercase tracking-wider">Est. Tokens</th>
              </tr>
            </thead>
            <tbody>
              {documents.map((doc, i) => {
                const ext = getExtension(doc.filename);
                return (
                  <tr key={doc.id} className={i % 2 === 0 ? "" : "bg-warm-white/30"}>
                    <td className="px-4 py-2 text-charcoal truncate max-w-[200px]">{doc.filename}</td>
                    <td className="px-4 py-2 text-right">
                      <span className={`text-[10px] font-mono font-semibold px-1.5 py-0.5 rounded ${FILE_TYPE_COLORS[ext] || "bg-charcoal/10 text-charcoal"}`}>
                        .{ext}
                      </span>
                    </td>
                    <td className="px-4 py-2 text-right font-mono text-charcoal/70">{doc.chunk_count}</td>
                    <td className="px-4 py-2 text-right font-mono text-charcoal/70">{formatTokens(doc.chunk_count || 0)}</td>
                  </tr>
                );
              })}
            </tbody>
          </table>
        </div>
      </div>
    </div>
  );
}
