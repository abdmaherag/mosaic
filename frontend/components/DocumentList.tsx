"use client";

import { useState } from "react";
import { deleteDocument } from "@/lib/api";
import type { DocumentInfo } from "@/lib/types";

interface Props {
  documents: DocumentInfo[];
  onDeleted: (id: string) => void;
  compareMode?: boolean;
  selectedDocIds?: Set<string>;
  onToggleSelect?: (id: string) => void;
}

const EXT_COLORS: Record<string, string> = {
  pdf: "bg-coral text-white",
  docx: "bg-blue-500 text-white",
  doc: "bg-blue-500 text-white",
  html: "bg-orange-400 text-white",
  txt: "bg-gray-400 text-white",
  md: "bg-gray-400 text-white",
  csv: "bg-teal text-white",
  py: "bg-yellow-500 text-white",
  js: "bg-amber text-white",
  ts: "bg-blue-600 text-white",
  json: "bg-gray-500 text-white",
};

function getExt(filename: string) {
  return filename.split(".").pop()?.toLowerCase() ?? "";
}

export default function DocumentList({
  documents,
  onDeleted,
  compareMode = false,
  selectedDocIds = new Set(),
  onToggleSelect,
}: Props) {
  const [deleting, setDeleting] = useState<string | null>(null);
  const [deleteError, setDeleteError] = useState<string | null>(null);

  async function handleDelete(id: string) {
    setDeleting(id);
    setDeleteError(null);
    try {
      await deleteDocument(id);
      onDeleted(id);
    } catch {
      setDeleteError(id);
      setTimeout(() => setDeleteError(null), 3000);
    } finally {
      setDeleting(null);
    }
  }

  if (documents.length === 0) {
    return (
      <p className="text-xs text-charcoal/50 text-center py-4">
        No documents uploaded yet.
      </p>
    );
  }

  return (
    <ul className="space-y-1">
      {documents.map((doc) => {
        const ext = getExt(doc.filename);
        const isSelected = selectedDocIds.has(doc.id);
        return (
          <li
            key={doc.id}
            onClick={compareMode ? () => onToggleSelect?.(doc.id) : undefined}
            className={`group rounded-lg px-2.5 py-2.5 transition-colors border border-transparent ${
              compareMode
                ? "cursor-pointer hover:border-coral/20 hover:bg-coral-light " + (isSelected ? "bg-coral-light border-coral/20" : "")
                : "hover:bg-warm-white"
            }`}
          >
            <div className="flex items-start gap-2">
              {/* File type badge */}
              <span className={`text-[9px] font-bold uppercase rounded px-1 py-0.5 leading-none mt-0.5 shrink-0 ${
                EXT_COLORS[ext] ?? "bg-gray-400 text-white"
              }`}>
                {ext}
              </span>

              <div className="flex-1 min-w-0">
                <div className="flex items-center gap-1.5">
                  <span className="text-xs text-teal">✓</span>
                  <span className="text-[10px] text-teal font-medium">Ready</span>
                </div>
                <p className="text-sm text-charcoal truncate leading-tight" title={doc.filename}>
                  {doc.filename}
                </p>
                <p className="text-[11px] text-charcoal/50">
                  {deleteError === doc.id
                    ? <span className="text-coral">Delete failed — try again</span>
                    : <>{doc.chunk_count} chunk{doc.chunk_count !== 1 ? "s" : ""}</>}
                </p>
              </div>

              {!compareMode && (
                <button
                  onClick={(e) => { e.stopPropagation(); handleDelete(doc.id); }}
                  disabled={deleting === doc.id}
                  className="opacity-0 group-hover:opacity-100 text-charcoal/40 hover:text-coral transition-all disabled:opacity-40 mt-1 focus-visible:opacity-100 focus-visible:outline-none focus-visible:ring-2 focus-visible:ring-coral/50 rounded"
                  title="Delete document"
                >
                  {deleting === doc.id ? (
                    <span className="h-3.5 w-3.5 block rounded-full border-2 border-charcoal/20 border-t-transparent animate-spin" />
                  ) : (
                    <svg className="h-3.5 w-3.5" fill="none" viewBox="0 0 24 24" stroke="currentColor">
                      <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M6 18L18 6M6 6l12 12" />
                    </svg>
                  )}
                </button>
              )}
            </div>
          </li>
        );
      })}
    </ul>
  );
}
