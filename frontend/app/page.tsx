"use client";

import { useCallback, useEffect, useRef, useState } from "react";
import ReactMarkdown from "react-markdown";
import {
  clearSession,
  deleteDocument,
  listDocuments,
  streamQuery,
  uploadDocument,
} from "@/lib/api";
import type { DocumentInfo, DocumentSource, ProcessingEvent } from "@/lib/types";

interface Exchange {
  id: number;
  prompt: string;
  subQueries: string[];
  answer: string;
  sources: DocumentSource[];
  streaming: boolean;
  thinkingExpanded: boolean;
  completedSteps: number; // 0..4
  timestamp: string;
  startedAt: number;
  elapsedMs: number | null;
}

const THINKING_STEPS = [
  "Decomposing query into sub-questions",
  "Retrieving relevant passages from active documents",
  "Ranking and filtering retrieved content",
  "Synthesizing comprehensive response",
];

function categoryFor(doc: DocumentInfo): string {
  const ct = doc.content_type?.toLowerCase() ?? "";
  if (ct.includes("pdf")) return "PDFs";
  if (ct.includes("word") || ct.includes("docx")) return "Documents";
  if (ct.includes("html")) return "Web";
  if (ct.includes("csv")) return "Data";
  if (ct.includes("markdown") || ct.includes("md")) return "Markdown";
  if (ct.includes("text") || ct.includes("plain")) return "Text";
  return "Other";
}

function deriveSubQueries(prompt: string): string[] {
  const lower = prompt.toLowerCase();
  if (/compar|diff|versus|\bvs\b/.test(lower)) {
    return [
      "What are the fundamental differences between the concepts mentioned?",
      "How do the approaches compare in practice?",
      "What are the trade-offs between them?",
      "In which use cases does each perform better?",
    ];
  }
  if (/summar|finding|overview/.test(lower)) {
    return [
      "What are the primary topics in the documents?",
      "What methodologies are used?",
      "What are the key results?",
      "What limitations are mentioned?",
    ];
  }
  return [
    "What specific information is requested in the query?",
    "Which documents contain relevant information?",
    "What are the key points from the retrieved passages?",
    "How can the information be synthesized?",
  ];
}

export default function Home() {
  const [documents, setDocuments] = useState<DocumentInfo[]>([]);
  const [selected, setSelected] = useState<Set<string>>(new Set());
  const [search, setSearch] = useState("");
  const [input, setInput] = useState("");
  const [exchanges, setExchanges] = useState<Exchange[]>([]);
  const [sessionId, setSessionId] = useState<string | null>(null);
  const [streaming, setStreaming] = useState(false);
  const [sidebarOpen, setSidebarOpen] = useState(true);
  // Track viewport for drawer-vs-inline sidebar behavior. md = 768px in Tailwind v4.
  const [isMobile, setIsMobile] = useState(false);
  const [toast, setToast] = useState<{ msg: string; type: "success" | "warning" } | null>(null);
  const [uploadState, setUploadState] = useState<{
    filename: string;
    steps: { step: string; status: string; detail: string | null }[];
    completed: boolean;
    error: string | null;
  } | null>(null);

  const chatRef = useRef<HTMLDivElement>(null);
  const fileInputRef = useRef<HTMLInputElement>(null);
  const idCounter = useRef(0);

  const initialLoad = useRef(true);
  const knownIds = useRef<Set<string>>(new Set());

  const fetchDocs = useCallback(async () => {
    try {
      const docs = await listDocuments();
      setDocuments(docs);
      const ids = new Set(docs.map((d) => d.id));
      setSelected((prev) => {
        if (initialLoad.current) {
          initialLoad.current = false;
          knownIds.current = ids;
          return new Set(ids);
        }
        const next = new Set<string>();
        for (const id of prev) if (ids.has(id)) next.add(id);
        for (const id of ids) if (!knownIds.current.has(id)) next.add(id);
        knownIds.current = ids;
        return next;
      });
    } catch {
      // backend unreachable
    }
  }, []);

  useEffect(() => {
    fetchDocs();
  }, [fetchDocs]);

  // Sync sidebar default state with viewport: open on desktop, closed on mobile.
  // Re-evaluate on resize so rotating a tablet picks the right default.
  useEffect(() => {
    const mql = window.matchMedia("(max-width: 767px)");
    const apply = () => {
      setIsMobile(mql.matches);
      setSidebarOpen(!mql.matches);
    };
    apply();
    mql.addEventListener("change", apply);
    return () => mql.removeEventListener("change", apply);
  }, []);

  // Lock body scroll when the mobile drawer is open so background doesn't
  // scroll behind the overlay.
  useEffect(() => {
    if (isMobile && sidebarOpen) {
      const prev = document.body.style.overflow;
      document.body.style.overflow = "hidden";
      return () => { document.body.style.overflow = prev; };
    }
  }, [isMobile, sidebarOpen]);

  useEffect(() => {
    if (chatRef.current) chatRef.current.scrollTop = chatRef.current.scrollHeight;
  }, [exchanges]);

  function showToast(msg: string, type: "success" | "warning" = "success") {
    setToast({ msg, type });
    setTimeout(() => setToast(null), 2500);
  }

  function toggleDoc(id: string) {
    setSelected((prev) => {
      const next = new Set(prev);
      if (next.has(id)) next.delete(id);
      else next.add(id);
      return next;
    });
  }

  function selectAll() {
    const all = selected.size === documents.length;
    setSelected(all ? new Set() : new Set(documents.map((d) => d.id)));
  }

  const filteredDocs = documents.filter((d) =>
    d.filename.toLowerCase().includes(search.toLowerCase())
  );
  const categories: Record<string, DocumentInfo[]> = {};
  for (const d of filteredDocs) {
    const c = categoryFor(d);
    (categories[c] ||= []).push(d);
  }

  async function handleUpload(files: FileList | null) {
    if (!files || files.length === 0) return;
    for (const file of Array.from(files)) {
      setUploadState({ filename: file.name, steps: [], completed: false, error: null });
      let sawError: string | null = null;
      let sawComplete = false;
      try {
        await uploadDocument(file, (event: ProcessingEvent) => {
          setUploadState((prev) => {
            if (!prev) return prev;
            const steps = [...prev.steps];
            const existingIdx = steps.findIndex((s) => s.step === event.step);
            if (existingIdx >= 0) steps[existingIdx] = event;
            else steps.push(event);
            return { ...prev, steps };
          });
          if (event.status === "error") {
            sawError = event.detail ?? `${event.step} failed`;
          }
          if (event.step === "complete" && event.status === "done") {
            sawComplete = true;
          }
        });
      } catch (e) {
        sawError = e instanceof Error ? e.message : "Upload failed";
      }

      if (sawError) {
        setUploadState((prev) => (prev ? { ...prev, error: sawError, completed: true } : prev));
        showToast(`Upload failed: ${sawError}`, "warning");
      } else if (!sawComplete) {
        setUploadState((prev) =>
          prev ? { ...prev, error: "Upload ended without completion", completed: true } : prev
        );
        showToast("Upload ended without completion", "warning");
      } else {
        setUploadState((prev) => (prev ? { ...prev, completed: true } : prev));
        showToast(`"${file.name}" uploaded`);
        await fetchDocs();
        setTimeout(() => setUploadState((prev) => (prev?.filename === file.name ? null : prev)), 2000);
      }
    }
  }

  async function handleDelete(id: string) {
    try {
      await deleteDocument(id);
      showToast("Document deleted");
      await fetchDocs();
    } catch {
      showToast("Delete failed", "warning");
    }
  }

  async function send() {
    const prompt = input.trim();
    if (!prompt || streaming) return;
    if (selected.size === 0) {
      showToast("Please select at least one document", "warning");
      return;
    }

    const id = ++idCounter.current;
    const exchange: Exchange = {
      id,
      prompt,
      subQueries: deriveSubQueries(prompt),
      answer: "",
      sources: [],
      streaming: true,
      thinkingExpanded: false,
      completedSteps: 1,
      timestamp: new Date().toLocaleTimeString(),
      startedAt: Date.now(),
      elapsedMs: null,
    };
    setExchanges((prev) => [...prev, exchange]);
    setInput("");
    setStreaming(true);

    // If user has a subset selected, filter. If all selected, send null so BM25 runs too.
    const docIds =
      selected.size > 0 && selected.size < documents.length
        ? Array.from(selected)
        : null;

    try {
      await streamQuery(
        prompt,
        sessionId,
        ({ sources, session_id }) => {
          if (!sessionId) setSessionId(session_id);
          setExchanges((prev) =>
            prev.map((e) =>
              e.id === id ? { ...e, sources, completedSteps: 3 } : e
            )
          );
        },
        (token) => {
          setExchanges((prev) =>
            prev.map((e) =>
              e.id === id ? { ...e, answer: e.answer + token } : e
            )
          );
        },
        (sid) => {
          setSessionId(sid);
          setExchanges((prev) =>
            prev.map((e) =>
              e.id === id
                ? {
                    ...e,
                    streaming: false,
                    completedSteps: 4,
                    elapsedMs: Date.now() - e.startedAt,
                  }
                : e
            )
          );
        },
        5,
        docIds
      );
    } catch (e) {
      setExchanges((prev) =>
        prev.map((ex) =>
          ex.id === id
            ? {
                ...ex,
                answer:
                  "**Error:** " + (e instanceof Error ? e.message : "Unknown error"),
                streaming: false,
                elapsedMs: Date.now() - ex.startedAt,
              }
            : ex
        )
      );
    } finally {
      setStreaming(false);
    }
  }

  async function clearChat() {
    if (sessionId) {
      try {
        await clearSession(sessionId);
      } catch {}
    }
    setSessionId(null);
    setExchanges([]);
  }

  function exportChat() {
    if (exchanges.length === 0) {
      showToast("No chat history to export", "warning");
      return;
    }
    const activeNames = documents
      .filter((d) => selected.has(d.id))
      .map((d) => d.filename)
      .join(", ");
    let text = "Multi-Document RAG Chat Export\n" + "=".repeat(40) + "\n\n";
    text += `Exported: ${new Date().toLocaleString()}\n`;
    text += `Documents active: ${activeNames}\n\n`;
    text += "-".repeat(40) + "\n\n";
    exchanges.forEach((e, i) => {
      text += `Exchange ${i + 1}\n${"-".repeat(20)}\nUser: ${e.prompt}\n\n`;
      text += `Sub-queries:\n${e.subQueries.map((q) => `  • ${q}`).join("\n")}\n\n`;
      text += `Answer:\n${e.answer}\n\n${"=".repeat(40)}\n\n`;
    });
    navigator.clipboard.writeText(text).then(() => showToast("Chat exported"));
  }

  function copyAnswer(ex: Exchange) {
    navigator.clipboard.writeText(ex.answer).then(() => showToast("Answer copied"));
  }

  function toggleThinking(id: number) {
    setExchanges((prev) =>
      prev.map((e) => (e.id === id ? { ...e, thinkingExpanded: !e.thinkingExpanded } : e))
    );
  }

  function setPrompt(p: string) {
    setInput(p);
  }

  const activeCount = selected.size;

  return (
    <div className="h-[100dvh] flex overflow-hidden bg-rag-gray-50 relative">
      {/* Mobile backdrop (only when drawer is open on mobile) */}
      {isMobile && sidebarOpen && (
        <div
          className="fixed inset-0 bg-black/40 backdrop-blur-[2px] z-30 md:hidden animate-fade-in"
          onClick={() => setSidebarOpen(false)}
          aria-label="Close sidebar"
        />
      )}

      {/* Sidebar — drawer on mobile, inline on md+ */}
      <aside
        className={`
          fixed md:relative inset-y-0 left-0 z-40 h-full
          w-[85vw] max-w-xs md:w-72
          bg-white border-r border-rag-gray-200 flex flex-col flex-shrink-0
          shadow-xl md:shadow-none
          transition-transform md:transition-all duration-300 ease-out
          ${sidebarOpen
            ? "translate-x-0 md:w-72"
            : "-translate-x-full md:translate-x-0 md:w-0 md:overflow-hidden"}
        `}
      >
        <div className="p-4 border-b border-rag-gray-200">
          <div className="flex items-center gap-2.5 mb-1">
            <div className="relative w-8 h-8 flex-shrink-0">
              <div className="absolute inset-0 grid grid-cols-2 gap-0.5">
                <div className="rounded-sm bg-gradient-to-br from-indigo-500 to-indigo-600" />
                <div className="rounded-sm bg-gradient-to-br from-violet-500 to-violet-600" />
                <div className="rounded-sm bg-gradient-to-br from-fuchsia-500 to-fuchsia-600" />
                <div className="rounded-sm bg-gradient-to-br from-rose-500 to-rose-600" />
              </div>
            </div>
            <div className="flex flex-col leading-tight flex-1 min-w-0">
              <h1 className="text-lg font-semibold text-black tracking-tight">Mosaic</h1>
              <span className="text-[10px] text-rag-gray-400 tracking-wide">
                MULTI-DOC INTELLIGENCE
              </span>
            </div>
            {/* Close drawer button — mobile only */}
            <button
              onClick={() => setSidebarOpen(false)}
              className="md:hidden -mr-1 w-11 h-11 flex items-center justify-center rounded-lg text-rag-gray-500 hover:bg-rag-gray-100 active:bg-rag-gray-200 transition-colors"
              aria-label="Close menu"
            >
              <svg className="w-5 h-5" fill="none" stroke="currentColor" viewBox="0 0 24 24">
                <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M6 18L18 6M6 6l12 12" />
              </svg>
            </button>
          </div>
          <p className="text-xs text-rag-gray-500 mt-2 mb-3 italic">
            Synthesize answers across your documents.
          </p>
          <div className="flex items-center justify-between">
            <h2 className="text-sm font-medium text-black uppercase tracking-wider">
              Library
            </h2>
            <div className="flex items-center gap-2">
              {documents.length > 0 && (
                <button
                  onClick={selectAll}
                  className="text-xs text-indigo-600 hover:text-indigo-700 font-medium"
                >
                  {selected.size === documents.length ? "Clear" : "All"}
                </button>
              )}
              <span className="text-xs bg-indigo-100 text-indigo-700 px-2 py-0.5 rounded-full font-medium">
                {activeCount}/{documents.length}
              </span>
            </div>
          </div>
        </div>

        <div className="p-3">
          <div className="relative">
            <svg className="absolute left-2.5 top-1/2 -translate-y-1/2 w-4 h-4 text-rag-gray-400" fill="none" stroke="currentColor" viewBox="0 0 24 24">
              <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M21 21l-6-6m2-5a7 7 0 11-14 0 7 7 0 0114 0z" />
            </svg>
            <input
              type="text"
              placeholder="Search documents..."
              value={search}
              onChange={(e) => setSearch(e.target.value)}
              className="w-full pl-9 pr-3 py-2 text-sm border border-rag-gray-200 rounded-lg focus:outline-none focus:ring-2 focus:ring-rag-blue-500 focus:border-transparent bg-rag-gray-50"
            />
          </div>
        </div>

        <div className="flex-1 overflow-y-auto scrollbar-thin px-2 pb-2">
          {Object.keys(categories).length === 0 ? (
            <div className="text-center py-8 text-sm text-rag-gray-400">No documents</div>
          ) : (
            Object.entries(categories).map(([cat, docs]) => (
              <div key={cat} className="mb-1">
                <div className="flex items-center gap-1.5 px-2 py-1.5 text-xs font-medium text-rag-gray-400 uppercase tracking-wider">
                  <svg className="w-3.5 h-3.5" fill="none" stroke="currentColor" viewBox="0 0 24 24">
                    <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M3 7v10a2 2 0 002 2h14a2 2 0 002-2V9a2 2 0 00-2-2h-6l-2-2H5a2 2 0 00-2 2z" />
                  </svg>
                  {cat}
                </div>
                {docs.map((doc) => {
                  const checked = selected.has(doc.id);
                  return (
                    <div
                      key={doc.id}
                      className="flex items-center gap-2 px-2 py-2.5 mx-1 rounded-lg hover:bg-rag-gray-50 active:bg-rag-gray-100 transition-colors group min-h-[44px]"
                    >
                      <button
                        onClick={() => toggleDoc(doc.id)}
                        className={`w-5 h-5 border-2 ${
                          checked
                            ? "bg-rag-blue-600 border-rag-blue-600"
                            : "border-rag-gray-300 group-hover:border-rag-gray-400"
                        } rounded flex items-center justify-center transition-all flex-shrink-0`}
                        aria-label={checked ? "Deselect document" : "Select document"}
                      >
                        {checked && (
                          <svg className="w-3.5 h-3.5 text-white" fill="none" stroke="currentColor" viewBox="0 0 24 24">
                            <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={3} d="M5 13l4 4L19 7" />
                          </svg>
                        )}
                      </button>
                      <button
                        onClick={() => toggleDoc(doc.id)}
                        className="flex-1 min-w-0 text-left py-1"
                      >
                        <div className="text-sm text-rag-gray-700 truncate">
                          {doc.filename.replace(/\.[^.]+$/, "")}
                        </div>
                        <div className="text-xs text-rag-gray-400">
                          {doc.chunk_count} chunks · {doc.status}
                        </div>
                      </button>
                      <button
                        onClick={() => handleDelete(doc.id)}
                        className="w-9 h-9 flex items-center justify-center rounded-md text-rag-gray-400 hover:text-red-500 hover:bg-red-50 active:bg-red-100 md:opacity-0 md:group-hover:opacity-100 transition-all flex-shrink-0"
                        aria-label="Delete document"
                      >
                        <svg className="w-4 h-4" fill="none" stroke="currentColor" viewBox="0 0 24 24">
                          <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M19 7l-.867 12.142A2 2 0 0116.138 21H7.862a2 2 0 01-1.995-1.858L5 7m5 4v6m4-6v6m1-10V4a1 1 0 00-1-1h-4a1 1 0 00-1 1v3M4 7h16" />
                        </svg>
                      </button>
                    </div>
                  );
                })}
              </div>
            ))
          )}
        </div>

        <div className="p-4 border-t border-rag-gray-200">
          <input
            type="file"
            ref={fileInputRef}
            onChange={(e) => {
              handleUpload(e.target.files);
              e.target.value = "";
            }}
            className="hidden"
            accept=".pdf,.docx,.html,.htm,.txt,.md,.csv"
          />
          {uploadState && (
            <div
              className={`mb-3 rounded-lg border p-3 text-xs ${
                uploadState.error
                  ? "border-red-200 bg-red-50"
                  : uploadState.completed
                  ? "border-green-200 bg-green-50"
                  : "border-rag-blue-200 bg-rag-blue-50"
              }`}
            >
              <div className="font-medium text-rag-gray-800 truncate mb-1.5">
                {uploadState.filename}
              </div>
              <div className="space-y-1">
                {uploadState.steps.length === 0 && !uploadState.error && (
                  <div className="text-rag-gray-500">Starting…</div>
                )}
                {uploadState.steps.map((s) => (
                  <div key={s.step} className="flex items-center gap-1.5">
                    {s.status === "error" ? (
                      <svg className="w-3 h-3 text-red-500 flex-shrink-0" fill="none" stroke="currentColor" viewBox="0 0 24 24">
                        <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M6 18L18 6M6 6l12 12" />
                      </svg>
                    ) : s.status === "done" ? (
                      <svg className="w-3 h-3 text-green-600 flex-shrink-0" fill="none" stroke="currentColor" viewBox="0 0 24 24">
                        <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M5 13l4 4L19 7" />
                      </svg>
                    ) : (
                      <svg className="w-3 h-3 text-rag-blue-600 flex-shrink-0 animate-spin" fill="none" viewBox="0 0 24 24">
                        <circle cx="12" cy="12" r="10" stroke="currentColor" strokeWidth={2} className="opacity-25" />
                        <path fill="currentColor" d="M4 12a8 8 0 018-8v4a4 4 0 00-4 4H4z" />
                      </svg>
                    )}
                    <span className="capitalize text-rag-gray-700">{s.step}</span>
                    {s.detail && (
                      <span className="text-rag-gray-400 truncate">· {s.detail}</span>
                    )}
                  </div>
                ))}
                {uploadState.error && (
                  <div className="mt-1.5 text-red-600 font-medium">{uploadState.error}</div>
                )}
              </div>
            </div>
          )}
          <button
            onClick={() => fileInputRef.current?.click()}
            disabled={!!uploadState && !uploadState.completed}
            className="w-full flex items-center justify-center gap-2 px-4 py-4 bg-rag-blue-600 hover:bg-rag-blue-700 disabled:opacity-60 disabled:cursor-not-allowed text-white text-sm font-medium rounded-lg transition-colors shadow-sm"
          >
            <svg className="w-4 h-4" fill="none" stroke="currentColor" viewBox="0 0 24 24">
              <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M12 4v16m8-8H4" />
            </svg>
            {uploadState && !uploadState.completed ? "Uploading…" : "Upload Document"}
          </button>
          <div className="flex items-center justify-between mt-2 px-1 text-xs text-rag-gray-400">
            <span>PDF · DOCX · TXT · MD</span>
            <span>max 50MB</span>
          </div>
        </div>

      </aside>

      {/* Main */}
      <main className="flex-1 flex flex-col min-w-0 w-full">
        <header className="h-14 bg-white border-b border-rag-gray-200 flex items-center justify-between gap-2 px-3 sm:px-4 md:px-6 flex-shrink-0">
          <div className="flex items-center gap-2 sm:gap-3 min-w-0">
            <button
              onClick={() => setSidebarOpen(!sidebarOpen)}
              className="w-10 h-10 flex items-center justify-center rounded-lg hover:bg-rag-gray-100 active:bg-rag-gray-200 transition-colors flex-shrink-0"
              aria-label={sidebarOpen ? "Close menu" : "Open menu"}
            >
              <svg className="w-5 h-5 text-rag-gray-600" fill="none" stroke="currentColor" viewBox="0 0 24 24">
                <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M4 6h16M4 12h16M4 18h16" />
              </svg>
            </button>
            <div className="flex items-center gap-2 sm:gap-2.5 min-w-0">
              <span className="relative flex h-2 w-2 flex-shrink-0">
                <span className="animate-ping absolute inline-flex h-full w-full rounded-full bg-emerald-400 opacity-75" />
                <span className="relative inline-flex rounded-full h-2 w-2 bg-emerald-500" />
              </span>
              <span className="text-sm font-medium text-black truncate">
                {exchanges.length === 0 ? "New conversation" : `${exchanges.length} exchange${exchanges.length === 1 ? "" : "s"}`}
              </span>
              <span className="hidden sm:inline text-xs text-rag-gray-400 flex-shrink-0">
                · {activeCount}/{documents.length} docs active
              </span>
            </div>
          </div>
          <div className="flex items-center gap-2 flex-shrink-0">
            {/* Mobile: compact docs counter pill in place of the kbd hint */}
            <span className="sm:hidden text-[11px] font-medium text-indigo-700 bg-indigo-100 px-2 py-1 rounded-full">
              {activeCount}/{documents.length}
            </span>
            <kbd className="hidden sm:inline-flex items-center gap-1 px-2 py-1 text-[10px] font-medium text-rag-gray-500 bg-rag-gray-100 border border-rag-gray-200 rounded">
              <span className="text-rag-gray-400">⏎</span> to send
            </kbd>
          </div>
        </header>

        <div ref={chatRef} className="flex-1 overflow-y-auto scrollbar-thin p-3 sm:p-4 md:p-6 space-y-4 md:space-y-6">
          {exchanges.length === 0 ? (
            <div className="max-w-3xl mx-auto text-center py-8 md:py-16 px-2 animate-fade-in">
              <div className="relative w-14 h-14 md:w-16 md:h-16 mx-auto mb-4 md:mb-5">
                <div className="absolute inset-0 grid grid-cols-2 gap-1">
                  <div className="rounded-md bg-gradient-to-br from-indigo-500 to-indigo-600 shadow-lg shadow-indigo-500/20" />
                  <div className="rounded-md bg-gradient-to-br from-violet-500 to-violet-600 shadow-lg shadow-violet-500/20" />
                  <div className="rounded-md bg-gradient-to-br from-fuchsia-500 to-fuchsia-600 shadow-lg shadow-fuchsia-500/20" />
                  <div className="rounded-md bg-gradient-to-br from-rose-500 to-rose-600 shadow-lg shadow-rose-500/20" />
                </div>
              </div>
              <h3 className="text-xl md:text-2xl font-semibold text-black mb-2 tracking-tight">
                Welcome to Mosaic
              </h3>
              <p className="text-sm md:text-base text-rag-gray-500 mb-6 md:mb-8 max-w-md mx-auto">
                Ask a question and we'll assemble the answer from pieces across your library —
                with every source cited.
              </p>
              <div className="flex flex-col sm:flex-row sm:flex-wrap justify-center gap-2">
                {[
                  "What are the key differences between Transformer and CNN architectures?",
                  "Summarize the main findings from the uploaded research papers",
                  "What are the recommended best practices for fine-tuning LLMs?",
                ].map((p) => (
                  <button
                    key={p}
                    onClick={() => setPrompt(p)}
                    className="text-left sm:text-center px-4 py-3 sm:py-2 text-sm bg-white border border-rag-gray-200 rounded-lg hover:border-rag-blue-300 hover:bg-rag-blue-50 active:bg-rag-blue-100 text-rag-gray-600 hover:text-rag-blue-700 transition-all"
                  >
                    <span className="sm:hidden">{p}</span>
                    <span className="hidden sm:inline">{p.length > 45 ? p.slice(0, 42) + "..." : p}</span>
                  </button>
                ))}
              </div>
            </div>
          ) : (
            exchanges.map((ex) => <ExchangeCard key={ex.id} ex={ex} onToggle={toggleThinking} onCopyAnswer={copyAnswer} />)
          )}
        </div>

        <div className="border-t border-rag-gray-200 bg-white p-3 sm:p-4 flex-shrink-0 pb-[max(0.75rem,env(safe-area-inset-bottom))]">
          <div className="max-w-3xl mx-auto">
            <div className="relative border border-rag-gray-200 rounded-xl shadow-sm focus-within:ring-2 focus-within:ring-rag-blue-500 focus-within:border-transparent transition-all">
              <textarea
                rows={1}
                placeholder="Ask a question about your documents..."
                value={input}
                onChange={(e) => setInput(e.target.value)}
                onKeyDown={(e) => {
                  if (e.key === "Enter" && !e.shiftKey) {
                    e.preventDefault();
                    send();
                  }
                }}
                className="w-full pl-3 pr-14 sm:pl-4 py-3 text-base sm:text-sm text-black bg-transparent resize-none focus:outline-none scrollbar-thin max-h-32"
                style={{ minHeight: 48 }}
              />
              <div className="absolute right-2 bottom-2">
                <button
                  onClick={send}
                  disabled={streaming || !input.trim()}
                  className="w-10 h-10 flex items-center justify-center bg-rag-blue-600 hover:bg-rag-blue-700 active:bg-rag-blue-800 text-white rounded-lg transition-colors disabled:opacity-50 disabled:cursor-not-allowed"
                  aria-label="Send"
                >
                  <svg className="w-4 h-4" fill="none" stroke="currentColor" viewBox="0 0 24 24">
                    <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M12 19l9 2-9-18-9 18 9-2zm0 0v-8" />
                  </svg>
                </button>
              </div>
            </div>
            <div className="flex items-center justify-between mt-2 px-1 gap-2">
              <div className="flex items-center gap-1.5 text-xs text-rag-gray-400 min-w-0">
                <svg className="w-3.5 h-3.5 flex-shrink-0" fill="none" stroke="currentColor" viewBox="0 0 24 24">
                  <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M13 16h-1v-4h-1m1-4h.01M21 12a9 9 0 11-18 0 9 9 0 0118 0z" />
                </svg>
                <span className="truncate">
                  <span className="hidden sm:inline">Querying across </span>
                  {activeCount} doc{activeCount !== 1 ? "s" : ""}
                </span>
              </div>
              <span className="text-xs text-rag-gray-400 flex-shrink-0">{input.length}/2000</span>
            </div>
          </div>
        </div>
      </main>

      {toast && (
        <div
          className={`fixed left-3 right-3 bottom-24 mx-auto max-w-fit sm:left-auto sm:right-6 sm:bottom-6 sm:mx-0 ${
            toast.type === "success" ? "bg-rag-gray-900" : "bg-amber-600"
          } text-white px-4 py-2.5 rounded-lg shadow-lg text-sm flex items-center gap-2 z-50 animate-slide-in`}
        >
          <svg className="w-4 h-4" fill="none" stroke="currentColor" viewBox="0 0 24 24">
            {toast.type === "success" ? (
              <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M5 13l4 4L19 7" />
            ) : (
              <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M12 9v2m0 4h.01m-6.938 4h13.856c1.54 0 2.502-1.667 1.732-3L13.732 4c-.77-1.333-2.694-1.333-3.464 0L3.34 16c-.77 1.333.192 3 1.732 3z" />
            )}
          </svg>
          <span>{toast.msg}</span>
        </div>
      )}
    </div>
  );
}

function ExchangeCard({
  ex,
  onToggle,
  onCopyAnswer,
}: {
  ex: Exchange;
  onToggle: (id: number) => void;
  onCopyAnswer: (ex: Exchange) => void;
}) {
  return (
    <div className="max-w-3xl mx-auto animate-slide-in">
      <div className="bg-white rounded-xl border border-rag-gray-200 shadow-sm overflow-hidden">
        {/* User query */}
        <div className="px-3 sm:px-5 py-3 sm:py-4 bg-rag-blue-50/50 border-b border-rag-gray-100">
          <div className="flex items-start gap-2.5 sm:gap-3">
            <div className="w-7 h-7 bg-rag-blue-600 rounded-lg flex items-center justify-center flex-shrink-0 mt-0.5">
              <svg className="w-4 h-4 text-white" fill="none" stroke="currentColor" viewBox="0 0 24 24">
                <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M16 7a4 4 0 11-8 0 4 4 0 018 0zM12 14a7 7 0 00-7 7h14a7 7 0 00-7-7z" />
              </svg>
            </div>
            <div className="flex-1 min-w-0">
              <p className="text-sm font-medium text-rag-gray-900 break-words">{ex.prompt}</p>
            </div>
          </div>
        </div>

        {/* Thinking */}
        <div className="border-b border-rag-gray-100">
          <button
            onClick={() => onToggle(ex.id)}
            className="w-full flex items-center gap-2 px-3 sm:px-5 py-3 hover:bg-rag-gray-50 active:bg-rag-gray-100 transition-colors text-left"
          >
            <div
              className={`px-2 py-0.5 rounded text-[10px] font-semibold ${
                ex.streaming
                  ? "thinking-badge text-amber-700"
                  : "bg-emerald-50 text-emerald-700 border border-emerald-200"
              }`}
            >
              {ex.streaming ? "THINKING" : "DONE"}
            </div>
            <span className="text-xs font-medium text-rag-gray-600">Processing Steps</span>
            <svg
              className={`w-4 h-4 text-rag-gray-400 ml-auto transition-transform ${
                ex.thinkingExpanded ? "rotate-180" : ""
              }`}
              fill="none"
              stroke="currentColor"
              viewBox="0 0 24 24"
            >
              <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M19 9l-7 7-7-7" />
            </svg>
          </button>
          <div className={`collapse-content ${ex.thinkingExpanded ? "expanded" : "collapsed"}`}>
            <div className="px-3 sm:px-5 pb-3 space-y-1">
              {THINKING_STEPS.map((step, i) => {
                const done = i < ex.completedSteps;
                return (
                  <div key={i} className="flex items-start gap-2 py-1">
                    <div
                      className={`w-5 h-5 rounded-full ${
                        done ? "bg-green-100" : "bg-rag-blue-100"
                      } flex items-center justify-center flex-shrink-0 mt-0.5`}
                    >
                      {done ? (
                        <svg className="w-3 h-3 text-green-600" fill="none" stroke="currentColor" viewBox="0 0 24 24">
                          <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M5 13l4 4L19 7" />
                        </svg>
                      ) : (
                        <svg className="w-3 h-3 text-rag-blue-600" fill="none" stroke="currentColor" viewBox="0 0 24 24">
                          <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M12 8v4l3 3m6-3a9 9 0 11-18 0 9 9 0 0118 0z" />
                        </svg>
                      )}
                    </div>
                    <span className="text-xs text-rag-gray-600 flex-1 min-w-0">{step}</span>
                    <span
                      className={`text-[10px] font-medium flex-shrink-0 ${
                        done ? "text-green-600" : "text-rag-blue-600"
                      }`}
                    >
                      {done ? "Completed" : i === ex.completedSteps ? "In Progress" : "Pending"}
                    </span>
                  </div>
                );
              })}
            </div>
          </div>
        </div>

        {/* Final answer */}
        <div className="px-3 sm:px-5 py-4">
          <div className="flex items-center gap-2 mb-3">
            <div className="w-5 h-5 bg-rag-blue-600 rounded flex items-center justify-center flex-shrink-0">
              <svg className="w-3 h-3 text-white" fill="none" stroke="currentColor" viewBox="0 0 24 24">
                <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M9.663 17h4.673M12 3v1m6.364 1.636l-.707.707M21 12h-1M4 12H3m3.343-5.657l-.707-.707m2.828 9.9a5 5 0 117.072 0l-.548.547A3.374 3.374 0 0014 18.469V19a2 2 0 11-4 0v-.531c0-.895-.356-1.754-.988-2.386l-.548-.547z" />
              </svg>
            </div>
            <span className="text-sm font-semibold text-rag-gray-900">Final Answer</span>
            {ex.streaming && (
              <span className="text-[10px] text-rag-blue-600 font-medium animate-pulse">Streaming...</span>
            )}
          </div>
          <div className="prose prose-sm max-w-none text-rag-gray-700 text-sm leading-relaxed break-words">
            <ReactMarkdown>{ex.answer || (ex.streaming ? "_Thinking..._" : "")}</ReactMarkdown>
          </div>

          {ex.sources.length > 0 && (
            <div className="mt-4 pt-3 border-t border-rag-gray-100">
              <div className="text-xs font-semibold text-rag-gray-500 uppercase tracking-wider mb-2">
                Sources ({ex.sources.length})
              </div>
              <div className="space-y-2">
                {ex.sources.map((s, i) => (
                  <details key={i} className="group">
                    <summary className="cursor-pointer text-xs text-rag-blue-700 hover:text-rag-blue-900 flex flex-wrap items-center gap-x-2 gap-y-1 py-1">
                      <span className="font-medium break-all">{s.document}</span>
                      {s.pages.length > 0 && (
                        <span className="text-rag-gray-400">pp. {s.pages.join(", ")}</span>
                      )}
                      <span className="ml-auto text-rag-gray-400 flex-shrink-0">score {s.score.toFixed(4)}</span>
                    </summary>
                    <div className="mt-2 pl-3 border-l-2 border-rag-blue-100 space-y-2">
                      {s.chunks.map((c, j) => (
                        <div key={j} className="text-xs text-rag-gray-600 bg-rag-gray-50 p-2 rounded break-words">
                          {c.text}
                        </div>
                      ))}
                    </div>
                  </details>
                ))}
              </div>
            </div>
          )}
        </div>

        {/* Actions */}
        <div className="px-3 sm:px-5 py-2.5 bg-rag-gray-50 border-t border-rag-gray-100 flex items-center gap-2 flex-wrap">
          <button
            onClick={() => onCopyAnswer(ex)}
            className="flex items-center gap-1.5 px-3 py-1.5 text-xs text-rag-gray-600 hover:text-rag-blue-600 hover:bg-white rounded-md transition-colors"
          >
            <svg className="w-3.5 h-3.5" fill="none" stroke="currentColor" viewBox="0 0 24 24">
              <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M8 16H6a2 2 0 01-2-2V6a2 2 0 012-2h8a2 2 0 012 2v2m-6 12h8a2 2 0 002-2v-8a2 2 0 00-2-2h-8a2 2 0 00-2 2v8a2 2 0 002 2z" />
            </svg>
            Copy
          </button>
          <div className="flex-1" />
          {ex.elapsedMs !== null && (
            <span className="flex items-center gap-1 text-[10px] text-rag-gray-500">
              <svg className="w-3 h-3" fill="none" stroke="currentColor" viewBox="0 0 24 24">
                <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M12 8v4l3 3m6-3a9 9 0 11-18 0 9 9 0 0118 0z" />
              </svg>
              {(ex.elapsedMs / 1000).toFixed(2)}s
            </span>
          )}
          <span className="text-[10px] text-rag-gray-400">{ex.timestamp}</span>
        </div>
      </div>
    </div>
  );
}
