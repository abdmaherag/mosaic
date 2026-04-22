"use client";

import { useEffect, useRef } from "react";
import ReactMarkdown from "react-markdown";
import type { ChatMessage } from "@/lib/types";
import SourceCitations from "./SourceCitations";

interface Props {
  messages: ChatMessage[];
  isStreaming: boolean;
  onSend: (question: string) => void;
  onClearSession: () => void;
  onOpenSidebar: () => void;
  hasDocuments: boolean;
  compareMode: boolean;
  compareDocCount: number;
  topK: number;
  onTopKChange: (k: number) => void;
}

const EXAMPLE_QUESTIONS = [
  "What were the main risks identified across all uploaded reports?",
  "How does the Q3 revenue compare to the product strategy budget?",
  "What is the competitive landscape and how are we positioned?",
  "What are the key growth areas identified in the market research?",
];

export default function ChatPanel({
  messages,
  isStreaming,
  onSend,
  onClearSession,
  onOpenSidebar,
  hasDocuments,
  compareMode,
  compareDocCount,
  topK,
  onTopKChange,
}: Props) {
  const bottomRef = useRef<HTMLDivElement>(null);
  const inputRef = useRef<HTMLInputElement>(null);

  useEffect(() => {
    bottomRef.current?.scrollIntoView({ behavior: "smooth" });
  }, [messages]);

  function handleKeyDown(e: React.KeyboardEvent<HTMLInputElement>) {
    if (e.key === "Enter" && !e.shiftKey) {
      e.preventDefault();
      submit();
    }
  }

  function submit() {
    const q = inputRef.current?.value.trim();
    if (!q || isStreaming) return;
    if (compareMode && compareDocCount < 2) return;
    inputRef.current!.value = "";
    onSend(q);
  }

  const inputDisabled = !hasDocuments || isStreaming || (compareMode && compareDocCount < 2);

  return (
    <div className="flex flex-col h-full">
      {/* Messages area */}
      <div className="flex-1 overflow-y-auto px-6 py-6">
        {messages.length === 0 ? (
          <div className="flex flex-col items-center justify-center h-full text-center max-w-xl mx-auto">
            <h1 className="font-heading text-slate-deep text-3xl md:text-4xl mb-3">
              Multi-Document RAG
            </h1>
            <p className="text-sm text-charcoal/50 leading-relaxed mb-8 max-w-md">
              Ask natural-language questions across your entire uploaded corpus.
              Responses include source citations, confidence scores, and cross-document analysis.
            </p>

            {hasDocuments && (
              <>
                <div className="flex items-center gap-3 mb-4 w-full max-w-md">
                  <div className="flex-1 h-px bg-border" />
                  <span className="text-[11px] font-semibold text-charcoal/50 uppercase tracking-[0.15em]">
                    Try a question
                  </span>
                  <div className="flex-1 h-px bg-border" />
                </div>

                <div className="space-y-2 w-full max-w-md">
                  {EXAMPLE_QUESTIONS.map((q) => (
                    <button
                      key={q}
                      onClick={() => onSend(q)}
                      className="group/q w-full flex items-center gap-2 text-left text-sm text-charcoal/70 hover:text-coral
                        border-b border-dashed border-border hover:border-coral/30
                        pb-2 transition-colors focus-visible:outline-none focus-visible:text-coral"
                    >
                      <span className="opacity-0 group-hover/q:opacity-100 transition-opacity text-coral shrink-0">&rarr;</span>
                      <span>{q}</span>
                    </button>
                  ))}
                </div>
              </>
            )}

            {!hasDocuments && (
              <p className="text-xs text-charcoal/50 mt-2">
                Upload documents from the sidebar to get started.
              </p>
            )}
          </div>
        ) : (
          <div className="space-y-6 max-w-3xl mx-auto">
            {messages.map((msg) => (
              <div key={msg.id} className={`flex gap-3 ${msg.role === "user" ? "justify-end" : "justify-start"}`}>
                {msg.role === "assistant" && (
                  <div className={`h-7 w-7 rounded-full flex items-center justify-center shrink-0 mt-0.5 ${
                    msg.isComparison ? "bg-teal" : "bg-coral"
                  }`}>
                    {msg.isComparison ? (
                      <svg className="h-4 w-4 text-white" fill="none" viewBox="0 0 24 24" stroke="currentColor">
                        <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2}
                          d="M9 17V7m0 10a2 2 0 01-2 2H5a2 2 0 01-2-2V7a2 2 0 012-2h2a2 2 0 012 2m0 10a2 2 0 002 2h2a2 2 0 002-2M9 7a2 2 0 012-2h2a2 2 0 012 2m0 10V7" />
                      </svg>
                    ) : (
                      <svg className="h-4 w-4 text-white" fill="none" viewBox="0 0 24 24" stroke="currentColor">
                        <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2}
                          d="M9.813 15.904L9 18.75l-.813-2.846a4.5 4.5 0 00-3.09-3.09L2.25 12l2.846-.813a4.5 4.5 0 003.09-3.09L9 5.25l.813 2.846a4.5 4.5 0 003.09 3.09L15.75 12l-2.846.813a4.5 4.5 0 00-3.09 3.09z" />
                      </svg>
                    )}
                  </div>
                )}

                <div className={`max-w-[80%] ${msg.role === "user" ? "order-first" : ""}`}>
                  {msg.isComparison && !msg.streaming && msg.content && (
                    <p className="text-xs font-medium text-teal mb-1.5">Comparison</p>
                  )}
                  <div
                    className={`rounded-2xl px-4 py-2.5 text-sm leading-relaxed ${
                      msg.role === "user"
                        ? "bg-coral text-white rounded-tr-sm whitespace-pre-wrap"
                        : msg.isComparison
                        ? "bg-teal-light border border-teal/20 text-charcoal rounded-tl-sm prose prose-sm max-w-none"
                        : "card text-charcoal rounded-tl-sm prose prose-sm max-w-none"
                    }`}
                  >
                    {msg.role === "user" ? (
                      msg.content
                    ) : (
                      <ReactMarkdown>{msg.content}</ReactMarkdown>
                    )}
                    {msg.streaming && (
                      <span className="inline-block w-2 h-4 ml-0.5 bg-coral animate-pulse rounded-sm align-text-bottom" />
                    )}
                  </div>

                  {msg.role === "assistant" && msg.sources && msg.sources.length > 0 && !msg.streaming && (
                    <SourceCitations sources={msg.sources} />
                  )}
                </div>
              </div>
            ))}
            <div ref={bottomRef} />
          </div>
        )}
      </div>

      {/* Bottom input bar */}
      <div className="border-t border-border bg-card px-6 py-3 shrink-0">
        <div className="flex items-center gap-4 max-w-3xl mx-auto mb-2">
          {compareMode && compareDocCount >= 2 && (
            <p className="text-xs text-teal">
              Comparing {compareDocCount} document{compareDocCount !== 1 ? "s" : ""}
            </p>
          )}
          <div className="flex items-center gap-2 ml-auto">
            <span className="text-[11px] font-semibold text-charcoal/50 uppercase tracking-wider">Top-K</span>
            <div className="flex items-center gap-1">
              <button
                onClick={() => onTopKChange(Math.max(1, topK - 1))}
                className="w-7 h-7 rounded flex items-center justify-center text-xs text-charcoal/50 hover:text-charcoal hover:bg-warm-white border border-border transition-colors focus-visible:outline-none focus-visible:ring-2 focus-visible:ring-coral/50"
              >
                -
              </button>
              <span className="w-6 text-center text-sm font-mono font-semibold text-deep">{topK}</span>
              <button
                onClick={() => onTopKChange(Math.min(20, topK + 1))}
                className="w-7 h-7 rounded flex items-center justify-center text-xs text-charcoal/50 hover:text-charcoal hover:bg-warm-white border border-border transition-colors focus-visible:outline-none focus-visible:ring-2 focus-visible:ring-coral/50"
              >
                +
              </button>
            </div>
            <input
              type="range"
              min={1}
              max={20}
              value={topK}
              onChange={(e) => onTopKChange(Number(e.target.value))}
              className="w-20 h-1 accent-coral cursor-pointer"
            />
          </div>
        </div>
        <div className="flex gap-3 items-center max-w-3xl mx-auto">
          <input
            ref={inputRef}
            type="text"
            placeholder={
              !hasDocuments
                ? "Upload documents first..."
                : compareMode && compareDocCount < 2
                ? "Select at least 2 documents to compare..."
                : "Ask across your documents..."
            }
            disabled={inputDisabled}
            onKeyDown={handleKeyDown}
            className="flex-1 rounded-lg bg-warm-white border border-border px-4 py-2.5 text-sm text-charcoal placeholder-charcoal/30
              focus:outline-none focus:border-coral focus:ring-1 focus:ring-coral/20
              focus-visible:ring-2 focus-visible:ring-coral/50
              disabled:opacity-40 disabled:cursor-not-allowed transition-colors"
          />
          <button
            onClick={submit}
            disabled={inputDisabled}
            className={`rounded-lg px-5 py-2.5 text-sm font-medium text-white transition-colors shrink-0
              disabled:opacity-40 disabled:cursor-not-allowed
              focus-visible:outline-none focus-visible:ring-2 focus-visible:ring-coral/50 focus-visible:ring-offset-2 ${
              compareMode ? "bg-teal hover:bg-teal-hover" : "bg-coral hover:bg-coral-hover"
            }`}
          >
            {isStreaming ? (
              <span className="h-4 w-4 block rounded-full border-2 border-white border-t-transparent animate-spin" />
            ) : (
              "Send"
            )}
          </button>
        </div>
      </div>
    </div>
  );
}
