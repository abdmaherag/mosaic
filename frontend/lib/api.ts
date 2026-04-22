import type { DocumentInfo, DocumentSource, ProcessingEvent, SourceCitation } from "./types";

const BASE = process.env.NEXT_PUBLIC_API_URL ?? "http://localhost:8000";

function getDeviceId(): string {
  let id = localStorage.getItem("device_id");
  if (!id) {
    id = crypto.randomUUID();
    localStorage.setItem("device_id", id);
  }
  return id;
}

// Parse raw SSE buffer into event blocks
function* parseSSEBuffer(buffer: string): Generator<{ event: string; data: string }> {
  const blocks = buffer.split("\n\n");
  for (const block of blocks) {
    if (!block.trim()) continue;
    const lines = block.split("\n");
    let event = "message";
    const dataLines: string[] = [];
    for (const line of lines) {
      if (line.startsWith("event: ")) event = line.slice(7).trim();
      else if (line.startsWith("data: ")) dataLines.push(line.slice(6));
      else if (line === "data:") dataLines.push("");
    }
    if (dataLines.length > 0) {
      yield { event, data: dataLines.join("\n") };
    }
  }
}

export async function listDocuments(): Promise<DocumentInfo[]> {
  const res = await fetch(`${BASE}/documents/`, {
    headers: { "X-Device-ID": getDeviceId() },
  });
  if (!res.ok) throw new Error("Failed to fetch documents");
  return res.json();
}

export async function deleteDocument(id: string): Promise<void> {
  const res = await fetch(`${BASE}/documents/${id}`, {
    method: "DELETE",
    headers: { "X-Device-ID": getDeviceId() },
  });
  if (!res.ok) throw new Error("Failed to delete document");
}

export async function uploadDocument(
  file: File,
  onEvent: (event: ProcessingEvent) => void
): Promise<void> {
  const formData = new FormData();
  formData.append("file", file);

  const res = await fetch(`${BASE}/documents/upload`, {
    method: "POST",
    headers: { "X-Device-ID": getDeviceId() },
    body: formData,
  });
  if (!res.ok) {
    const body = await res.json().catch(() => null);
    throw new Error(body?.detail ?? "Upload failed");
  }

  const reader = res.body!.getReader();
  const decoder = new TextDecoder();
  let buffer = "";

  while (true) {
    const { done, value } = await reader.read();
    if (done) break;
    buffer += decoder.decode(value, { stream: true }).replace(/\r\n/g, "\n");

    // Keep only the last incomplete block in buffer
    const lastDouble = buffer.lastIndexOf("\n\n");
    if (lastDouble === -1) continue;
    const complete = buffer.slice(0, lastDouble + 2);
    buffer = buffer.slice(lastDouble + 2);

    for (const { data } of parseSSEBuffer(complete)) {
      try {
        onEvent(JSON.parse(data) as ProcessingEvent);
      } catch {
        // ignore malformed events
      }
    }
  }
}

export interface CitationsPayload {
  citations: SourceCitation[];
  sources: DocumentSource[];
  has_relevant: boolean;
  session_id: string;
}

export async function streamQuery(
  question: string,
  sessionId: string | null,
  onCitations: (payload: CitationsPayload) => void,
  onToken: (token: string) => void,
  onDone: (sessionId: string) => void,
  nResults: number = 5,
  docIds: string[] | null = null
): Promise<void> {
  const res = await fetch(`${BASE}/query/stream`, {
    method: "POST",
    headers: { "Content-Type": "application/json", "X-Device-ID": getDeviceId() },
    body: JSON.stringify({
      question,
      session_id: sessionId,
      n_results: nResults,
      doc_ids: docIds,
    }),
  });
  if (!res.ok) throw new Error("Query failed");

  const reader = res.body!.getReader();
  const decoder = new TextDecoder();
  let buffer = "";

  while (true) {
    const { done, value } = await reader.read();
    if (done) break;
    buffer += decoder.decode(value, { stream: true }).replace(/\r\n/g, "\n");

    const lastDouble = buffer.lastIndexOf("\n\n");
    if (lastDouble === -1) continue;
    const complete = buffer.slice(0, lastDouble + 2);
    buffer = buffer.slice(lastDouble + 2);

    for (const { event, data } of parseSSEBuffer(complete)) {
      if (event === "citations") {
        try { onCitations(JSON.parse(data)); } catch { /* skip */ }
      } else if (event === "token") {
        onToken(data);
      } else if (event === "done") {
        try { onDone(JSON.parse(data).session_id); } catch { /* skip */ }
      }
    }
  }
}

export async function streamCompare(
  question: string,
  docIds: string[],
  onCitations: (payload: CitationsPayload) => void,
  onToken: (token: string) => void,
  onDone: () => void,
  nResults: number = 3
): Promise<void> {
  const res = await fetch(`${BASE}/query/compare/stream`, {
    method: "POST",
    headers: { "Content-Type": "application/json", "X-Device-ID": getDeviceId() },
    body: JSON.stringify({ question, doc_ids: docIds, n_results: nResults }),
  });
  if (!res.ok) throw new Error("Compare query failed");

  const reader = res.body!.getReader();
  const decoder = new TextDecoder();
  let buffer = "";

  while (true) {
    const { done, value } = await reader.read();
    if (done) break;
    buffer += decoder.decode(value, { stream: true }).replace(/\r\n/g, "\n");

    const lastDouble = buffer.lastIndexOf("\n\n");
    if (lastDouble === -1) continue;
    const complete = buffer.slice(0, lastDouble + 2);
    buffer = buffer.slice(lastDouble + 2);

    for (const { event, data } of parseSSEBuffer(complete)) {
      if (event === "citations") {
        try { onCitations(JSON.parse(data)); } catch { /* skip */ }
      } else if (event === "token") {
        onToken(data);
      } else if (event === "done") {
        onDone();
      }
    }
  }
}

export async function clearSession(sessionId: string): Promise<void> {
  const res = await fetch(`${BASE}/query/session/${sessionId}`, {
    method: "DELETE",
    headers: { "X-Device-ID": getDeviceId() },
  });
  if (!res.ok) throw new Error("Failed to clear session");
}
