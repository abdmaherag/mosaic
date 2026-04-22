export interface DocumentInfo {
  id: string;
  filename: string;
  content_type: string;
  chunk_count: number;
  status: "ready" | "error" | "processing";
}

export interface SourceCitation {
  document: string;
  chunk_text: string;
  score: number;
  page: number | null;
  section: string | null;
}

export interface SourceChunk {
  text: string;
  score: number;
  page: number | null;
  section: string | null;
}

export interface DocumentSource {
  document: string;
  pages: number[];
  score: number;
  chunks: SourceChunk[];
}

export interface ProcessingEvent {
  step: string;
  status: string;
  detail: string | null;
  progress: number | null;
}

export interface ChatMessage {
  id: string;
  role: "user" | "assistant";
  content: string;
  sources?: DocumentSource[];
  streaming?: boolean;
  isComparison?: boolean;
}
