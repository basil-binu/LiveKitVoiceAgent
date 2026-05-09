import { useEffect, useMemo, useRef, useState, useCallback } from "react";
import {
  Room,
  RoomEvent,
  Track,
  type RemoteTrack,
  type RemoteTrackPublication,
  type RemoteParticipant,
  type Participant,
  type TranscriptionSegment,
} from "livekit-client";
import {
  Mic,
  MicOff,
  Volume2,
  PhoneOff,
  ChevronDown,
  Upload,
  X,
  FileText,
  Loader2,
  CheckCircle2,
  AlertCircle,
} from "lucide-react";

const API_BASE = import.meta.env.VITE_API_BASE_URL as string;

type MicState = "idle" | "connecting" | "connected" | "user" | "agent" | "disconnected";

type Message = {
  id: string;
  role: "user" | "agent";
  text: string;
  ts: Date;
  interim?: boolean;
};

type DocItem = { filename: string; size_kb: number; type: string; origin: "static" | "uploaded" };
type Toast = { id: number; kind: "success" | "error"; text: string };
type RagSource = { name: string; snippet?: string };

function fmtTime(d: Date) {
  return d.toTimeString().slice(0, 5);
}

function normalizeSource(s: any): RagSource {
  if (typeof s === "string") return { name: s };
  const name = s.source || s.filename || s.name || "Unknown source";
  const snippet = s.snippet || s.content || s.text;
  return { name, snippet: typeof snippet === "string" ? snippet : undefined };
}

export default function VoiceAgent() {
  const ROOM_NAME = useMemo(() => "voice-room-" + crypto.randomUUID(), []);
  const roomRef = useRef<Room | null>(null);
  const audioElsRef = useRef<HTMLMediaElement[]>([]);

  const [micState, setMicState] = useState<MicState>("idle");
  const [messages, setMessages] = useState<Message[]>([]);
  const interimIdRef = useRef<string | null>(null);

  const [sources, setSources] = useState<RagSource[] | null>(null);

  const [promptOpen, setPromptOpen] = useState(false);
  const [systemPrompt, setSystemPrompt] = useState("");
  const [kbOpen, setKbOpen] = useState(true);

  const [docs, setDocs] = useState<{ static: DocItem[]; uploaded: DocItem[] }>({
    static: [],
    uploaded: [],
  });
  const [uploadProgress, setUploadProgress] = useState<number | null>(null);
  const [uploadingName, setUploadingName] = useState<string | null>(null);
  const [dragOver, setDragOver] = useState(false);

  const [toasts, setToasts] = useState<Toast[]>([]);
  const toast = useCallback((kind: Toast["kind"], text: string) => {
    const id = Date.now() + Math.random();
    setToasts((t) => [...t, { id, kind, text }]);
    setTimeout(() => setToasts((t) => t.filter((x) => x.id !== id)), 3000);
  }, []);

  const logEndRef = useRef<HTMLDivElement>(null);
  useEffect(() => {
    logEndRef.current?.scrollIntoView({ behavior: "smooth" });
  }, [messages]);

  const refreshDocs = useCallback(async () => {
    try {
      const res = await fetch(`${API_BASE}/list-docs`);
      if (!res.ok) throw new Error("Failed");
      const data = await res.json();
      setDocs({ static: data.static || [], uploaded: data.uploaded || [] });
    } catch (e) {
      // silent
    }
  }, []);

  useEffect(() => {
    refreshDocs();
  }, [refreshDocs]);

  const resetUI = useCallback(() => {
    setMicState("idle");
    interimIdRef.current = null;
    audioElsRef.current.forEach((el) => el.remove());
    audioElsRef.current = [];
  }, []);

  const handleConnect = async () => {
    if (micState !== "idle" && micState !== "disconnected") return;
    setMicState("connecting");
    try {
      const url = `${API_BASE}/token?room=${encodeURIComponent(ROOM_NAME)}&system_prompt=${encodeURIComponent(systemPrompt)}`;
      const res = await fetch(url);
      if (!res.ok) throw new Error("Token fetch failed");
      const data = (await res.json()) as { token: string; url: string };

      const room = new Room();
      roomRef.current = room;

      room.on(RoomEvent.TrackSubscribed, (track: RemoteTrack, _pub: RemoteTrackPublication) => {
        if (track.kind === Track.Kind.Audio) {
          const el = track.attach();
          el.style.display = "none";
          document.body.appendChild(el);
          audioElsRef.current.push(el);
        }
      });

      room.on(
        RoomEvent.TranscriptionReceived,
        (segments: TranscriptionSegment[], participant?: Participant) => {
          const isAgent = participant?.identity !== room.localParticipant.identity;
          for (const seg of segments) {
            if (isAgent) {
              if (seg.final) {
                setMessages((m) => [
                  ...m,
                  { id: seg.id || crypto.randomUUID(), role: "agent", text: seg.text, ts: new Date() },
                ]);
              }
            } else {
              if (seg.final) {
                setMessages((m) => {
                  const filtered = interimIdRef.current
                    ? m.filter((x) => x.id !== interimIdRef.current)
                    : m;
                  interimIdRef.current = null;
                  return [
                    ...filtered,
                    { id: seg.id || crypto.randomUUID(), role: "user", text: seg.text, ts: new Date() },
                  ];
                });
              } else {
                setMessages((m) => {
                  if (interimIdRef.current) {
                    return m.map((x) =>
                      x.id === interimIdRef.current ? { ...x, text: seg.text } : x
                    );
                  }
                  const id = "interim-" + crypto.randomUUID();
                  interimIdRef.current = id;
                  return [...m, { id, role: "user", text: seg.text, ts: new Date(), interim: true }];
                });
              }
            }
          }
        }
      );

      room.on(RoomEvent.ActiveSpeakersChanged, (speakers) => {
        const localId = room.localParticipant.identity;
        const agentSpeaking = speakers.some((s) => s.identity !== localId);
        const localSpeaking = speakers.some((s) => s.identity === localId);
        if (agentSpeaking) setMicState("agent");
        else if (localSpeaking) setMicState("user");
        else setMicState("connected");
      });

      room.on(RoomEvent.DataReceived, (payload: Uint8Array, _p?: RemoteParticipant) => {
        try {
          const text = new TextDecoder().decode(payload);
          const msg = JSON.parse(text);
          if (msg.type === "rag_sources" && Array.isArray(msg.sources)) {
            setSources(msg.sources.map(normalizeSource));
          }
        } catch {}
      });

      room.on(RoomEvent.Disconnected, () => {
        resetUI();
      });

      await room.connect(data.url, data.token);
      await room.localParticipant.setMicrophoneEnabled(true);
      setMicState("connected");
    } catch (e: any) {
      toast("error", e?.message || "Connection failed");
      setMicState("idle");
    }
  };

  const handleDisconnect = async () => {
    await roomRef.current?.disconnect();
    roomRef.current = null;
    resetUI();
  };

  const updatePrompt = async () => {
    try {
      const res = await fetch(`${API_BASE}/update-prompt`, {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({ room: ROOM_NAME, system_prompt: systemPrompt }),
      });
      if (!res.ok) throw new Error("Failed to update prompt");
      const data = await res.json();
      if (data.success) toast("success", "Prompt updated");
      else throw new Error("Update failed");
    } catch (e: any) {
      toast("error", e?.message || "Failed");
    }
  };

  const uploadFile = (file: File) =>
    new Promise<void>((resolve, reject) => {
      const allowed = [".pdf", ".txt", ".docx"];
      const ok = allowed.some((ext) => file.name.toLowerCase().endsWith(ext));
      if (!ok) {
        toast("error", `Skipped ${file.name}: unsupported type`);
        return resolve();
      }
      const xhr = new XMLHttpRequest();
      const fd = new FormData();
      fd.append("file", file);
      xhr.open("POST", `${API_BASE}/upload`);
      xhr.upload.onprogress = (e) => {
        if (e.lengthComputable) setUploadProgress(Math.round((e.loaded / e.total) * 100));
      };
      xhr.onload = () => {
        if (xhr.status >= 200 && xhr.status < 300) {
          toast("success", `Uploaded ${file.name}`);
          resolve();
        } else {
          toast("error", `Upload failed: ${file.name}`);
          reject(new Error("upload failed"));
        }
      };
      xhr.onerror = () => {
        toast("error", `Upload error: ${file.name}`);
        reject(new Error("upload error"));
      };
      xhr.send(fd);
    });

  const handleFiles = async (files: FileList | File[]) => {
    const arr = Array.from(files);
    for (const f of arr) {
      setUploadingName(f.name);
      setUploadProgress(0);
      try {
        await uploadFile(f);
      } catch {}
      await refreshDocs();
    }
    setUploadingName(null);
    setUploadProgress(null);
  };

  const deleteDoc = async (filename: string) => {
    try {
      const res = await fetch(`${API_BASE}/delete-doc/${encodeURIComponent(filename)}`, {
        method: "DELETE",
      });
      if (!res.ok) throw new Error("Delete failed");
      toast("success", `Deleted ${filename}`);
      await refreshDocs();
    } catch (e: any) {
      toast("error", e?.message || "Delete failed");
    }
  };

  // Mic visuals
  const micVisual = (() => {
    switch (micState) {
      case "idle":
      case "disconnected":
        return { bg: "bg-slate-800", ring: "", icon: <Mic className="w-12 h-12 text-slate-300" /> };
      case "connecting":
        return {
          bg: "bg-slate-700 animate-pulse",
          ring: "",
          icon: <Loader2 className="w-12 h-12 text-slate-200 animate-spin" />,
        };
      case "connected":
        return {
          bg: "bg-violet-600",
          ring: "mic-glow-purple",
          icon: <Mic className="w-12 h-12 text-white" />,
        };
      case "user":
        return {
          bg: "bg-blue-500",
          ring: "mic-glow-blue",
          icon: <Mic className="w-12 h-12 text-white" />,
        };
      case "agent":
        return {
          bg: "bg-emerald-500",
          ring: "mic-glow-green",
          icon: <Volume2 className="w-12 h-12 text-white" />,
        };
    }
  })();

  const isConnected = ["connected", "user", "agent"].includes(micState);

  return (
    <div className="min-h-screen bg-slate-950 text-slate-100" style={{ fontFamily: "Inter, system-ui, sans-serif" }}>
      {/* Toasts */}
      <div className="fixed top-4 right-4 z-50 flex flex-col gap-2">
        {toasts.map((t) => (
          <div
            key={t.id}
            className={`animate-toast-in flex items-center gap-2 bg-slate-800 border border-slate-700 border-l-4 ${
              t.kind === "success" ? "border-l-emerald-500" : "border-l-red-500"
            } px-4 py-3 rounded-lg shadow-lg min-w-[240px]`}
          >
            {t.kind === "success" ? (
              <CheckCircle2 className="w-4 h-4 text-emerald-400" />
            ) : (
              <AlertCircle className="w-4 h-4 text-red-400" />
            )}
            <span className="text-sm">{t.text}</span>
          </div>
        ))}
      </div>

      <div className="max-w-6xl mx-auto px-4 sm:px-6 py-10">
        <header className="mb-10 text-center">
          <h1 className="text-3xl sm:text-4xl font-semibold tracking-tight">Voice Agent</h1>
          <p className="mt-2 text-sm text-slate-400 tracking-wide">
            Real-time voice conversations with retrieval-augmented context
          </p>
        </header>

        {/* Mic hero */}
        <div className="flex flex-col items-center justify-center mb-12">
          <button
            onClick={isConnected ? undefined : handleConnect}
            disabled={micState === "connecting" || isConnected}
            aria-label="Microphone"
            className={`relative w-28 h-28 sm:w-32 sm:h-32 rounded-full flex items-center justify-center transition-colors duration-200 ${micVisual.bg} ${micVisual.ring} ${
              !isConnected && micState !== "connecting" ? "hover:brightness-110 cursor-pointer" : ""
            } focus:outline-none focus:ring-4 focus:ring-violet-500/40`}
          >
            {micVisual.icon}
          </button>
          <div className="mt-5 text-sm uppercase tracking-widest text-slate-400">
            {micState === "idle" || micState === "disconnected"
              ? "Tap to connect"
              : micState === "connecting"
                ? "Connecting…"
                : micState === "connected"
                  ? "Listening"
                  : micState === "user"
                    ? "You're speaking"
                    : "Agent speaking"}
          </div>
          {isConnected && (
            <button
              onClick={handleDisconnect}
              className="mt-5 inline-flex items-center gap-2 px-4 py-2 rounded-full bg-slate-800 hover:bg-slate-700 border border-slate-700 text-sm transition-colors"
            >
              <PhoneOff className="w-4 h-4" />
              Disconnect
            </button>
          )}
        </div>

        <div className="grid grid-cols-1 lg:grid-cols-2 gap-6">
          {/* Left column: conversation + sources */}
          <div className="flex flex-col gap-6">
            {/* Conversation */}
            <section className="bg-slate-900 border border-slate-800 rounded-2xl flex flex-col h-[440px]">
              <div className="px-5 py-3 border-b border-slate-800 text-xs uppercase tracking-widest text-slate-400">
                Conversation
              </div>
              <div className="flex-1 overflow-y-auto px-5 py-4 space-y-3">
                {messages.length === 0 ? (
                  <div className="h-full flex items-center justify-center text-slate-500 text-sm">
                    Conversation will appear here...
                  </div>
                ) : (
                  messages.map((m) => (
                    <div
                      key={m.id}
                      className={`flex flex-col animate-bubble-in ${m.role === "user" ? "items-end" : "items-start"}`}
                    >
                      <div
                        className={`max-w-[80%] px-4 py-2.5 rounded-2xl leading-relaxed text-sm ${
                          m.role === "user"
                            ? "bg-violet-600 text-white"
                            : "bg-slate-800 text-slate-100 border border-slate-700"
                        } ${m.interim ? "italic opacity-60" : ""}`}
                      >
                        {m.text}
                      </div>
                      <span className="text-[11px] text-slate-500 mt-1 px-1">{fmtTime(m.ts)}</span>
                    </div>
                  ))
                )}
                <div ref={logEndRef} />
              </div>
            </section>

            {/* RAG sources */}
            <section className="bg-slate-900 border border-slate-800 rounded-2xl">
              <div className="px-5 py-3 border-b border-slate-800 text-xs uppercase tracking-widest text-slate-400">
                Sources Used
              </div>
              <div className="p-4 space-y-2">
                {!sources || sources.length === 0 ? (
                  <div className="text-sm text-slate-500 px-1 py-3">No sources retrieved yet</div>
                ) : (
                  sources.map((s, i) => (
                    <div
                      key={i}
                      className="animate-source-in bg-slate-800 border border-slate-700 border-l-4 border-l-violet-500 rounded-xl px-4 py-3"
                      style={{ animationDelay: `${i * 60}ms` }}
                    >
                      <div className="flex items-center gap-2">
                        <FileText className="w-4 h-4 text-violet-400" />
                        <span className="text-violet-400 font-medium text-sm">{s.name}</span>
                      </div>
                      {s.snippet && (
                        <p className="mt-1.5 text-xs text-slate-400 leading-relaxed line-clamp-3">
                          {s.snippet}
                        </p>
                      )}
                    </div>
                  ))
                )}
              </div>
            </section>
          </div>

          {/* Right column: prompt + KB */}
          <div className="flex flex-col gap-6">
            {/* Prompt editor */}
            <section className="bg-slate-900 border border-slate-800 rounded-2xl overflow-hidden">
              <button
                onClick={() => setPromptOpen((v) => !v)}
                className="w-full flex items-center justify-between px-5 py-3 text-xs uppercase tracking-widest text-slate-400 hover:bg-slate-800/40 transition-colors"
              >
                <span>System Prompt</span>
                <ChevronDown
                  className={`w-4 h-4 transition-transform duration-300 ${promptOpen ? "rotate-180" : ""}`}
                />
              </button>
              <div
                className="grid transition-[grid-template-rows] duration-300 ease-out"
                style={{ gridTemplateRows: promptOpen ? "1fr" : "0fr" }}
              >
                <div className="overflow-hidden">
                  <div className="p-4 space-y-3 border-t border-slate-800">
                    <textarea
                      value={systemPrompt}
                      onChange={(e) => setSystemPrompt(e.target.value)}
                      rows={6}
                      placeholder="You are a helpful voice assistant..."
                      className="w-full bg-slate-800 border border-slate-700 rounded-xl px-3 py-2.5 text-sm placeholder:text-slate-500 focus:outline-none focus:ring-2 focus:ring-violet-500/40 focus:border-violet-500/50 resize-none"
                    />
                    <div className="flex justify-end">
                      <button
                        onClick={updatePrompt}
                        className="inline-flex items-center gap-2 px-4 py-2 rounded-lg bg-violet-600 hover:bg-violet-500 text-white text-sm transition-colors"
                      >
                        Update Prompt
                      </button>
                    </div>
                  </div>
                </div>
              </div>
            </section>

            {/* Knowledge base */}
            <section className="bg-slate-900 border border-slate-800 rounded-2xl overflow-hidden">
              <button
                onClick={() => setKbOpen((v) => !v)}
                className="w-full flex items-center justify-between px-5 py-3 text-xs uppercase tracking-widest text-slate-400 hover:bg-slate-800/40 transition-colors"
              >
                <span>Knowledge Base</span>
                <ChevronDown
                  className={`w-4 h-4 transition-transform duration-300 ${kbOpen ? "rotate-180" : ""}`}
                />
              </button>
              <div
                className="grid transition-[grid-template-rows] duration-300 ease-out"
                style={{ gridTemplateRows: kbOpen ? "1fr" : "0fr" }}
              >
                <div className="overflow-hidden">
                  <div className="p-4 space-y-4 border-t border-slate-800">
                    {/* Drop zone */}
                    <label
                      onDragOver={(e) => {
                        e.preventDefault();
                        setDragOver(true);
                      }}
                      onDragLeave={() => setDragOver(false)}
                      onDrop={(e) => {
                        e.preventDefault();
                        setDragOver(false);
                        if (e.dataTransfer.files) handleFiles(e.dataTransfer.files);
                      }}
                      className={`flex flex-col items-center justify-center gap-2 border border-dashed rounded-xl px-4 py-8 cursor-pointer transition-colors ${
                        dragOver
                          ? "border-violet-500 bg-violet-500/5"
                          : "border-slate-700 hover:border-slate-600 hover:bg-slate-800/40"
                      }`}
                    >
                      <Upload className="w-5 h-5 text-slate-400" />
                      <div className="text-sm text-slate-300">Drag files here or click to browse</div>
                      <div className="text-xs text-slate-500">PDF, TXT, DOCX</div>
                      <input
                        type="file"
                        multiple
                        accept=".pdf,.txt,.docx"
                        className="hidden"
                        onChange={(e) => {
                          if (e.target.files) handleFiles(e.target.files);
                          e.target.value = "";
                        }}
                      />
                    </label>

                    {uploadProgress !== null && (
                      <div className="space-y-1.5">
                        <div className="flex justify-between text-xs text-slate-400">
                          <span className="truncate">{uploadingName}</span>
                          <span>{uploadProgress}%</span>
                        </div>
                        <div className="h-1.5 bg-slate-800 rounded-full overflow-hidden">
                          <div
                            className="h-full bg-violet-500 transition-all duration-200"
                            style={{ width: `${uploadProgress}%` }}
                          />
                        </div>
                      </div>
                    )}

                    {/* Doc list */}
                    <div className="space-y-2">
                      {[...docs.static, ...docs.uploaded].length === 0 ? (
                        <div className="text-sm text-slate-500 py-2">No documents yet</div>
                      ) : (
                        [...docs.static, ...docs.uploaded].map((d) => (
                          <div
                            key={d.origin + d.filename}
                            className="flex items-center gap-3 bg-slate-800 border border-slate-700 rounded-xl px-3 py-2"
                          >
                            <FileText className="w-4 h-4 text-slate-400 shrink-0" />
                            <div className="flex-1 min-w-0">
                              <div className="text-sm truncate">{d.filename}</div>
                              <div className="text-[11px] text-slate-500">{d.size_kb} KB</div>
                            </div>
                            <span
                              className={`text-[10px] uppercase tracking-wider px-2 py-0.5 rounded-full ${
                                d.origin === "static"
                                  ? "bg-violet-500/15 text-violet-300 border border-violet-500/30"
                                  : "bg-emerald-500/15 text-emerald-300 border border-emerald-500/30"
                              }`}
                            >
                              {d.origin}
                            </span>
                            {d.origin === "uploaded" && (
                              <button
                                onClick={() => deleteDoc(d.filename)}
                                className="p-1 rounded-md hover:bg-slate-700 text-slate-400 hover:text-red-400 transition-colors"
                                aria-label={`Delete ${d.filename}`}
                              >
                                <X className="w-4 h-4" />
                              </button>
                            )}
                          </div>
                        ))
                      )}
                    </div>
                  </div>
                </div>
              </div>
            </section>
          </div>
        </div>

        <footer className="mt-10 text-center text-[11px] text-slate-600 tracking-wider">
          ROOM · {ROOM_NAME}
        </footer>
      </div>
    </div>
  );
}
