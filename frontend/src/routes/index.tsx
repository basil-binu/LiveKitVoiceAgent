import { createFileRoute } from "@tanstack/react-router";
import VoiceAgent from "@/components/VoiceAgent";

export const Route = createFileRoute("/")({
  component: VoiceAgent,
  head: () => ({
    meta: [
      { title: "Voice Agent" },
      { name: "description", content: "Real-time voice agent with retrieval-augmented context" },
    ],
  }),
});
