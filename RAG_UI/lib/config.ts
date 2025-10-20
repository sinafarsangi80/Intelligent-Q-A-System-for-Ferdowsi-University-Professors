// lib/config.ts
export const BACKEND_URL =
    process.env.NEXT_PUBLIC_BACKEND_URL?.replace(/\/$/, "") || "";

export const CHAT_STREAM_URL = BACKEND_URL
    ? `${BACKEND_URL}/chat/stream`
    : "";
