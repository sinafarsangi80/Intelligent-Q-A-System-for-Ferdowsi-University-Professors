// lib/chat-api.ts
import { CHAT_STREAM_URL } from "./config"
import { parseSSE, SSEEvent } from "./sse"

export interface ChatMessageLite {
    role: "user" | "assistant" | "system"
    content: string
}

export interface StreamOptions {
    messages: ChatMessageLite[]
    chatId?: string
    signal?: AbortSignal
    headers?: Record<string, string>
}

/** Map your frontend messages to the backend request shape here (adjust if needed). */
function buildRequestPayload(opts: StreamOptions) {
    const last = opts.messages.at(-1)
    // ASSUMPTION: backend accepts { query, messages, chat_id }
    // If your backend expects a different shape, tweak only this function.
    return {
        query: last?.content ?? "",
        messages: opts.messages,
        chat_id: opts.chatId,
    }
}

export async function* streamChatFromBackend(
    opts: StreamOptions,
): AsyncGenerator<SSEEvent> {
    if (!CHAT_STREAM_URL) {
        throw new Error(
            "Missing NEXT_PUBLIC_BACKEND_URL. Set it in .env.local (e.g., http://localhost:8000).",
        )
    }

    const res = await fetch(CHAT_STREAM_URL, {
        method: "POST",
        headers: {
            "Content-Type": "application/json",
            Accept: "text/event-stream",
            ...(opts.headers || {}),
        },
        body: JSON.stringify(buildRequestPayload(opts)),
        signal: opts.signal,
    })

    if (!res.ok || !res.body) {
        const text = await res.text().catch(() => "")
        throw new Error(`Stream request failed: ${res.status} ${res.statusText}\n${text}`)
    }

    for await (const evt of parseSSE(res.body)) {
        yield evt
    }
}
