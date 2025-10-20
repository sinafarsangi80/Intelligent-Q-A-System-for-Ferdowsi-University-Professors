/* eslint-disable @typescript-eslint/no-explicit-any */

/**
 * Minimal SSE parser for "data: {...}\n\n" events.
 * Works over fetch(Response.body) streams. No external deps.
 */
export type SSEEvent =
    | { type: "token"; delta: string }
    | { type: "route"; route: string }
    | { type: "sources"; sources: any }
    | { type: "error"; error: string }
    | { type: "done" }
    | Record<string, any>

export async function* parseSSE(
    stream: ReadableStream<Uint8Array>,
): AsyncGenerator<SSEEvent> {
    const reader = stream.getReader()
    const decoder = new TextDecoder()
    let buffer = ""

    try {
        while (true) {
            const { value, done } = await reader.read()
            if (done) break
            buffer += decoder.decode(value, { stream: true })

            let sepIndex
            while ((sepIndex = buffer.indexOf("\n\n")) !== -1) {
                const raw = buffer.slice(0, sepIndex)
                buffer = buffer.slice(sepIndex + 2)

                const lines = raw.split("\n")
                const dataLines: string[] = []
                for (const line of lines) {
                    if (line.startsWith("data:")) dataLines.push(line.slice(5).trimStart())
                }
                if (dataLines.length === 0) continue
                const payload = dataLines.join("\n")
                if (!payload) continue
                try {
                    const obj = JSON.parse(payload)
                    yield obj as SSEEvent
                } catch (err: any) {
                    yield { type: "error", error: `Bad JSON chunk: ${err?.message || err}` }
                }
            }
        }
    } finally {
        if (buffer.trim().startsWith("data:")) {
            const data = buffer.trim().replace(/^data:\s*/gm, "")
            try { yield JSON.parse(data) as SSEEvent } catch {}
        }
        yield { type: "done" }
    }
}
