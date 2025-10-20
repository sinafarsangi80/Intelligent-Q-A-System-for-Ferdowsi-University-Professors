/* eslint-disable @typescript-eslint/no-explicit-any */
/* eslint-disable @typescript-eslint/no-non-null-asserted-optional-chain */

"use client"

import { useEffect, useRef, useState } from "react"
import { ScrollArea } from "@/components/ui/scroll-area"
import { Skeleton } from "@/components/ui/skeleton"
import { useChat } from "@/components/providers/chat-provider"
import { streamChatFromBackend } from "@/lib/chat-api"
import { toast } from "sonner"
import {ChatMessage, ChatMessageData} from "@/components/chat/chat-message";
import {SourceDoc, SourcesBar} from "@/components/chat/sources-bar";
import {ThinkingBlock} from "@/components/chat/thinking-block";
import { useStream } from "@/components/providers/stream-provider"

const OPEN_TAG = "<think>"
const CLOSE_TAG = "</think>"

export function ChatList({ messages }: { messages: ChatMessageData[] }) {
    const endRef = useRef<HTMLDivElement | null>(null)
    const { currentChat, updateCurrentChat } = useChat()
    const { startStreaming, stopStreaming } = useStream()

    const [pendingMessage, setPendingMessage] = useState<ChatMessageData | null>(null)
    const [loading, setLoading] = useState(true)

    // thinking UI state
    const [isThinking, setIsThinking] = useState(false)
    const [thinkingText, setThinkingText] = useState("")

    useEffect(() => {
        endRef.current?.scrollIntoView({ behavior: "smooth" })
    }, [messages, pendingMessage, isThinking, thinkingText])

    useEffect(() => {
        const t = setTimeout(() => setLoading(false), 300)
        return () => clearTimeout(t)
    }, [])

    useEffect(() => {
        if (!currentChat || messages.length === 0) return
        const last = messages[messages.length - 1]
        if (last.role !== "user") return

        let cancelled = false
        const controller = new AbortController()

        // local buffers for this assistant reply
        let answer = ""                  // visible final answer
        let thinking = ""                // transient thinking buffer
        let inThink = false              // state machine for <think>…</think>
        let capturedSources: SourceDoc[] | null = null // sources from SSE "sources" event

        // pending assistant message (visible as it streams)
        const reply: ChatMessageData = {
            id: crypto.randomUUID(),
            role: "assistant",
            name: "Assistant",
            content: "",
            timestamp: new Date().toISOString(),
        }

        async function handleStream() {
            setPendingMessage(reply)
            startStreaming(currentChat?.id!, controller)

            try {
                for await (const evt of streamChatFromBackend({
                    messages: messages.map(m => ({ role: m.role, content: m.content })),
                    chatId: currentChat?.id,
                    signal: controller.signal,
                })) {
                    if (cancelled) return
                    const type = (evt as any).type

                    if (type === "sources") {
                        // capture once (we'll attach after stream ends)
                        const sources = (evt as any).sources as SourceDoc[] | undefined
                        if (sources?.length) capturedSources = sources
                    } else if (type === "token") {
                        const delta = (evt as any).delta ?? ""
                        if (!delta) continue

                        // consume tags possibly mixed within the same delta
                        let cursor = delta
                        while (cursor.length) {
                            if (!inThink) {
                                const openIdx = cursor.indexOf(OPEN_TAG)
                                if (openIdx !== -1) {
                                    // append any visible text before <think>
                                    if (openIdx > 0) {
                                        answer += cursor.slice(0, openIdx)
                                    }
                                    inThink = true
                                    setIsThinking(true)
                                    cursor = cursor.slice(openIdx + OPEN_TAG.length)
                                    continue
                                } else {
                                    // in visible mode; also strip a stray </think> defensively
                                    const closeIdx = cursor.indexOf(CLOSE_TAG)
                                    if (closeIdx !== -1) {
                                        answer += cursor.replaceAll(CLOSE_TAG, "")
                                        cursor = ""
                                    } else {
                                        answer += cursor
                                        cursor = ""
                                    }
                                }
                            } else {
                                // inside <think>…</think>
                                const closeIdx = cursor.indexOf(CLOSE_TAG)
                                if (closeIdx !== -1) {
                                    thinking += cursor.slice(0, closeIdx)
                                    setThinkingText(thinking)

                                    // close the thinking panel
                                    inThink = false
                                    setIsThinking(false)
                                    thinking = ""
                                    setThinkingText("")

                                    // continue with the remainder after </think>
                                    cursor = cursor.slice(closeIdx + CLOSE_TAG.length)
                                    continue
                                } else {
                                    thinking += cursor
                                    setThinkingText(thinking)
                                    cursor = ""
                                }
                            }
                        }

                        // reflect visible answer as it grows
                        setPendingMessage({ ...reply, content: answer })
                    } else if (type === "error") {
                        toast.error((evt as any).error ?? "Stream error")
                    } else if (type === "done") {
                        break
                    }
                }
            } catch (err: any) {
                if (err?.name === "AbortError") {
                    if (!cancelled) toast.info("پاسخ متوقف شد")
                } else {
                    toast.error(err?.message || "Failed to stream from backend")
                }
            } finally {
                // ensure thinking panel is gone even if stream ended without </think>
                setIsThinking(false)
                setThinkingText("")

                if (!cancelled) {
                    // finalize: attach sources to the stored message
                    const finalizedAssistant: ChatMessageData = {
                        ...reply,
                        content: answer,
                        ...(capturedSources ? { sources: capturedSources } : {}),
                    }

                    // persist in chat state
                    updateCurrentChat({ messages: [...messages, finalizedAssistant] })
                    setPendingMessage(null)
                }
                stopStreaming()
            }
        }

        handleStream()
        return () => {
            cancelled = true
            controller.abort()
            stopStreaming()
        }
    }, [messages, currentChat, updateCurrentChat, startStreaming, stopStreaming])

    return (
        <ScrollArea className="h-full">
            <div className="mx-auto flex max-w-3xl flex-col gap-4 p-4">
                {loading ? (
                    <>
                        <Skeleton className="h-6 w-24" />
                        <Skeleton className="h-24 w-full" />
                        <Skeleton className="h-24 w-4/5" />
                    </>
                ) : (
                    <>
                        {messages.map((m) => (
                            <div key={m.id} className="flex flex-col gap-2">
                                <ChatMessage message={m} />
                                {/* render sources only after the assistant message is fully finalized */}
                                {m.role === "assistant" && m.sources?.length ? (
                                    <SourcesBar sources={m.sources} />
                                ) : null}
                            </div>
                        ))}

                        {/* transient thinking + pending stream */}
                        {isThinking && <ThinkingBlock text={thinkingText} />}

                        {pendingMessage && (
                            <div className="flex flex-col gap-2">
                                <ChatMessage key={pendingMessage.id} message={pendingMessage} />
                                {/* Do NOT show sources while streaming; requirement says after stream is done */}
                            </div>
                        )}
                    </>
                )}
                <div ref={endRef} />
            </div>
        </ScrollArea>
    )
}
