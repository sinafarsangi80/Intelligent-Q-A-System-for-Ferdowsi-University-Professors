"use client"

import {
  createContext,
  useContext,
  useState,
  ReactNode,
  useCallback,
  useMemo,
} from "react"

interface StreamContextValue {
  streamingChatId: string | null
  startStreaming: (chatId: string, controller: AbortController) => void
  stopStreaming: () => void
  abort: () => void
}

const StreamContext = createContext<StreamContextValue | undefined>(undefined)

export function useStream() {
  const context = useContext(StreamContext)
  if (!context) throw new Error("useStream must be used within a StreamProvider")
  return context
}

export function StreamProvider({ children }: { children: ReactNode }) {
  const [streamingChatId, setStreamingChatId] = useState<string | null>(null)
  const [controller, setController] = useState<AbortController | null>(null)

  const startStreaming = useCallback(
    (chatId: string, controller: AbortController) => {
      setStreamingChatId(chatId)
      setController(controller)
    },
    []
  )

  const stopStreaming = useCallback(() => {
    setStreamingChatId(null)
    setController(null)
  }, [])

  const abort = useCallback(() => {
    controller?.abort()
  }, [controller])

  const value = useMemo<StreamContextValue>(
    () => ({ streamingChatId, startStreaming, stopStreaming, abort }),
    [streamingChatId, startStreaming, stopStreaming, abort]
  )

  return <StreamContext.Provider value={value}>{children}</StreamContext.Provider>
}

export default StreamProvider
