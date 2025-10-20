"use client"

import {
  createContext,
  useContext,
  useEffect,
  useState,
  ReactNode,
  Dispatch,
  SetStateAction,
  useCallback,
  useMemo,
} from "react"
import { getLocalStorage, setLocalStorage } from "@/lib/storage"
import { ChatMessageData as ChatMessage } from "@/components/chat/chat-message"

export interface Chat {
  id: string
  title: string
  messages: ChatMessage[]
}

export type Settings = Record<string, unknown>

interface ChatContextValue {
  chats: Chat[]
  currentChat: Chat | null
  settings: Settings
  setChats: Dispatch<SetStateAction<Chat[]>>
  setCurrentChat: (chat: Chat | null) => void
  setSettings: (settings: Settings) => void
  updateCurrentChat: (update: Partial<Chat>) => void
  clearCurrentChat: () => void
}

const ChatContext = createContext<ChatContextValue | undefined>(undefined)

export function useChat() {
  const context = useContext(ChatContext)
  if (!context) throw new Error("useChat must be used within a ChatProvider")
  return context
}

export function ChatProvider({ children }: { children: ReactNode }) {
  const [chats, setChats] = useState<Chat[]>(() => getLocalStorage<Chat[]>("chats", []))
  const [currentChat, setCurrentChat] = useState<Chat | null>(() =>
    getLocalStorage<Chat | null>("currentChat", null)
  )
  const [settings, setSettings] = useState<Settings>(() =>
    getLocalStorage<Settings>("settings", {})
  )

  const updateCurrentChat = useCallback((update: Partial<Chat>) => {
    setCurrentChat((prev) => {
      if (!prev) return prev
      const next = { ...prev, ...update }
      setChats((prevChats) => prevChats.map((c) => (c.id === next.id ? next : c)))
      return next
    })
  }, [])

  const clearCurrentChat = useCallback(() => {
    updateCurrentChat({ messages: [] })
  }, [updateCurrentChat])

  useEffect(() => {
    setLocalStorage("chats", chats)
  }, [chats])

  useEffect(() => {
    setLocalStorage("currentChat", currentChat)
  }, [currentChat])

  useEffect(() => {
    setLocalStorage("settings", settings)
  }, [settings])

  const value = useMemo<ChatContextValue>(
    () => ({
      chats,
      currentChat,
      settings,
      setChats,
      setCurrentChat,
      setSettings,
      updateCurrentChat,
      clearCurrentChat,
    }),
    [
      chats,
      currentChat,
      settings,
      setChats,
      setCurrentChat,
      setSettings,
      updateCurrentChat,
      clearCurrentChat,
    ]
  )

  return <ChatContext.Provider value={value}>{children}</ChatContext.Provider>
}

export default ChatProvider
