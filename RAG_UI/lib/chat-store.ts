import { Chat } from "@/components/providers/chat-provider"
import { getLocalStorage } from "./storage"

export type NewChatListener = () => void

const listeners = new Set<NewChatListener>()

export function subscribeNewChat(listener: NewChatListener) {
  listeners.add(listener)
  return () => listeners.delete(listener)
}

export function triggerNewChat() {
  for (const listener of listeners) {
    listener()
  }
}

export function getChatById(id: string): Chat | null {
  const chats = getLocalStorage<Chat[]>("chats", [])
  return chats.find((c) => c.id === id) ?? null
}
