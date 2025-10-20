"use client"

import {
  useEffect,
  useRef,
  useState,
  KeyboardEvent,
  ChangeEvent,
} from "react"
import { Button } from "@/components/ui/button"
import { Textarea } from "@/components/ui/textarea"
import { Send, Square } from "lucide-react"
import { useChat, Chat } from "@/components/providers/chat-provider"
import { ChatMessageData } from "./chat-message"
import { useRouter } from "next/navigation"
import { subscribeNewChat } from "@/lib/chat-store"
import { useStream } from "@/components/providers/stream-provider"

export function Composer() {
  const [value, setValue] = useState("")
  const textareaRef = useRef<HTMLTextAreaElement>(null)
  const { currentChat, setChats, setCurrentChat, updateCurrentChat } = useChat()
  const { streamingChatId, abort } = useStream()
  const router = useRouter()

  useEffect(() => {
    subscribeNewChat(() => {
      setValue("")
      if (textareaRef.current) {
        textareaRef.current.style.height = "auto"
        textareaRef.current.focus()
      }
    })
  }, [])

  const resize = (el: HTMLTextAreaElement) => {
    const lineHeight = parseInt(window.getComputedStyle(el).lineHeight)
    const maxHeight = lineHeight * 6
    el.style.height = "auto"
    el.style.height = Math.min(el.scrollHeight, maxHeight) + "px"
  }

  const handleChange = (e: ChangeEvent<HTMLTextAreaElement>) => {
    setValue(e.target.value)
    resize(e.target)
  }

  const isStreaming = streamingChatId === currentChat?.id
  const isOtherStreaming =
    streamingChatId !== null && streamingChatId !== currentChat?.id

  const handleKeyDown = (e: KeyboardEvent<HTMLTextAreaElement>) => {
    if (e.key === "Enter" && !e.shiftKey) {
      e.preventDefault()
      if (isStreaming) abort()
      else handleSend()
    }
  }

  const generateTitle = (text: string) => {
    const words = text.trim().split(/\s+/).slice(0, 8)
    return words.join(" ").replace(/[.,!?;:]+$/, "")
  }

  const handleSend = () => {
    const message = value.trim()
    if (!message) return
    const newMessage: ChatMessageData = {
      id: Date.now().toString(),
      role: "user",
      name: "You",
      content: message,
      timestamp: new Date().toISOString(),
    }

    const title = generateTitle(message)

    if (!currentChat) {
      const chat: Chat = {
        id: crypto.randomUUID(),
        title,
        messages: [newMessage],
      }
      setChats((prev) => [chat, ...prev])
      setCurrentChat(chat)
      router.push(`/chat/${chat.id}`)
    } else {
      const isFirst = currentChat.messages.length === 0
      updateCurrentChat({
        messages: [...currentChat.messages, newMessage],
        ...(isFirst ? { title } : {}),
      })
    }

    setValue("")
    if (textareaRef.current) {
      textareaRef.current.style.height = "auto"
    }
  }

  return (
    <div
      id="composer"
      className="flex items-end gap-2 border-t p-4 print:hidden"
    >
      <Textarea
        ref={textareaRef}
        value={value}
        onChange={handleChange}
        onKeyDown={handleKeyDown}
        placeholder="Type a message..."
        rows={1}
        className="flex-1 resize-none"
        disabled={isOtherStreaming}
      />
      {isStreaming ? (
        <Button
          type="button"
          size="icon"
          className="self-stretch"
          onClick={abort}
          aria-label="Abort streaming"
          variant="destructive"
        >
          <Square className="size-4" />
        </Button>
      ) : (
        <Button
          type="button"
          size="icon"
          className="self-stretch"
          onClick={handleSend}
          disabled={isOtherStreaming || !value.trim()}
          aria-label="Send message"
        >
          <Send className="size-4" />
        </Button>
      )}
    </div>
  )
}

export default Composer
