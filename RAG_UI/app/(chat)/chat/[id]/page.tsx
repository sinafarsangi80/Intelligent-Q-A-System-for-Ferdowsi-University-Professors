"use client"

import { useEffect, useRef, useState } from "react"
import { ChatList } from "@/components/chat-transcript"
import { Composer } from "@/components/chat/composer"
import { TopBar } from "@/components/chat/top-bar"
import { Chat, useChat } from "@/components/providers/chat-provider"
import { getChatById } from "@/lib/chat-store"
import { useReactToPrint } from "react-to-print"

export default function ChatIdPage({ params }: { params: { id: string } }) {
  const { currentChat, setCurrentChat } = useChat()
  const [chat, setChat] = useState<Chat | null>(null)
  const printRef = useRef<HTMLDivElement>(null)
  const handlePrint = useReactToPrint({ contentRef: printRef })

  useEffect(() => {
    const stored = getChatById(params.id)
    setChat(stored)
    setCurrentChat(stored)
  }, [params.id, setCurrentChat])

  if (!chat) {
    return (
      <div className="flex h-full items-center justify-center">
        Chat not found
      </div>
    )
  }

  const messages =
    currentChat?.id === params.id ? currentChat.messages : chat.messages

  return (
    <div className="flex h-full flex-col min-h-0">
      <div ref={printRef} className="flex flex-col flex-1 min-h-0">
        <TopBar onExport={handlePrint} />
        <div className="flex-1 overflow-hidden">
          <ChatList messages={messages} />
        </div>
      </div>
      <Composer />
    </div>
  )
}
