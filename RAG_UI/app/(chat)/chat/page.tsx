"use client"

import { useRef } from "react"
import { ChatList } from "@/components/chat-transcript"
import { Composer } from "@/components/chat/composer"
import { TopBar } from "@/components/chat/top-bar"
import { useChat } from "@/components/providers/chat-provider"
import { useReactToPrint } from "react-to-print"

export default function ChatPage() {
  const { currentChat } = useChat()
  const messages = currentChat?.messages ?? []

  const printRef = useRef<HTMLDivElement>(null)
  const handlePrint = useReactToPrint({
    contentRef: printRef,
  })

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
