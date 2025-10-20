"use client"

import { ReactNode, useState } from "react"
import { ChatLayout } from "@/components/layout/chat-layout"
import { ChatProvider } from "@/components/providers/chat-provider"
import { StreamProvider } from "@/components/providers/stream-provider"
import { Sidebar } from "@/components/sidebar/sidebar"
import { CommandPalette } from "@/components/command-palette"

export default function ChatGroupLayout({ children }: { children: ReactNode }) {
  const [open, setOpen] = useState(false)

  return (
    <StreamProvider>
      <ChatProvider>
        <CommandPalette open={open} onOpenChange={setOpen} />
        <ChatLayout sidebar={<Sidebar />}>{children}</ChatLayout>
      </ChatProvider>
    </StreamProvider>
  )
}
