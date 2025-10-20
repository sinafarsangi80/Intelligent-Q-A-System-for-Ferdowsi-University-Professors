'use client'

import { Chat, useChat } from '@/components/providers/chat-provider'
import { cn } from '@/lib/utils'
import { useRouter } from 'next/navigation'

interface ConversationItemProps {
  chat: Chat
}

export function ConversationItem({ chat }: ConversationItemProps) {
  const { currentChat, setCurrentChat } = useChat()
  const router = useRouter()
  const isActive = currentChat?.id === chat.id
  const title = chat.title?.trim() || 'New Chat'

  return (
    <button
      onClick={() => {
        setCurrentChat(chat)
        router.push(`/chat/${chat.id}`)
      }}
      className={cn(
        'w-full truncate rounded-md px-3 py-2 pr-8 text-left text-sm hover:bg-accent',
        isActive && 'bg-accent text-accent-foreground'
      )}
    >
      {title}
    </button>
  )
}

export default ConversationItem
