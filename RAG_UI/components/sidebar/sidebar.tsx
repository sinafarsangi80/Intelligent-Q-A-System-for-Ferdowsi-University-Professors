'use client'

import { useEffect, useState } from 'react'
import { IconPlus } from '@tabler/icons-react'
import { Chat, useChat } from '@/components/providers/chat-provider'
import { triggerNewChat } from '@/lib/chat-store'
import { useRouter } from 'next/navigation'
import ConversationItem from './conversation-item'
import { Button } from '@/components/ui/button'
import { ScrollArea } from '@/components/ui/scroll-area'
import { Sheet, SheetContent } from '@/components/ui/sheet'
import { Avatar, AvatarFallback, AvatarImage } from '@/components/ui/avatar'
import {
  DropdownMenu,
  DropdownMenuContent,
  DropdownMenuItem,
  DropdownMenuTrigger,
} from '@/components/ui/dropdown-menu'
import { Skeleton } from '@/components/ui/skeleton'
import { useIsMobile } from '@/hooks/use-mobile'
import ChatDeleteDialog from '@/components/chat-delete-dialog'
import { useStream } from '@/components/providers/stream-provider'

export function Sidebar({
  open,
  onOpenChange,
}: {
  open?: boolean
  onOpenChange?: (open: boolean) => void
}) {
  const { chats, setChats, setCurrentChat } = useChat()
  const { streamingChatId } = useStream()
  const router = useRouter()
  const isMobile = useIsMobile()
  const [loading, setLoading] = useState(true)

  useEffect(() => {
    setLoading(false)
  }, [chats])

  const content = (
    <aside
      id='sidebar'
      className='flex h-full min-h-0 flex-col overflow-hidden print:hidden'
    >
      <div className='flex items-center gap-2 p-2'>
        <Button
          onClick={() => {
            const chat: Chat = {
              id: crypto.randomUUID(),
              title: 'New Chat',
              messages: [],
            }
            setChats((prev) => [chat, ...prev])
            setCurrentChat(chat)
            triggerNewChat()
            router.push(`/chat/${chat.id}`)
          }}
          className='flex-1'
          variant='secondary'
          aria-label='Start a new chat'
          disabled={!!streamingChatId}
          aria-disabled={!!streamingChatId}
        >
          <IconPlus className='mr-2 size-4' aria-hidden='true' />
          New Chat
        </Button>
        <DropdownMenu>
          <DropdownMenuTrigger asChild>
            <Avatar className='size-8 cursor-pointer' aria-label='User menu'>
              <AvatarImage src='' alt='@user' />
              <AvatarFallback>AC</AvatarFallback>
            </Avatar>
          </DropdownMenuTrigger>
          <DropdownMenuContent align='end'>
            <DropdownMenuItem>Profile</DropdownMenuItem>
            <DropdownMenuItem>Logout</DropdownMenuItem>
          </DropdownMenuContent>
        </DropdownMenu>
      </div>
      <ScrollArea className='flex-1'>
        <ul className='space-y-1 p-2' role='list'>
          {loading
            ? Array.from({ length: 5 }).map((_, i) => (
                <li key={i}>
                  <Skeleton className='h-8 w-full' />
                </li>
              ))
            : chats.map((chat) => (
                <li key={chat.id} className='group relative'>
                  <ConversationItem chat={chat} />
                  <ChatDeleteDialog chat={chat} />
                </li>
              ))}
        </ul>
      </ScrollArea>
    </aside>
  )

  if (isMobile) {
    return (
      <Sheet open={open} onOpenChange={onOpenChange}>
        <SheetContent side='left' className='w-[260px] p-0'>
          {content}
        </SheetContent>
      </Sheet>
    )
  }

  return content
}

export default Sidebar
