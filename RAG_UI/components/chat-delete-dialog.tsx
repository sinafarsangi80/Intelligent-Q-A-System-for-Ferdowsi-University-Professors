'use client'

import { Chat, useChat } from '@/components/providers/chat-provider'
import { Button } from '@/components/ui/button'
import {
  AlertDialog,
  AlertDialogAction,
  AlertDialogCancel,
  AlertDialogContent,
  AlertDialogDescription,
  AlertDialogFooter,
  AlertDialogHeader,
  AlertDialogTitle,
  AlertDialogTrigger,
} from '@/components/ui/alert-dialog'
import { Trash2 } from 'lucide-react'
import { useRouter } from 'next/navigation'
import { toast } from 'sonner'

interface ChatDeleteDialogProps {
  chat: Chat
}

export function ChatDeleteDialog({ chat }: ChatDeleteDialogProps) {
  const { chats, setChats, currentChat, setCurrentChat } = useChat()
  const router = useRouter()

  const handleDelete = () => {
    setChats(chats.filter((c) => c.id !== chat.id))
    if (currentChat?.id === chat.id) {
      setCurrentChat(null)
      router.push('/chat')
    }
    toast.success('Chat deleted')
  }

  return (
    <AlertDialog>
      <AlertDialogTrigger asChild>
        <Button
          variant='ghost'
          size='icon'
          className='absolute right-1 top-1/2 -translate-y-1/2 opacity-0 group-hover:opacity-100'
        >
          <Trash2 className='size-4' />
          <span className='sr-only'>Delete chat</span>
        </Button>
      </AlertDialogTrigger>
      <AlertDialogContent>
        <AlertDialogHeader>
          <AlertDialogTitle>Delete chat?</AlertDialogTitle>
          <AlertDialogDescription>
            This action cannot be undone.
          </AlertDialogDescription>
        </AlertDialogHeader>
        <AlertDialogFooter>
          <AlertDialogCancel>Cancel</AlertDialogCancel>
          <AlertDialogAction onClick={handleDelete}>
            Delete
          </AlertDialogAction>
        </AlertDialogFooter>
      </AlertDialogContent>
    </AlertDialog>
  )
}

export default ChatDeleteDialog

