"use client"

import { useEffect } from "react"
import {
  Command,
  CommandEmpty,
  CommandGroup,
  CommandInput,
  CommandItem,
  CommandList,
  CommandSeparator,
} from "@/components/ui/command"
import { Dialog, DialogContent } from "@/components/ui/dialog"
import { useChat } from "@/components/providers/chat-provider"
import { triggerNewChat } from "@/lib/chat-store"
import { useRouter } from "next/navigation"
import { useStream } from "@/components/providers/stream-provider"

interface CommandPaletteProps {
  open: boolean
  onOpenChange: (open: boolean) => void
}

export function CommandPalette({ open, onOpenChange }: CommandPaletteProps) {
  const { chats, currentChat, setChats, setCurrentChat, updateCurrentChat } =
    useChat()
  const router = useRouter()
  const { streamingChatId } = useStream()

  useEffect(() => {
    const onKeyDown = (e: KeyboardEvent) => {
      if (e.key === "k" && (e.metaKey || e.ctrlKey)) {
        e.preventDefault()
        onOpenChange(!open)
      }
    }
    window.addEventListener("keydown", onKeyDown)
    return () => window.removeEventListener("keydown", onKeyDown)
  }, [open, onOpenChange])

  const handleDelete = () => {
    if (!currentChat) return
    const idx = chats.findIndex((c) => c.id === currentChat.id)
    const nextChats = chats.filter((c) => c.id !== currentChat.id)
    setChats(nextChats)
    setCurrentChat(nextChats[idx] ?? nextChats[idx - 1] ?? null)
  }

  const handleRename = () => {
    if (!currentChat) return
    const name = prompt("Enter a new name", currentChat.title)
    if (name) updateCurrentChat({ title: name })
  }

  return (
    <Dialog open={open} onOpenChange={onOpenChange}>
      <DialogContent className="p-0 overflow-hidden">
        <Command>
          <CommandInput placeholder="Type a command or search..." />
          <CommandList>
            <CommandEmpty>No results found.</CommandEmpty>
            <CommandGroup heading="Conversations">
              {chats.map((chat) => (
                <CommandItem
                  key={chat.id}
                  onSelect={() => {
                    setCurrentChat(chat)
                    router.push(`/chat/${chat.id}`)
                    onOpenChange(false)
                  }}
                >
                  {chat.title}
                </CommandItem>
              ))}
            </CommandGroup>
            <CommandSeparator />
            <CommandGroup heading="Actions">
              <CommandItem
                disabled={!!streamingChatId}
                aria-disabled={!!streamingChatId}
                onSelect={() => {
                  if (streamingChatId) return
                  setCurrentChat(null)
                  triggerNewChat()
                  router.push(`/chat`)
                  onOpenChange(false)
                }}
              >
                New chat
              </CommandItem>
              <CommandItem
                onSelect={() => {
                  handleRename()
                  onOpenChange(false)
                }}
              >
                Rename
              </CommandItem>
              <CommandItem
                onSelect={() => {
                  handleDelete()
                  onOpenChange(false)
                }}
              >
                Delete
              </CommandItem>
            </CommandGroup>
          </CommandList>
        </Command>
      </DialogContent>
    </Dialog>
  )
}

export default CommandPalette
