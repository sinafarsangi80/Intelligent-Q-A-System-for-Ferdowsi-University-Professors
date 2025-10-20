"use client"

import { Avatar, AvatarFallback, AvatarImage } from "@/components/ui/avatar"
import { Button } from "@/components/ui/button"
import {
  Tooltip,
  TooltipContent,
  TooltipTrigger,
} from "@/components/ui/tooltip"
import { useChat } from "@/components/providers/chat-provider"
import CopyButton from "../copy-button"
import Markdown from "./markdown"
import { cn } from "@/lib/utils"
import { IconTrash } from "@tabler/icons-react"
import { toast } from "sonner"
import type { SourceDoc } from "@/components/chat/sources-bar"

export interface ChatMessageData {
  id: string
  role: "user" | "assistant"
  name: string
  content: string
  timestamp: string
  avatar?: string
  sources?: SourceDoc[]
}

export function ChatMessage({ message }: { message: ChatMessageData }) {
  const isUser = message.role === "user"
  const { currentChat, updateCurrentChat } = useChat()

  const handleDelete = () => {
    if (!currentChat) return
    const messages = currentChat.messages.filter((m) => m.id !== message.id)
    updateCurrentChat({ messages })
    toast.success("Message deleted")
  }

  return (
    <div
      className={cn(
        "group flex items-start gap-4",
        isUser && "flex-row-reverse text-right",
      )}
    >
      <Avatar>
        {message.avatar ? (
          <AvatarImage src={message.avatar} alt={message.name} />
        ) : (
          <AvatarFallback>
            {message.name
              .split(" ")
              .map((n) => n[0])
              .join("")}
          </AvatarFallback>
        )}
      </Avatar>
      <div className="grid gap-1">
        <div
          className={cn(
            "flex items-center gap-2 text-xs text-muted-foreground",
            isUser && "flex-row-reverse",
          )}
        >
          <span className="font-medium text-foreground">{message.name}</span>
          <span className="text-muted-foreground">
            {new Date(message.timestamp).toLocaleTimeString()}
          </span>
          <div
            className={cn(
              "flex items-center gap-1 opacity-0 group-hover:opacity-100",
              isUser && "flex-row-reverse",
            )}
          >
            <Tooltip>
              <TooltipTrigger asChild>
                <CopyButton value={message.content} />
              </TooltipTrigger>
              <TooltipContent>Copy</TooltipContent>
            </Tooltip>
            <Tooltip>
              <TooltipTrigger asChild>
                <Button
                  variant="ghost"
                  size="icon"
                  className="h-6 w-6"
                  onClick={handleDelete}
                >
                  <IconTrash className="size-4" />
                  <span className="sr-only">Delete</span>
                </Button>
              </TooltipTrigger>
              <TooltipContent>Delete</TooltipContent>
            </Tooltip>
          </div>
        </div>
        <div
          className={cn(
            "rounded-md px-3 py-2 text-sm",
            isUser
              ? "bg-primary text-primary-foreground"
              : "bg-muted text-foreground",
          )}
        >
          {isUser ? message.content : <Markdown>{message.content}</Markdown>}
        </div>
      </div>
    </div>
  )
}

export default ChatMessage
