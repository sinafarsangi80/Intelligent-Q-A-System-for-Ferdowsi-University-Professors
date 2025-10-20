"use client"

import { Input } from "@/components/ui/input"
import { Button } from "@/components/ui/button"
import {
  DropdownMenu,
  DropdownMenuContent,
  DropdownMenuItem,
  DropdownMenuTrigger,
} from "@/components/ui/dropdown-menu"
import { Dialog, DialogTrigger } from "@/components/ui/dialog"
import SettingsDialog from "@/components/settings/settings-dialog"
import { MoreVertical } from "lucide-react"
import { toast } from "sonner"
import { useChat } from "@/components/providers/chat-provider"

interface TopBarProps {
  onExport: () => void
}
export function TopBar({ onExport }: TopBarProps) {
  const { currentChat, updateCurrentChat, clearCurrentChat } = useChat()

  return (
    <div className="flex h-14 items-center gap-2 border-b px-4">
      <Input
        value={currentChat?.title ?? ""}
        onChange={(e) => updateCurrentChat({ title: e.target.value })}
        placeholder="Untitled chat"
        className="h-9 w-full max-w-sm border-0 bg-transparent px-0 font-medium focus-visible:outline-none focus-visible:ring-0"
      />
      <div className="ml-auto flex items-center gap-2">
        <DropdownMenu>
          <DropdownMenuTrigger asChild>
            <Button variant="ghost" size="icon">
              <MoreVertical className="size-4" />
              <span className="sr-only">Open menu</span>
            </Button>
          </DropdownMenuTrigger>
          <DropdownMenuContent align="end">
            <DropdownMenuItem onClick={onExport}>Export chat</DropdownMenuItem>
          </DropdownMenuContent>
        </DropdownMenu>
      </div>
    </div>
  )
}

export default TopBar

