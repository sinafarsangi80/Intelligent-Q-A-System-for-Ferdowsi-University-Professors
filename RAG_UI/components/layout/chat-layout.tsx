"use client"

import React, { ReactNode, useState } from "react"
import { IconMenu2 } from "@tabler/icons-react"
import {
  ResizableHandle,
  ResizablePanel,
  ResizablePanelGroup,
} from "@/components/ui/resizable"
import { Button } from "@/components/ui/button"
import { useIsMobile } from "@/hooks/use-mobile"

export function ChatLayout({
  sidebar,
  children,
}: {
  sidebar: ReactNode
  children: ReactNode
}) {
  const isMobile = useIsMobile()
  const [open, setOpen] = useState(false)

  if (isMobile) {
    const sidebarWithProps = React.isValidElement(sidebar)
      ? React.cloneElement(
          sidebar as React.ReactElement<{
            open?: boolean
            onOpenChange?: (open: boolean) => void
          }>,
          {
            open,
            onOpenChange: setOpen,
          }
        )
      : sidebar

    return (
      <div className="flex h-screen flex-col bg-background text-foreground">
        <Button
          variant="ghost"
          size="icon"
          className="m-2 md:hidden print:hidden"
          onClick={() => setOpen(true)}
          aria-label="Open sidebar"
        >
          <IconMenu2 className="size-5" aria-hidden="true" />
        </Button>
        {sidebarWithProps}
        <div className="flex flex-1 flex-col min-h-0">{children}</div>
      </div>
    )
  }

  return (
    <ResizablePanelGroup
      direction="horizontal"
      className="!h-screen w-full bg-background text-foreground"
    >
      <ResizablePanel defaultSize={25} minSize={18} maxSize={30} className="h-full print:hidden">
        {sidebar}
      </ResizablePanel>
      <ResizableHandle className="print:hidden" />
      <ResizablePanel className="flex h-full flex-col min-h-0">{children}</ResizablePanel>
    </ResizablePanelGroup>
  )
}

export default ChatLayout
