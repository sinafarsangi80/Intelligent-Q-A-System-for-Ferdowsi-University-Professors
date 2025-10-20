"use client"

import {
  PanelGroup,
  Panel,
  PanelResizeHandle,
  type PanelGroupProps,
  type PanelProps,
  type PanelResizeHandleProps,
} from "react-resizable-panels"

import { cn } from "@/lib/utils"

export function ResizablePanelGroup({ className, ...props }: PanelGroupProps) {
  return <PanelGroup className={cn(className)} {...props} />
}

export function ResizablePanel(props: PanelProps) {
  return <Panel {...props} />
}

export function ResizableHandle({ className, ...props }: PanelResizeHandleProps) {
  return (
    <PanelResizeHandle
      className={cn(
        "flex w-2 items-center justify-center bg-border hover:bg-border/80 data-[panel-group-direction=vertical]:h-2 data-[panel-group-direction=vertical]:w-full",
        className
      )}
      {...props}
    />
  )
}
