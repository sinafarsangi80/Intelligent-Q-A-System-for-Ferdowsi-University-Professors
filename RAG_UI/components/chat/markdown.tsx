"use client"

import ReactMarkdown from "react-markdown"
import remarkGfm from "remark-gfm"
import { cn } from "@/lib/utils"
import CopyButton from "../copy-button"
import type { ReactNode, HTMLAttributes } from "react"

export function Markdown({ children }: { children: string }) {
  return (
    <ReactMarkdown
      remarkPlugins={[remarkGfm]}
      components={{
        code({ inline, className, children, ...props }: HTMLAttributes<HTMLElement> & {
          inline?: boolean
          children?: ReactNode
        }) {
          if (!inline) {
            return (
              <div className="relative">
                <pre
                  className={cn(
                    "overflow-x-auto rounded-md bg-muted p-4 text-sm",
                    className
                  )}
                >
                  <code {...props}>{children}</code>
                </pre>
                <CopyButton
                  value={String(children).replace(/\n$/, "")}
                  className="absolute top-2 right-2"
                />
              </div>
            )
          }
          return (
            <code
              className={cn(
                "rounded bg-muted px-1 py-0.5 font-mono text-sm", className
              )}
              {...props}
            >
              {children}
            </code>
          )
        },
      }}
    >
      {children}
    </ReactMarkdown>
  )
}

export default Markdown
