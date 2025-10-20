"use client"

import { useState, type ComponentProps } from "react"
import { Button } from "@/components/ui/button"
import { cn } from "@/lib/utils"
import { IconCheck, IconCopy } from "@tabler/icons-react"
import { toast } from "sonner"

interface CopyButtonProps extends ComponentProps<typeof Button> {
  value: string
  successMessage?: string
  errorMessage?: string
}

export function CopyButton({
  value,
  successMessage = "Copied to clipboard",
  errorMessage = "Failed to copy",
  className,
  ...props
}: CopyButtonProps) {
  const [copied, setCopied] = useState(false)

  const onCopy = async () => {
    try {
      await navigator.clipboard.writeText(value)
      setCopied(true)
      toast.success(successMessage)
      setTimeout(() => setCopied(false), 2000)
    } catch {
      toast.error(errorMessage)
    }
  }

  return (
    <Button
      variant="ghost"
      size="icon"
      className={cn("h-6 w-6", className)}
      onClick={onCopy}
      {...props}
    >
      {copied ? (
        <IconCheck className="size-4" />
      ) : (
        <IconCopy className="size-4" />
      )}
      <span className="sr-only">Copy</span>
    </Button>
  )
}

export default CopyButton
