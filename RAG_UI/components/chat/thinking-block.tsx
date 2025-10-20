"use client"

import { Card, CardContent } from "@/components/ui/card"
import { ScrollArea } from "@/components/ui/scroll-area"
import { Loader2 } from "lucide-react"

export function ThinkingBlock({ text }: { text: string }) {
    return (
        <Card className="border-muted/50 bg-muted/40">
            <CardContent className="p-3">
                <div className="mb-2 flex items-center gap-2 text-sm text-muted-foreground">
                    <Loader2 className="h-4 w-4 animate-spin" aria-hidden />
                    <span>Thinking…</span>
                </div>
                <ScrollArea className="max-h-40">
          <pre className="whitespace-pre-wrap break-words text-xs leading-relaxed text-muted-foreground/90">
            {text || "…"}
          </pre>
                </ScrollArea>
            </CardContent>
        </Card>
    )
}
