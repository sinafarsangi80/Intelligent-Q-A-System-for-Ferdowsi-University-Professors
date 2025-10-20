"use client"

import { useState } from "react"
import { Button } from "@/components/ui/button"
import {
    Collapsible,
    CollapsibleTrigger,
    CollapsibleContent,
} from "@/components/ui/collapsible"
import { IconChevronDown } from "@tabler/icons-react"

export type SourceDoc = {
    id: string
    title: string
    venue?: string | null
    article_link?: string | null
}

export function SourcesBar({ sources }: { sources: SourceDoc[] }) {
    const [open, setOpen] = useState(false)
    if (!sources?.length) return null

    return (
        <Collapsible open={open} onOpenChange={setOpen} className="mt-2">
            <CollapsibleTrigger className="group flex items-center text-xs font-medium text-muted-foreground transition-colors hover:text-foreground">
                منابع پیشنهادی
                <IconChevronDown className="ml-1 h-4 w-4 transition-transform group-data-[state=open]:rotate-180" />
            </CollapsibleTrigger>
            <CollapsibleContent className="grid gap-2 sm:grid-cols-2 overflow-hidden transition-all duration-300 data-[state=open]:mt-2 data-[state=open]:max-h-96 data-[state=open]:opacity-100 data-[state=closed]:max-h-0 data-[state=closed]:opacity-0">
                {sources.map((s) => {
                    const href = s.article_link ?? "#"
                    return (
                        <a
                            key={s.id}
                            href={href}
                            target="_blank"
                            rel="noopener noreferrer"
                            className="block"
                        >
                            <Button
                                variant="secondary"
                                size="sm"
                                className="h-auto w-full justify-start whitespace-normal text-left py-2"
                            >
                                <div className="flex flex-col">
                                    <span className="text-sm">{s.title}</span>
                                    {s.venue ? (
                                        <span className="text-xs text-muted-foreground">{s.venue}</span>
                                    ) : null}
                                </div>
                            </Button>
                        </a>
                    )
                })}
            </CollapsibleContent>
        </Collapsible>
    )
}
