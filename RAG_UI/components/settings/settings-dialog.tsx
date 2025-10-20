"use client"

import { useChat } from "@/components/providers/chat-provider"
import { setLocalStorage } from "@/lib/storage"
import {
  DialogContent,
  DialogHeader,
  DialogTitle,
} from "@/components/ui/dialog"
import {
  Select,
  SelectTrigger,
  SelectValue,
  SelectContent,
  SelectItem,
} from "@/components/ui/select"
import { Slider } from "@/components/ui/slider"
import { Switch } from "@/components/ui/switch"
import { Label } from "@/components/ui/label"

export function SettingsDialog() {
  const { settings, setSettings } = useChat()

  const updateSettings = (next: Record<string, unknown>) => {
    setSettings(next)
    setLocalStorage("settings", next)
  }

  const handleModelChange = (value: string) => {
    const next = { ...settings, model: value }
    updateSettings(next)
  }

  const handleTemperatureChange = (value: number[]) => {
    const temp = value[0]
    const next = { ...settings, temperature: temp }
    updateSettings(next)
  }

  const handleHistoryChange = (checked: boolean) => {
    const next = { ...settings, history: checked }
    updateSettings(next)
  }

  const modelValue = (settings.model as string) ?? "gpt-4o-mini"
  const temperatureValue =
    typeof settings.temperature === "number" ? settings.temperature : 0.5
  const historyValue = (settings.history as boolean) ?? true

  return (
    <DialogContent className="max-w-md">
      <DialogHeader>
        <DialogTitle>Settings</DialogTitle>
      </DialogHeader>
      <div className="grid gap-4 py-2">
        <div className="grid gap-2">
          <Label htmlFor="model">Model</Label>
          <Select value={modelValue} onValueChange={handleModelChange}>
            <SelectTrigger id="model">
              <SelectValue placeholder="Select a model" />
            </SelectTrigger>
            <SelectContent>
              <SelectItem value="gpt-4o-mini">GPT-4o Mini</SelectItem>
              <SelectItem value="gpt-4o">GPT-4o</SelectItem>
            </SelectContent>
          </Select>
        </div>
        <div className="grid gap-2">
          <Label htmlFor="temperature">
            Temperature: {temperatureValue.toFixed(2)}
          </Label>
          <Slider
            id="temperature"
            min={0}
            max={1}
            step={0.01}
            value={[temperatureValue]}
            onValueChange={handleTemperatureChange}
          />
        </div>
        <div className="flex items-center justify-between">
          <Label htmlFor="history">Use history</Label>
          <Switch
            id="history"
            checked={historyValue}
            onCheckedChange={handleHistoryChange}
          />
        </div>
      </div>
    </DialogContent>
  )
}

export default SettingsDialog

