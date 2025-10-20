export async function* streamText(
  text: string,
  {
    delay = 20,
    chunkSize = 1,
  }: { delay?: number; chunkSize?: number } = {}
): AsyncGenerator<string> {
  let i = 0
  while (i < text.length) {
    const chunk = text.slice(i, i + chunkSize)
    await new Promise((res) => setTimeout(res, delay))
    yield chunk
    i += chunkSize
  }
}

export default streamText
