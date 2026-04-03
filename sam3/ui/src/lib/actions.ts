'use server'

const RUNPOD_API_KEY = process.env.RUNPOD_API_KEY!
const RUNPOD_ENDPOINT = process.env.RUNPOD_ENDPOINT || 'https://api.runpod.ai/v2/hfxkmoumn078y7/runsync'

export async function segmentImage(params: {
  image_data: string
  prompt?: string
  input_boxes?: number[][]
  input_labels?: number[]
  confidence_threshold?: number
}): Promise<{
  output_image_base64?: string | null
  output_mask_base64?: string | null
  message: string
  inference_time?: string
  error?: string
}> {
  const payload: Record<string, unknown> = { image_data: params.image_data }
  if (params.prompt) payload.prompt = params.prompt
  if (params.input_boxes?.length) payload.input_boxes = params.input_boxes
  if (params.input_labels?.length) payload.input_labels = params.input_labels
  if (params.confidence_threshold != null) payload.confidence_threshold = params.confidence_threshold

  const res = await fetch(RUNPOD_ENDPOINT, {
    method: 'POST',
    headers: {
      'Content-Type': 'application/json',
      'Authorization': `Bearer ${RUNPOD_API_KEY}`,
    },
    body: JSON.stringify({ input: payload }),
  })

  const data = await res.json()
  return data.output ?? { message: 'Error', error: data.error ?? 'Unknown error' }
}
