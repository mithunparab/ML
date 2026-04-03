'use client'

import { useState, useRef, useCallback, useEffect } from 'react'
import { segmentImage } from '@/lib/actions'

interface BBox {
  x1: number
  y1: number
  x2: number
  y2: number
  label: 1 | 0 // 1 = positive (include), 0 = negative (exclude)
}

export default function Home() {
  const [imageSrc, setImageSrc] = useState<string | null>(null)
  const [imageBase64, setImageBase64] = useState<string | null>(null)
  const [boxes, setBoxes] = useState<BBox[]>([])
  const [drawing, setDrawing] = useState(false)
  const [currentBox, setCurrentBox] = useState<{ x1: number; y1: number; x2: number; y2: number } | null>(null)
  const [labelMode, setLabelMode] = useState<1 | 0>(1)
  const [prompt, setPrompt] = useState('')
  const [confidence, setConfidence] = useState(0.45)
  const [maskSrc, setMaskSrc] = useState<string | null>(null)
  const [croppedSrc, setCroppedSrc] = useState<string | null>(null)
  const [loading, setLoading] = useState(false)
  const [error, setError] = useState<string | null>(null)
  const [showMaskOverlay, setShowMaskOverlay] = useState(true)
  const [naturalSize, setNaturalSize] = useState<{ w: number; h: number } | null>(null)

  const canvasRef = useRef<HTMLCanvasElement>(null)
  const imgRef = useRef<HTMLImageElement | null>(null)

  const handleImageUpload = (e: React.ChangeEvent<HTMLInputElement>) => {
    const file = e.target.files?.[0]
    if (!file) return
    setBoxes([])
    setMaskSrc(null)
    setCroppedSrc(null)
    setError(null)

    const reader = new FileReader()
    reader.onload = () => {
      const dataUrl = reader.result as string
      setImageSrc(dataUrl)
      setImageBase64(dataUrl.split(',')[1])
    }
    reader.readAsDataURL(file)
  }

  const drawCanvas = useCallback(() => {
    const canvas = canvasRef.current
    const img = imgRef.current
    if (!canvas || !img) return

    const ctx = canvas.getContext('2d')!
    canvas.width = img.clientWidth
    canvas.height = img.clientHeight

    ctx.clearRect(0, 0, canvas.width, canvas.height)

    // Draw existing boxes
    for (const box of boxes) {
      const sx = canvas.width / (naturalSize?.w || 1)
      const sy = canvas.height / (naturalSize?.h || 1)
      ctx.strokeStyle = box.label === 1 ? '#22c55e' : '#ef4444'
      ctx.lineWidth = 2
      ctx.setLineDash([6, 3])
      ctx.strokeRect(box.x1 * sx, box.y1 * sy, (box.x2 - box.x1) * sx, (box.y2 - box.y1) * sy)
      ctx.setLineDash([])
      // Label tag
      ctx.fillStyle = box.label === 1 ? '#22c55e' : '#ef4444'
      ctx.font = '12px sans-serif'
      ctx.fillText(box.label === 1 ? '+' : '-', box.x1 * sx + 4, box.y1 * sy + 14)
    }

    // Draw current box being drawn
    if (currentBox) {
      ctx.strokeStyle = labelMode === 1 ? '#22c55e' : '#ef4444'
      ctx.lineWidth = 2
      ctx.setLineDash([4, 4])
      ctx.strokeRect(currentBox.x1, currentBox.y1, currentBox.x2 - currentBox.x1, currentBox.y2 - currentBox.y1)
      ctx.setLineDash([])
    }
  }, [boxes, currentBox, labelMode, naturalSize])

  useEffect(() => {
    drawCanvas()
  }, [drawCanvas])

  const toNatural = (clientX: number, clientY: number) => {
    const canvas = canvasRef.current!
    const rect = canvas.getBoundingClientRect()
    const cx = clientX - rect.left
    const cy = clientY - rect.top
    const sx = (naturalSize?.w || 1) / canvas.width
    const sy = (naturalSize?.h || 1) / canvas.height
    return { nx: cx * sx, ny: cy * sy, cx, cy }
  }

  const handleMouseDown = (e: React.MouseEvent) => {
    const { cx, cy } = toNatural(e.clientX, e.clientY)
    setDrawing(true)
    setCurrentBox({ x1: cx, y1: cy, x2: cx, y2: cy })
  }

  const handleMouseMove = (e: React.MouseEvent) => {
    if (!drawing || !currentBox) return
    const canvas = canvasRef.current!
    const rect = canvas.getBoundingClientRect()
    const cx = e.clientX - rect.left
    const cy = e.clientY - rect.top
    setCurrentBox(prev => prev ? { ...prev, x2: cx, y2: cy } : null)
  }

  const handleMouseUp = () => {
    if (!drawing || !currentBox) return
    setDrawing(false)

    const canvas = canvasRef.current!
    const sx = (naturalSize?.w || 1) / canvas.width
    const sy = (naturalSize?.h || 1) / canvas.height

    const x1 = Math.round(Math.min(currentBox.x1, currentBox.x2) * sx)
    const y1 = Math.round(Math.min(currentBox.y1, currentBox.y2) * sy)
    const x2 = Math.round(Math.max(currentBox.x1, currentBox.x2) * sx)
    const y2 = Math.round(Math.max(currentBox.y1, currentBox.y2) * sy)

    if (Math.abs(x2 - x1) > 5 && Math.abs(y2 - y1) > 5) {
      setBoxes(prev => [...prev, { x1, y1, x2, y2, label: labelMode }])
    }
    setCurrentBox(null)
  }

  const removeBox = (index: number) => {
    setBoxes(prev => prev.filter((_, i) => i !== index))
  }

  const handleSegment = async () => {
    if (!imageBase64) return
    if (!boxes.length && !prompt.trim()) {
      setError('Add at least one bbox or text prompt')
      return
    }

    setLoading(true)
    setError(null)
    setMaskSrc(null)
    setCroppedSrc(null)

    try {
      const data = await segmentImage({
        image_data: imageBase64,
        prompt: prompt.trim() || undefined,
        input_boxes: boxes.length ? boxes.map(b => [b.x1, b.y1, b.x2, b.y2]) : undefined,
        input_labels: boxes.length ? boxes.map(b => b.label) : undefined,
        confidence_threshold: confidence,
      })

      if (data.error) {
        setError(data.error)
      } else if (data.message === 'No objects found.') {
        setError('No objects found. Try adjusting boxes or prompt.')
      } else {
        if (data.output_mask_base64) {
          setMaskSrc(`data:image/png;base64,${data.output_mask_base64}`)
        }
        if (data.output_image_base64) {
          setCroppedSrc(`data:image/png;base64,${data.output_image_base64}`)
        }
      }
    } catch (err) {
      setError(`Request failed: ${err}`)
    } finally {
      setLoading(false)
    }
  }

  const handleClear = () => {
    setBoxes([])
    setMaskSrc(null)
    setCroppedSrc(null)
    setError(null)
  }

  return (
    <div className="space-y-4">
      {/* Controls */}
      <div className="flex flex-wrap items-center gap-3">
        <input
          type="file"
          accept="image/*"
          onChange={handleImageUpload}
          className="text-sm"
        />
        <input
          type="text"
          placeholder="Text prompt (optional): sofa, chair..."
          value={prompt}
          onChange={e => setPrompt(e.target.value)}
          className="border rounded px-3 py-1.5 text-sm w-72"
        />
        <div className="flex items-center gap-2">
          <span className="text-sm font-medium">Box mode:</span>
          <button
            onClick={() => setLabelMode(1)}
            className={`px-3 py-1 rounded text-sm font-medium ${
              labelMode === 1 ? 'bg-green-600 text-white' : 'bg-gray-200 text-gray-700'
            }`}
          >
            + Include
          </button>
          <button
            onClick={() => setLabelMode(0)}
            className={`px-3 py-1 rounded text-sm font-medium ${
              labelMode === 0 ? 'bg-red-600 text-white' : 'bg-gray-200 text-gray-700'
            }`}
          >
            - Exclude
          </button>
        </div>
        <div className="flex items-center gap-2">
          <span className="text-sm font-medium">Confidence:</span>
          <input
            type="range"
            min="0.05"
            max="0.95"
            step="0.05"
            value={confidence}
            onChange={e => setConfidence(parseFloat(e.target.value))}
            className="w-24"
          />
          <span className="text-xs font-mono w-8">{confidence.toFixed(2)}</span>
        </div>
        <button
          onClick={handleSegment}
          disabled={loading || !imageSrc}
          className="bg-blue-600 text-white px-4 py-1.5 rounded text-sm font-medium disabled:opacity-50"
        >
          {loading ? 'Segmenting...' : 'Segment'}
        </button>
        <button
          onClick={handleClear}
          className="bg-gray-300 text-gray-800 px-4 py-1.5 rounded text-sm font-medium"
        >
          Clear
        </button>
      </div>

      {error && <div className="text-red-600 text-sm">{error}</div>}

      {/* Box list */}
      {boxes.length > 0 && (
        <div className="flex flex-wrap gap-2">
          {boxes.map((box, i) => (
            <span
              key={i}
              className={`inline-flex items-center gap-1 px-2 py-0.5 rounded text-xs font-mono ${
                box.label === 1 ? 'bg-green-100 text-green-800' : 'bg-red-100 text-red-800'
              }`}
            >
              {box.label === 1 ? '+' : '-'} [{box.x1},{box.y1},{box.x2},{box.y2}]
              <button onClick={() => removeBox(i)} className="ml-1 hover:text-black">&times;</button>
            </span>
          ))}
        </div>
      )}

      {/* Image + Canvas */}
      {imageSrc && (
        <div className="grid grid-cols-1 lg:grid-cols-2 gap-4">
          <div>
            <h3 className="text-sm font-medium mb-1">Input (draw boxes here)</h3>
            <div className="relative inline-block">
              {/* eslint-disable-next-line @next/next/no-img-element */}
              <img
                ref={imgRef}
                src={imageSrc}
                alt="Input"
                className="max-w-full"
                onLoad={e => {
                  const el = e.currentTarget
                  setNaturalSize({ w: el.naturalWidth, h: el.naturalHeight })
                }}
              />
              {maskSrc && showMaskOverlay && (
                /* eslint-disable-next-line @next/next/no-img-element */
                <img
                  src={maskSrc}
                  alt="Mask overlay"
                  className="absolute inset-0 w-full h-full pointer-events-none"
                  style={{ mixBlendMode: 'multiply', opacity: 0.4 }}
                />
              )}
              <canvas
                ref={canvasRef}
                className="absolute inset-0 w-full h-full cursor-crosshair"
                onMouseDown={handleMouseDown}
                onMouseMove={handleMouseMove}
                onMouseUp={handleMouseUp}
                onMouseLeave={() => { if (drawing) handleMouseUp() }}
              />
            </div>
          </div>

          {/* Results */}
          <div className="space-y-4">
            {maskSrc && (
              <div>
                <div className="flex items-center gap-3 mb-1">
                  <h3 className="text-sm font-medium">Combined Mask</h3>
                  <label className="text-xs flex items-center gap-1">
                    <input
                      type="checkbox"
                      checked={showMaskOverlay}
                      onChange={e => setShowMaskOverlay(e.target.checked)}
                    />
                    Overlay on input
                  </label>
                </div>
                {/* eslint-disable-next-line @next/next/no-img-element */}
                <img src={maskSrc} alt="Mask" className="max-w-full border" />
              </div>
            )}
            {croppedSrc && (
              <div>
                <h3 className="text-sm font-medium mb-1">Cropped Result</h3>
                {/* eslint-disable-next-line @next/next/no-img-element */}
                <img src={croppedSrc} alt="Cropped" className="max-w-full" style={{ background: 'repeating-conic-gradient(#ddd 0% 25%, white 0% 50%) 50% / 20px 20px' }} />
              </div>
            )}
          </div>
        </div>
      )}
    </div>
  )
}
