import type { Metadata } from 'next'
import './globals.css'

export const metadata: Metadata = {
  title: 'SAM3 Segmentation',
  description: 'Draw bounding boxes to segment objects with SAM3',
  robots: 'noindex, nofollow',
}

export default function RootLayout({
  children,
}: {
  children: React.ReactNode
}) {
  return (
    <html lang="en">
      <body>
        <header className="bg-gray-900 p-4">
          <nav className="max-w-7xl mx-auto flex items-center justify-between">
            <span className="text-lg font-semibold text-white">SAM3 Segmentation</span>
          </nav>
        </header>
        <main className="max-w-7xl mx-auto p-4">{children}</main>
      </body>
    </html>
  )
}
