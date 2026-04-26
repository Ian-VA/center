import type { Metadata } from "next";
import 'leaflet/dist/leaflet.css';
import { Geist, Geist_Mono } from "next/font/google";
import "./globals.css";

const geistSans = Geist({
  variable: "--font-geist-sans",
  subsets: ["latin"],
  display: "swap",
});

const geistMono = Geist_Mono({
  variable: "--font-geist-mono",
  subsets: ["latin"],
  display: "swap",
});

export const metadata: Metadata = {
  title: "Center — Data Center Pollution & Permitting Analytics",
  description:
    "Center is an analytics platform for data center pollution modeling and permitting compliance.",
};

export default function RootLayout({
  children,
}: Readonly<{
  children: React.ReactNode;
}>) {
  return (
    <html
      lang="en"
      className={`${geistSans.variable} ${geistMono.variable} h-full antialiased`}
      suppressHydrationWarning
    >
      <body
        className="min-h-full flex flex-col bg-background text-foreground"
        suppressHydrationWarning
      >
        <header
          className="app-header sticky top-0 z-30 flex items-center justify-between gap-4 border-b border-black/30 text-header-foreground px-6 py-3 shadow-sm"
          suppressHydrationWarning
        >
          <div className="flex items-center gap-3">
            <span className="inline-flex h-8 w-8 items-center justify-center rounded-md bg-gradient-to-br from-indigo-500 to-sky-400 text-white shadow ring-1 ring-white/15">
              <svg
                xmlns="http://www.w3.org/2000/svg"
                viewBox="0 0 24 24"
                fill="none"
                stroke="currentColor"
                strokeWidth="2"
                strokeLinecap="round"
                strokeLinejoin="round"
                className="h-4 w-4"
                aria-hidden="true"
              >
                <circle cx="12" cy="12" r="9" />
                <circle cx="12" cy="12" r="3" />
              </svg>
            </span>
            <div className="flex items-baseline gap-2">
              <span className="text-base font-semibold tracking-tight">
                Center
              </span>
            </div>
          </div>
          <div className="hidden items-center gap-2 text-xs font-medium text-header-foreground/60 md:flex">
            <span className="inline-flex h-2 w-2 rounded-full bg-emerald-400" />
            <span>Live preview</span>
          </div>
        </header>
        <main
          className="flex flex-1 min-h-0 flex-col"
          suppressHydrationWarning
        >
          {children}
        </main>
      </body>
    </html>
  );
}
