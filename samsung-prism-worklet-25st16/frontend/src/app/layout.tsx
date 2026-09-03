import type { Metadata, Viewport } from "next";
import Link from "next/link";
import "./globals.css";

export const metadata: Metadata = {
  title: "Samsung Research (Prism) Worklet:25ST16",
  description:
    "U-Net++ segmentation, sparse feature encoding and EfficientNet classification for plant leaf disease diagnosis.",
};

export const viewport: Viewport = {
  width: "device-width",
  initialScale: 1,
  themeColor: "#059669",
};

const NAV = [
  { href: "/", label: "Overview" },
  { href: "/analyze", label: "Analyze" },
  { href: "/research", label: "Research" },
];

export default function RootLayout({ children }: { children: React.ReactNode }) {
  return (
    <html lang="en">
      <body className="min-h-screen antialiased">
        <header className="sticky top-0 z-40 border-b border-line glass no-print">
          <nav className="mx-auto flex max-w-6xl items-center justify-between px-4 py-3 sm:px-6">
            <Link href="/" className="flex items-center gap-2">
              <span className="grid h-8 w-8 place-items-center rounded-lg bg-accent text-[10px] font-bold leading-none text-white">
                25
              </span>
              <span className="text-sm font-semibold tracking-tight">
                Samsung Research (Prism){" "}
                <span className="whitespace-nowrap">Worklet:25ST16</span>
                <span className="ml-2 hidden text-xs font-normal text-muted md:inline">
                  leaf disease diagnosis
                </span>
              </span>
            </Link>
            <div className="flex items-center gap-1 text-sm">
              {NAV.map((item) => (
                <Link
                  key={item.href}
                  href={item.href}
                  className="rounded-md px-3 py-1.5 text-muted transition hover:bg-accent-soft hover:text-accent"
                >
                  {item.label}
                </Link>
              ))}
            </div>
          </nav>
        </header>
        <main className="mx-auto max-w-6xl px-4 pb-20 pt-8 sm:px-6">{children}</main>
        <footer className="border-t border-line py-6 text-center text-xs text-muted no-print">
          Samsung Research (Prism) Worklet:25ST16 · U-Net++ · Sparse Stacked Encoder · EfficientNet
        </footer>
      </body>
    </html>
  );
}
