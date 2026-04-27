import type { Metadata } from "next";
import { DM_Serif_Display, IBM_Plex_Sans, JetBrains_Mono } from "next/font/google";
import "./globals.css";

const heading = DM_Serif_Display({
  weight: "400",
  style: ["normal", "italic"],
  subsets: ["latin"],
  variable: "--font-heading",
  display: "swap",
});

const body = IBM_Plex_Sans({
  weight: ["300", "400", "500", "600", "700"],
  subsets: ["latin"],
  variable: "--font-body",
  display: "swap",
});

const mono = JetBrains_Mono({
  weight: ["400", "500", "600"],
  subsets: ["latin"],
  variable: "--font-mono",
  display: "swap",
});

// metadataBase makes relative og:image URLs resolve to absolute ones.
// LinkedIn/Twitter REQUIRE absolute URLs for preview images.
export const metadata: Metadata = {
  metadataBase: new URL("https://mosaic-r.netlify.app"),
  title: "Mosaic — Synthesize answers across your documents",
  description:
    "Mosaic assembles answers from pieces of your documents, with every source cited.",
  openGraph: {
    title: "Mosaic — Synthesize answers across your documents",
    description:
      "Mosaic assembles answers from pieces of your documents, with every source cited.",
    url: "https://mosaic-r.netlify.app",
    siteName: "Mosaic",
    images: [
      {
        url: "/og-image.png",
        width: 1200,
        height: 630,
        alt: "Mosaic — multi-document RAG assistant",
      },
    ],
    type: "website",
  },
  twitter: {
    card: "summary_large_image",
    title: "Mosaic — Synthesize answers across your documents",
    description:
      "Mosaic assembles answers from pieces of your documents, with every source cited.",
    images: ["/og-image.png"],
  },
};

export default function RootLayout({
  children,
}: Readonly<{
  children: React.ReactNode;
}>) {
  return (
    <html lang="en" className={`${heading.variable} ${body.variable} ${mono.variable}`} suppressHydrationWarning>
      <head>
        <script dangerouslySetInnerHTML={{ __html: `(function(){try{var t=localStorage.getItem("theme");if(t==="dark"||(t==null&&matchMedia("(prefers-color-scheme:dark)").matches))document.documentElement.classList.add("dark")}catch(e){}})()` }} />
      </head>
      <body className="antialiased">
        {children}
      </body>
    </html>
  );
}
