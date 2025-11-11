import type { Metadata } from "next";
import { Inter } from "next/font/google";
import localFont from "next/font/local";
import { Providers } from "./providers";
import "./globals.css";

// Inter font for Latin characters
const inter = Inter({
  subsets: ["latin"],
  variable: "--font-inter",
  display: "swap",
});

// SUIT font for Korean characters
// Temporarily commented out until font file is downloaded
// const suit = localFont({
//   src: "../fonts/SUIT-Variable.woff2",
//   variable: "--font-suit",
//   display: "swap",
//   weight: "100 900",
// });

export const metadata: Metadata = {
  title: "Vision AI Training Platform",
  description: "Natural language-driven computer vision model training platform",
};

export default function RootLayout({
  children,
}: Readonly<{
  children: React.ReactNode;
}>) {
  return (
    <html lang="ko" className={`${inter.variable}`}>
      <body className="font-sans antialiased">
        <Providers>{children}</Providers>
      </body>
    </html>
  );
}
