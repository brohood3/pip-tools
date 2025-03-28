import type { Metadata } from "next";
import { headers } from 'next/headers'
import './globals.css';
import ContextProvider from '@/context'

export const metadata: Metadata = {
  title: "Chat with Pip",
  description: "Chat with Pip, a baby AI agent learning about the world",
};

export default async function RootLayout({
  children,
}: Readonly<{
  children: React.ReactNode;
}>) {
  const headersData = await headers();
  const cookies = headersData.get('cookie');

  return (
    <html lang="en">
      <head>
        <link 
          rel="stylesheet" 
          href="https://cdnjs.cloudflare.com/ajax/libs/font-awesome/6.4.0/css/all.min.css" 
        />
      </head>
      <body>
        <ContextProvider cookies={cookies}>{children}</ContextProvider>
      </body>
    </html>
  );
}
