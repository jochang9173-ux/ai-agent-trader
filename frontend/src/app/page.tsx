'use client'

import { QueryClient, QueryClientProvider } from '@tanstack/react-query'
import StreamingLLMRunner from '@/components/llm/StreamingLLMRunner'
import { useState } from 'react'

export default function Home() {
  const [queryClient] = useState(
    () => new QueryClient({
      defaultOptions: {
        queries: {
          staleTime: 60 * 1000, // 1 minute
          retry: 1,
        },
      },
    })
  )

  return (
    <QueryClientProvider client={queryClient}>
      <StreamingLLMRunner />
    </QueryClientProvider>
  )
}
