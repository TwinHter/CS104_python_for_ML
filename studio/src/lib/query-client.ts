import { QueryClient } from '@tanstack/react-query';

const queryClient = new QueryClient({
  defaultOptions: {
    queries: {
      // Global default query options
      staleTime: 1000 * 60 * 5, // 5 minutes
      refetchOnWindowFocus: false, // To prevent excessive refetches for this demo
    },
  },
});

export default queryClient;
