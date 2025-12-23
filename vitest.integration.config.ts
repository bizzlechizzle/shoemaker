import { defineConfig } from 'vitest/config';

export default defineConfig({
  test: {
    include: ['tests/integration/**/*.test.ts'],
    testTimeout: 60000, // 60 seconds for integration tests
    hookTimeout: 30000,
    pool: 'forks', // Use forks for isolation
    poolOptions: {
      forks: {
        singleFork: true, // Run sequentially to avoid resource conflicts
      },
    },
  },
});
