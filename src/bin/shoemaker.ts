#!/usr/bin/env node
/**
 * Shoemaker CLI Entry Point
 */

import { run } from '../cli/index.js';

run().catch((err) => {
  console.error('Fatal error:', err.message);
  process.exit(1);
});
