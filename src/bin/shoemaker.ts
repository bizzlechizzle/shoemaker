#!/usr/bin/env node
/**
 * Shoemaker CLI Entry Point
 */

import { run } from '../cli/index.js';
import { shutdownExiftool } from '../core/extractor.js';

run().catch(async (err) => {
  console.error('Fatal error:', err.message);
  // Ensure ExifTool is properly shut down before exiting
  try {
    await shutdownExiftool();
  } catch {
    // Ignore shutdown errors during fatal exit
  }
  process.exit(1);
});
