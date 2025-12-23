/**
 * status command
 *
 * Show which files need thumbnail generation.
 */

import { Command } from 'commander';
import fs from 'fs/promises';
import path from 'path';
import { loadConfig } from '../../core/config.js';
import { hasExistingThumbnails } from '../../services/xmp-updater.js';
import { findImageFiles } from '../../services/thumbnail-generator.js';
import { MAX_PENDING_TO_DISPLAY } from '../../schemas/index.js';
import { wrapError } from '../../core/errors.js';

/**
 * Validate that input path exists and is a directory
 */
async function validateDirectoryPath(dirPath: string): Promise<void> {
  try {
    const stat = await fs.stat(dirPath);
    if (!stat.isDirectory()) {
      console.error(`Error: Not a directory: ${dirPath}`);
      process.exit(1);
    }
  } catch (err) {
    const nodeErr = err as NodeJS.ErrnoException;
    if (nodeErr.code === 'ENOENT') {
      console.error(`Error: Path not found: ${dirPath}`);
      process.exit(1);
    }
    throw wrapError(err, dirPath);
  }
}

export const statusCommand = new Command('status')
  .description('Show files that need thumbnail generation')
  .argument('<path>', 'Directory to check')
  .option('-r, --recursive', 'Check subdirectories', false)
  .option('--json', 'Output as JSON', false)
  .action(async (dirPath: string, options) => {
    await validateDirectoryPath(dirPath);
    const config = await loadConfig();
    const files = await findImageFiles(dirPath, config, options.recursive);

    if (files.length === 0) {
      if (!options.json) {
        console.log('No supported image files found.');
      } else {
        console.log(JSON.stringify({ total: 0, pending: [], processed: [] }));
      }
      return;
    }

    const pending: string[] = [];
    const processed: string[] = [];

    for (const file of files) {
      const hasThumbs = await hasExistingThumbnails(file);
      if (hasThumbs) {
        processed.push(file);
      } else {
        pending.push(file);
      }
    }

    if (options.json) {
      console.log(JSON.stringify({
        total: files.length,
        pendingCount: pending.length,
        processedCount: processed.length,
        pending: pending.map(f => path.relative(dirPath, f)),
        processed: processed.map(f => path.relative(dirPath, f)),
      }, null, 2));
      return;
    }

    console.log(`\nTotal files: ${files.length}`);
    console.log(`  ✓ Processed: ${processed.length} (${((processed.length / files.length) * 100).toFixed(1)}%)`);
    console.log(`  ○ Pending: ${pending.length} (${((pending.length / files.length) * 100).toFixed(1)}%)`);

    if (pending.length > 0 && pending.length <= MAX_PENDING_TO_DISPLAY) {
      console.log('\nPending files:');
      for (const file of pending) {
        console.log(`  ${path.relative(dirPath, file)}`);
      }
    } else if (pending.length > MAX_PENDING_TO_DISPLAY) {
      console.log(`\nFirst ${MAX_PENDING_TO_DISPLAY} pending files:`);
      for (const file of pending.slice(0, MAX_PENDING_TO_DISPLAY)) {
        console.log(`  ${path.relative(dirPath, file)}`);
      }
      console.log(`  ... and ${pending.length - MAX_PENDING_TO_DISPLAY} more`);
    }

    if (pending.length > 0) {
      console.log(`\nRun: shoemaker thumb "${dirPath}" to process pending files`);
    }
  });
