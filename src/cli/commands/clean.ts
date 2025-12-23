/**
 * clean command
 *
 * Remove generated thumbnails and clear XMP metadata.
 */

import { Command } from 'commander';
import fs from 'fs/promises';
import path from 'path';
import ora from 'ora';
import { loadConfig } from '../../core/config.js';
import { readThumbnailInfo, clearThumbnailInfo } from '../../services/xmp-updater.js';
import { findImageFiles } from '../../services/thumbnail-generator.js';

export const cleanCommand = new Command('clean')
  .description('Remove generated thumbnails')
  .argument('<path>', 'File or directory to clean')
  .option('-r, --recursive', 'Process subdirectories', false)
  .option('--dry-run', 'Show what would be removed without deleting', false)
  .option('-q, --quiet', 'Suppress progress output', false)
  .action(async (inputPath: string, options) => {
    const config = await loadConfig();
    const stat = await fs.stat(inputPath);

    if (stat.isDirectory()) {
      await cleanDirectory(inputPath, options, config);
    } else {
      await cleanFile(inputPath, options, config);
    }
  });

async function cleanFile(
  filePath: string,
  options: { dryRun?: boolean; quiet?: boolean },
  config: { output: { sidecarFolder: string } }
): Promise<void> {
  const info = await readThumbnailInfo(filePath);

  if (!info.exists) {
    if (!options.quiet) {
      console.log(`No thumbnails found for ${path.basename(filePath)}`);
    }
    return;
  }

  const stem = path.basename(filePath, path.extname(filePath));
  const thumbDir = path.join(
    path.dirname(filePath),
    config.output.sidecarFolder.replace('{stem}', stem)
  );

  if (options.dryRun) {
    console.log(`Would remove: ${thumbDir}`);
    console.log(`Would clear XMP: ${filePath}.xmp`);
    return;
  }

  // Remove thumbnail directory
  try {
    await fs.rm(thumbDir, { recursive: true });
    if (!options.quiet) {
      console.log(`Removed: ${thumbDir}`);
    }
  } catch {
    // Directory might not exist
  }

  // Clear XMP metadata
  await clearThumbnailInfo(filePath);
  if (!options.quiet) {
    console.log(`Cleared XMP: ${filePath}.xmp`);
  }
}

async function cleanDirectory(
  dirPath: string,
  options: { recursive?: boolean; dryRun?: boolean; quiet?: boolean },
  config: { output: { sidecarFolder: string }; filetypes: { include: string[]; exclude: string[] } }
): Promise<void> {
  const files = await findImageFiles(dirPath, config, options.recursive);

  if (files.length === 0) {
    if (!options.quiet) {
      console.log('No supported image files found.');
    }
    return;
  }

  const spinner = options.quiet ? null : ora('Scanning for thumbnails...').start();
  let cleaned = 0;
  let skipped = 0;

  for (const file of files) {
    const info = await readThumbnailInfo(file);

    if (!info.exists) {
      skipped++;
      continue;
    }

    if (spinner) {
      spinner.text = `Cleaning ${path.basename(file)}...`;
    }

    const stem = path.basename(file, path.extname(file));
    const thumbDir = path.join(
      path.dirname(file),
      config.output.sidecarFolder.replace('{stem}', stem)
    );

    if (options.dryRun) {
      console.log(`Would remove: ${thumbDir}`);
    } else {
      try {
        await fs.rm(thumbDir, { recursive: true });
      } catch {
        // Directory might not exist
      }
      await clearThumbnailInfo(file);
    }

    cleaned++;
  }

  if (spinner) {
    spinner.stop();
  }

  if (!options.quiet) {
    console.log(`\nCleaned: ${cleaned} files`);
    console.log(`Skipped: ${skipped} files (no thumbnails)`);
    if (options.dryRun) {
      console.log('\n(Dry run - no files were actually removed)');
    }
  }
}
