/**
 * thumb command
 *
 * Generate thumbnails for image files.
 */

import { Command } from 'commander';
import fs from 'fs/promises';
import path from 'path';
import ora from 'ora';
import { loadConfig, loadPreset, applyPreset } from '../../core/config.js';
import { generateForFile, generateForBatch, findImageFiles, type ProgressInfo } from '../../services/thumbnail-generator.js';
import { hasExistingThumbnails } from '../../services/xmp-updater.js';
import type { Config, Preset, BatchResult } from '../../schemas/index.js';
import { MAX_CONCURRENCY, MAX_ERRORS_TO_DISPLAY } from '../../schemas/index.js';
import { wrapError } from '../../core/errors.js';

/**
 * Validate that input path exists and is accessible
 */
async function validateInputPath(inputPath: string): Promise<Awaited<ReturnType<typeof fs.stat>>> {
  try {
    return await fs.stat(inputPath);
  } catch (err) {
    const nodeErr = err as NodeJS.ErrnoException;
    if (nodeErr.code === 'ENOENT') {
      console.error(`Error: Path not found: ${inputPath}`);
      process.exit(1);
    }
    if (nodeErr.code === 'EACCES') {
      console.error(`Error: Permission denied: ${inputPath}`);
      process.exit(1);
    }
    throw wrapError(err, inputPath);
  }
}

/**
 * Validate and parse concurrency option
 */
function validateConcurrency(value: string): number {
  const concurrency = parseInt(value, 10);
  if (isNaN(concurrency) || concurrency < 1 || concurrency > MAX_CONCURRENCY) {
    console.error(`Error: Concurrency must be between 1 and ${MAX_CONCURRENCY}`);
    process.exit(1);
  }
  return concurrency;
}

export const thumbCommand = new Command('thumb')
  .description('Generate thumbnails for image files')
  .argument('<path>', 'File or directory to process')
  .option('-r, --recursive', 'Process subdirectories', false)
  .option('-p, --preset <name>', 'Preset to use (fast, quality, portable)', 'fast')
  .option('-f, --force', 'Regenerate even if thumbnails exist', false)
  .option('--resume', 'Skip files that already have thumbnails (resume interrupted batch)', false)
  .option('--dry-run', 'Show what would be done without writing files', false)
  .option('-c, --concurrency <n>', 'Number of files to process in parallel', '4')
  .option('-q, --quiet', 'Suppress progress output', false)
  .option('--json', 'Output results as JSON', false)
  .option('--error-log <path>', 'Write error log to JSON file')
  .option('--proxy', 'Generate video proxy files (lower-res copies for editing)', false)
  .option('--proxy-size <sizes>', 'Proxy sizes to generate (small,medium,large)', 'medium')
  .option('--lut <path>', 'Apply .cube LUT file to proxies for color grading')
  .action(async (inputPath: string, options) => {
    const config = await loadConfig();
    const preset = await loadPreset(options.preset, config);
    const finalConfig = applyPreset(config, preset);

    // Validate and override concurrency if specified
    if (options.concurrency) {
      finalConfig.processing.concurrency = validateConcurrency(options.concurrency);
    }

    // Enable proxy generation if --proxy flag is set
    if (options.proxy) {
      finalConfig.video.proxy.enabled = true;
      // Parse proxy sizes if specified
      if (options.proxySize) {
        const requestedSizes = options.proxySize.split(',').map((s: string) => s.trim());
        const defaultSizes = finalConfig.video.proxy.sizes;
        const filteredSizes: typeof defaultSizes = {};
        for (const size of requestedSizes) {
          if (defaultSizes[size]) {
            filteredSizes[size] = defaultSizes[size];
          }
        }
        if (Object.keys(filteredSizes).length > 0) {
          finalConfig.video.proxy.sizes = filteredSizes;
        }
      }
      // Apply LUT if specified
      if (options.lut) {
        finalConfig.video.proxy.lutPath = options.lut;
      }
    }

    // Validate input path exists
    const stat = await validateInputPath(inputPath);
    const isDirectory = stat.isDirectory();

    if (isDirectory) {
      await processDirectory(inputPath, finalConfig, preset, options);
    } else {
      await processSingleFile(inputPath, finalConfig, preset, options);
    }
  });

async function processSingleFile(
  filePath: string,
  config: Config,
  preset: Preset,
  options: { force?: boolean; dryRun?: boolean; quiet?: boolean; json?: boolean }
): Promise<void> {
  const spinner = options.quiet ? null : ora(`Processing ${path.basename(filePath)}`).start();

  try {
    const result = await generateForFile(filePath, {
      config,
      preset,
      force: options.force,
      dryRun: options.dryRun,
    });

    if (options.json) {
      console.log(JSON.stringify(result, null, 2));
    } else if (spinner) {
      const icon = result.warnings.some(w => w.includes('Skipped')) ? '⏭' : '✓';
      spinner.succeed(`${icon} ${path.basename(filePath)} → ${result.method} → ${result.thumbnails.length} thumbs (${result.duration}ms)`);

      for (const warning of result.warnings) {
        console.log(`  ⚠ ${warning}`);
      }
    }
  } catch (err) {
    if (spinner) {
      spinner.fail(`✗ ${path.basename(filePath)}: ${(err as Error).message}`);
    }
    if (options.json) {
      console.log(JSON.stringify({ error: (err as Error).message }, null, 2));
    }
    process.exitCode = 1;
  }
}

async function processDirectory(
  dirPath: string,
  config: Config,
  preset: Preset,
  options: { recursive?: boolean; force?: boolean; resume?: boolean; dryRun?: boolean; quiet?: boolean; json?: boolean; errorLog?: string }
): Promise<void> {
  let files = await findImageFiles(dirPath, config, options.recursive);

  if (files.length === 0) {
    if (!options.quiet && !options.json) {
      console.log('No supported image files found.');
    }
    return;
  }

  // If resume mode, filter out files that already have thumbnails
  if (options.resume && !options.force) {
    const spinner = options.quiet ? null : ora('Checking existing thumbnails...').start();
    const pendingFiles: string[] = [];
    let checked = 0;

    for (const file of files) {
      const hasThumbs = await hasExistingThumbnails(file);
      if (!hasThumbs) {
        pendingFiles.push(file);
      }
      checked++;
      if (spinner) {
        spinner.text = `Checking existing thumbnails... ${checked}/${files.length}`;
      }
    }

    if (spinner) {
      spinner.stop();
    }

    const skippedCount = files.length - pendingFiles.length;
    if (skippedCount > 0 && !options.quiet && !options.json) {
      console.log(`Resuming: skipping ${skippedCount} files with existing thumbnails`);
    }

    files = pendingFiles;

    if (files.length === 0) {
      if (!options.quiet && !options.json) {
        console.log('All files already have thumbnails. Use --force to regenerate.');
      }
      return;
    }
  }

  if (!options.quiet && !options.json) {
    console.log(`\nProcessing ${files.length} files...\n`);
  }

  const spinner = options.quiet ? null : ora('Starting...').start();

  const result = await generateForBatch(files, {
    config,
    preset,
    force: options.force,
    dryRun: options.dryRun,
    onProgress: (info: ProgressInfo) => {
      if (spinner && !options.quiet) {
        const icon = info.status === 'success' ? '✓' : info.status === 'error' ? '✗' : info.status === 'skipped' ? '⏭' : '→';
        spinner.text = `[${info.completed}/${info.total}] ${icon} ${path.basename(info.current)}`;
      }
    },
  });

  if (spinner) {
    spinner.stop();
  }

  if (options.json) {
    console.log(JSON.stringify(result, null, 2));
  } else if (!options.quiet) {
    console.log(`\nDone: ${result.total} files`);

    // Avoid division by zero for percentage calculations
    const pct = (n: number) => result.total > 0 ? ((n / result.total) * 100).toFixed(1) : '0.0';

    console.log(`  ✓ Succeeded: ${result.succeeded} (${pct(result.succeeded)}%)`);
    if (result.skipped > 0) {
      console.log(`  ⏭ Skipped: ${result.skipped} (${pct(result.skipped)}%)`);
    }
    if (result.failed > 0) {
      console.log(`  ✗ Failed: ${result.failed} (${pct(result.failed)}%)`);
    }

    const avgTime = result.total > 0 ? (result.duration / result.total).toFixed(0) : '0';
    console.log(`\nTime: ${(result.duration / 1000).toFixed(1)}s (avg ${avgTime}ms/file)`);

    if (result.errors.length > 0 && !options.json) {
      console.log('\nErrors:');
      for (const error of result.errors.slice(0, MAX_ERRORS_TO_DISPLAY)) {
        console.log(`  ${path.basename(error.file)}: ${error.message}`);
      }
      if (result.errors.length > MAX_ERRORS_TO_DISPLAY) {
        console.log(`  ... and ${result.errors.length - MAX_ERRORS_TO_DISPLAY} more`);
      }
    }
  }

  // Write error log file if specified or if there are errors
  if (result.errors.length > 0) {
    const errorLogPath = options.errorLog ?? path.join(dirPath, '.shoemaker-errors.json');
    await writeErrorLog(errorLogPath, result, dirPath);

    if (!options.quiet && !options.json) {
      console.log(`\nError log written to: ${path.relative(process.cwd(), errorLogPath)}`);
    }
  }

  if (result.failed > 0) {
    process.exitCode = 1;
  }
}

/**
 * Write error log to JSON file
 */
async function writeErrorLog(
  logPath: string,
  result: BatchResult,
  basePath: string
): Promise<void> {
  const errorLog = {
    timestamp: new Date().toISOString(),
    version: '0.1.6',
    basePath,
    batch: {
      total: result.total,
      succeeded: result.succeeded,
      failed: result.failed,
      skipped: result.skipped,
      warnings: result.warnings,
      durationMs: result.duration,
    },
    errors: result.errors.map(e => ({
      file: e.file,
      relativeFile: path.relative(basePath, e.file),
      code: e.code,
      message: e.message,
      timestamp: new Date().toISOString(),
    })),
  };

  await fs.writeFile(logPath, JSON.stringify(errorLog, null, 2));
}
