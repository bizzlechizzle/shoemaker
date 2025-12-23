/**
 * Thumbnail Generator Service
 *
 * Main orchestration service for generating thumbnails.
 * Handles preview extraction, fallback decoding, and batch processing.
 */

import fs from 'fs/promises';
import path from 'path';
import PQueue from 'p-queue';
import type { Config, Preset, GenerationResult, BatchResult } from '../schemas/index.js';
import { analyzeEmbeddedPreviews, extractBestPreview, isRawFormat, isDecodedFormat } from '../core/extractor.js';
import { generateThumbnails as resizeThumbnails } from '../core/resizer.js';
import { getBehavior, expandPath } from '../core/config.js';
import { ShoemakerError, ErrorCode, wrapError, shouldStopBatch, shouldReduceConcurrency } from '../core/errors.js';
import { updateXmpSidecar, hasExistingThumbnails } from './xmp-updater.js';

export interface GenerateOptions {
  config: Config;
  preset: Preset;
  force?: boolean;
  dryRun?: boolean;
  onProgress?: (info: ProgressInfo) => void;
}

export interface ProgressInfo {
  current: string;
  completed: number;
  total: number;
  method?: 'extracted' | 'decoded' | 'direct';
  status: 'processing' | 'success' | 'error' | 'skipped';
  message?: string;
  duration?: number;
}

/**
 * Generate thumbnails for a single file
 */
export async function generateForFile(
  filePath: string,
  options: GenerateOptions
): Promise<GenerationResult> {
  const startTime = Date.now();
  const { config, preset, force, dryRun } = options;
  const behavior = getBehavior(preset);
  const warnings: string[] = [];

  // Check if already processed
  if (!force && config.processing.skipExisting) {
    const hasExisting = await hasExistingThumbnails(filePath);
    if (hasExisting) {
      return {
        source: filePath,
        method: 'extracted',
        thumbnails: [],
        warnings: ['Skipped: thumbnails already exist'],
        duration: Date.now() - startTime,
      };
    }
  }

  // Determine source method and get source buffer
  let sourceBuffer: Buffer;
  let method: 'extracted' | 'decoded' | 'direct';

  if (isDecodedFormat(filePath)) {
    // Already decoded (JPEG, PNG, etc) - read directly
    sourceBuffer = await fs.readFile(filePath);
    // Check for empty file
    if (sourceBuffer.length === 0) {
      throw new ShoemakerError(
        `File is empty: ${filePath}`,
        ErrorCode.CORRUPT_FILE,
        filePath
      );
    }
    method = 'direct';
  } else if (isRawFormat(filePath)) {
    // RAW file - try to extract preview
    const analysis = await analyzeEmbeddedPreviews(filePath);

    if (analysis.bestPreview && analysis.bestPreview.width >= config.processing.minPreviewSize) {
      // Fast path: extract embedded preview
      const extracted = await extractBestPreview(filePath);
      sourceBuffer = extracted.buffer;
      method = 'extracted';
    } else if (behavior.fallbackToRaw) {
      // Slow path: decode RAW
      sourceBuffer = await decodeRawFile(filePath, behavior.decoder);
      method = 'decoded';
    } else if (behavior.useLargestAvailable && analysis.bestPreview) {
      // Use whatever's available
      const extracted = await extractBestPreview(filePath);
      sourceBuffer = extracted.buffer;
      method = 'extracted';
      warnings.push(`Preview size ${analysis.bestPreview.width}px below minimum ${config.processing.minPreviewSize}px`);
    } else {
      throw new ShoemakerError(
        `No suitable preview found and RAW fallback disabled`,
        ErrorCode.NO_PREVIEW,
        filePath
      );
    }
  } else {
    // Unknown format - try to read directly
    sourceBuffer = await fs.readFile(filePath);
    // Check for empty file
    if (sourceBuffer.length === 0) {
      throw new ShoemakerError(
        `File is empty: ${filePath}`,
        ErrorCode.CORRUPT_FILE,
        filePath
      );
    }
    method = 'direct';
  }

  if (dryRun) {
    return {
      source: filePath,
      method,
      thumbnails: [],
      warnings: ['Dry run: no files written'],
      duration: Date.now() - startTime,
    };
  }

  // Determine output directory
  const stem = path.basename(filePath, path.extname(filePath));
  const outputDir = getOutputDir(filePath, stem, config);

  // Generate all thumbnail sizes
  const thumbnails = await resizeThumbnails(
    sourceBuffer,
    outputDir,
    stem,
    config.sizes,
    {
      stripExif: config.processing.stripExif,
      autoRotate: config.processing.autoRotate,
    }
  );

  // Update XMP sidecar
  if (config.xmp.updateSidecars) {
    await updateXmpSidecar(filePath, {
      thumbnails,
      method,
    });
  }

  return {
    source: filePath,
    method,
    thumbnails,
    warnings,
    duration: Date.now() - startTime,
  };
}

/**
 * Generate thumbnails for multiple files
 */
export async function generateForBatch(
  files: string[],
  options: GenerateOptions
): Promise<BatchResult> {
  const startTime = Date.now();
  const { config, onProgress } = options;

  let concurrency = config.processing.concurrency;
  const queue = new PQueue({ concurrency });

  const results: { file: string; result?: GenerationResult; error?: ShoemakerError }[] = [];
  let completed = 0;

  for (const file of files) {
    queue.add(async () => {
      const progressInfo: ProgressInfo = {
        current: file,
        completed,
        total: files.length,
        status: 'processing',
      };
      onProgress?.(progressInfo);

      try {
        const result = await generateForFile(file, options);
        completed++;
        results.push({ file, result });

        onProgress?.({
          ...progressInfo,
          completed,
          method: result.method,
          status: result.warnings.some(w => w.includes('Skipped')) ? 'skipped' : 'success',
          duration: result.duration,
        });
      } catch (err) {
        const shoemakerError = wrapError(err, file);
        completed++;
        results.push({ file, error: shoemakerError });

        // Check if we should stop the batch
        if (shouldStopBatch(err)) {
          queue.clear();
          throw shoemakerError;
        }

        // Check if we should reduce concurrency
        if (shouldReduceConcurrency(err)) {
          concurrency = Math.max(1, concurrency - 1);
          queue.concurrency = concurrency;
        }

        onProgress?.({
          ...progressInfo,
          completed,
          status: 'error',
          message: shoemakerError.message,
        });
      }
    });
  }

  await queue.onIdle();

  // Compile batch result
  const succeeded = results.filter(r => r.result && !r.result.warnings.some(w => w.includes('Skipped'))).length;
  const skipped = results.filter(r => r.result?.warnings.some(w => w.includes('Skipped'))).length;
  const failed = results.filter(r => r.error).length;
  const withWarnings = results.filter(r => r.result && r.result.warnings.length > 0 && !r.result.warnings.some(w => w.includes('Skipped'))).length;

  return {
    total: files.length,
    succeeded,
    failed,
    skipped,
    warnings: withWarnings,
    duration: Date.now() - startTime,
    errors: results
      .filter(r => r.error)
      .map(r => ({
        file: r.file,
        code: r.error!.code,
        message: r.error!.message,
      })),
  };
}

/**
 * Get output directory for thumbnails
 */
function getOutputDir(filePath: string, stem: string, config: Config): string {
  if (config.output.location === 'cache') {
    // Use centralized cache with safe path expansion
    const cacheDir = expandPath(config.output.cacheDir);
    return path.join(cacheDir, stem);
  }

  // Sidecar folder next to source
  const sourceDir = path.dirname(filePath);
  const folderName = config.output.sidecarFolder.replace('{stem}', stem);
  return path.join(sourceDir, folderName);
}

/**
 * Decode RAW file using available decoder
 * Note: This is a placeholder - actual implementation would use libraw, RawTherapee, etc.
 */
async function decodeRawFile(filePath: string, decoder?: string): Promise<Buffer> {
  // For now, throw an error - RAW decoding requires additional implementation
  throw new ShoemakerError(
    `RAW decoding not yet implemented (decoder: ${decoder ?? 'none'})`,
    ErrorCode.DECODER_NOT_AVAILABLE,
    filePath
  );
}

/**
 * Find all supported image files in a directory
 */
export async function findImageFiles(
  dirPath: string,
  config: { filetypes: { include: string[]; exclude: string[] } },
  recursive: boolean = false
): Promise<string[]> {
  const files: string[] = [];
  const include = new Set(config.filetypes.include.map(ext => ext.toLowerCase()));
  const exclude = new Set(config.filetypes.exclude.map(ext => ext.toLowerCase()));

  async function scanDir(dir: string) {
    const entries = await fs.readdir(dir, { withFileTypes: true });

    for (const entry of entries) {
      const fullPath = path.join(dir, entry.name);

      if (entry.isDirectory() && recursive) {
        await scanDir(fullPath);
      } else if (entry.isFile()) {
        const ext = path.extname(entry.name).slice(1).toLowerCase();
        if (include.has(ext) && !exclude.has(ext)) {
          files.push(fullPath);
        }
      }
    }
  }

  await scanDir(dirPath);
  return files.sort();
}
