/**
 * Thumbnail Generator Service
 *
 * Main orchestration service for generating thumbnails.
 * Handles preview extraction, fallback decoding, and batch processing.
 */

import fs from 'fs/promises';
import path from 'path';
import PQueue from 'p-queue';
import type { Config, Preset, GenerationResult, BatchResult, ThumbnailResult } from '../schemas/index.js';
import { VIDEO_EXTENSIONS } from '../schemas/index.js';
import { analyzeEmbeddedPreviews, extractBestPreview, isRawFormat, isDecodedFormat, isVideoFormat } from '../core/extractor.js';
import { generateThumbnails as resizeThumbnails, generateThumbnail } from '../core/resizer.js';
import { getBehavior, expandPath } from '../core/config.js';
import { ShoemakerError, ErrorCode, wrapError, shouldStopBatch, shouldReduceConcurrency } from '../core/errors.js';
import { decodeRawFile as decodeRaw, type DecodeOptions } from '../core/decoder.js';
import { updateXmpSidecar, hasExistingThumbnails } from './xmp-updater.js';
import { probeVideo, checkFfprobeAvailable } from '../core/ffprobe.js';
import { extractPosterFrame, extractPreviewFrame, generateTimelineStrip } from '../core/frame-extractor.js';

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
  method?: 'extracted' | 'decoded' | 'direct' | 'video';
  status: 'processing' | 'success' | 'error' | 'skipped';
  message?: string;
  duration?: number;
}

/**
 * Generate thumbnails for a video file
 */
async function generateVideoThumbnails(
  filePath: string,
  options: GenerateOptions
): Promise<GenerationResult> {
  const startTime = Date.now();
  const { config, dryRun } = options;
  const warnings: string[] = [];

  // Check FFprobe availability
  const ffprobeAvailable = await checkFfprobeAvailable();
  if (!ffprobeAvailable) {
    throw new ShoemakerError(
      'FFprobe not found. Install FFmpeg to process video files.',
      ErrorCode.DECODER_NOT_AVAILABLE,
      filePath
    );
  }

  // Get video info
  const videoInfo = await probeVideo(filePath);

  if (videoInfo.isInterlaced) {
    warnings.push('Video is interlaced, applying deinterlacing');
  }
  if (videoInfo.isHdr) {
    warnings.push('Video is HDR, applying tone mapping');
  }
  if (videoInfo.rotation) {
    warnings.push(`Video has rotation: ${videoInfo.rotation}°`);
  }

  if (dryRun) {
    return {
      source: filePath,
      method: 'video',
      thumbnails: [],
      warnings: ['Dry run: no files written', ...warnings],
      duration: Date.now() - startTime,
    };
  }

  // Determine output directory
  const stem = path.basename(filePath, path.extname(filePath));
  const outputDir = getOutputDir(filePath, stem, config);
  await fs.mkdir(outputDir, { recursive: true });

  const thumbnails: ThumbnailResult[] = [];
  const videoConfig = config.video;

  // Extract poster frame
  const posterFrame = await extractPosterFrame(filePath, videoConfig);
  const posterResult = await generateThumbnail(
    posterFrame.buffer,
    path.join(outputDir, `${stem}_poster_${config.sizes.thumb?.width ?? 300}.webp`),
    {
      width: config.sizes.thumb?.width ?? 300,
      format: (config.sizes.thumb?.format ?? 'webp') as 'webp' | 'jpeg' | 'png' | 'avif',
      quality: config.sizes.thumb?.quality ?? 80,
    }
  );
  thumbnails.push({ ...posterResult, size: 'poster' });

  // Extract preview frame
  const previewFrame = await extractPreviewFrame(filePath, videoConfig);
  const previewResult = await generateThumbnail(
    previewFrame.buffer,
    path.join(outputDir, `${stem}_preview_${config.sizes.preview?.width ?? 1600}.webp`),
    {
      width: config.sizes.preview?.width ?? 1600,
      format: (config.sizes.preview?.format ?? 'webp') as 'webp' | 'jpeg' | 'png' | 'avif',
      quality: config.sizes.preview?.quality ?? 85,
    }
  );
  thumbnails.push({ ...previewResult, size: 'preview' });

  // Generate timeline strip
  const timelineBuffer = await generateTimelineStrip(
    filePath,
    videoConfig.timelineFrames,
    videoConfig.timelineHeight,
    {
      deinterlace: videoConfig.autoDeinterlace,
      rotate: videoConfig.autoRotate,
      skipBlackFrames: videoConfig.skipBlackFrames,
      hdrToneMap: videoConfig.hdrToneMap,
    }
  );
  const timelinePath = path.join(outputDir, `${stem}_timeline.jpg`);
  await fs.writeFile(timelinePath, timelineBuffer);

  const timelineStats = await fs.stat(timelinePath);
  thumbnails.push({
    size: 'timeline',
    width: 0, // Will be calculated by strip dimensions
    height: videoConfig.timelineHeight,
    format: 'jpeg',
    path: timelinePath,
    bytes: timelineStats.size,
  });

  // Update XMP sidecar with video metadata
  if (config.xmp.updateSidecars) {
    await updateXmpSidecar(filePath, {
      thumbnails,
      method: 'video',
      videoInfo,
    });
  }

  return {
    source: filePath,
    method: 'video',
    thumbnails,
    warnings,
    duration: Date.now() - startTime,
  };
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

  // Route video files to video processor
  if (isVideoFormat(filePath)) {
    return generateVideoThumbnails(filePath, options);
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
    // RAW file - check for empty file before expensive operations
    const stats = await fs.stat(filePath);
    if (stats.size === 0) {
      throw new ShoemakerError(
        `File is empty: ${filePath}`,
        ErrorCode.CORRUPT_FILE,
        filePath
      );
    }
    // RAW file - try to extract preview
    const analysis = await analyzeEmbeddedPreviews(filePath);

    if (analysis.bestPreview && analysis.bestPreview.width >= config.processing.minPreviewSize) {
      // Fast path: extract embedded preview
      const extracted = await extractBestPreview(filePath);
      sourceBuffer = extracted.buffer;
      method = 'extracted';
    } else if (behavior.fallbackToRaw) {
      // Slow path: decode RAW using configured decoder
      const decodeOptions: DecodeOptions = {
        decoder: behavior.decoder as DecodeOptions['decoder'],
        fallbackDecoder: behavior.fallbackDecoder as DecodeOptions['decoder'],
        targetWidth: config.processing.minPreviewSize,
        quality: 95,
      };
      sourceBuffer = await decodeRaw(filePath, decodeOptions);
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
 * Find all supported media files (images and videos) in a directory
 */
export async function findImageFiles(
  dirPath: string,
  config: { filetypes: { include: string[]; exclude: string[] } },
  recursive: boolean = false,
  includeVideo: boolean = true
): Promise<string[]> {
  const files: string[] = [];
  const include = new Set(config.filetypes.include.map(ext => ext.toLowerCase()));
  const exclude = new Set(config.filetypes.exclude.map(ext => ext.toLowerCase()));

  // Add video extensions if enabled
  const videoExtensions = includeVideo
    ? new Set(VIDEO_EXTENSIONS as readonly string[])
    : new Set<string>();

  async function scanDir(dir: string) {
    const entries = await fs.readdir(dir, { withFileTypes: true });

    for (const entry of entries) {
      const fullPath = path.join(dir, entry.name);

      if (entry.isDirectory() && recursive) {
        await scanDir(fullPath);
      } else if (entry.isFile()) {
        const ext = path.extname(entry.name).slice(1).toLowerCase();
        const isIncluded = include.has(ext) || videoExtensions.has(ext);
        if (isIncluded && !exclude.has(ext)) {
          files.push(fullPath);
        }
      }
    }
  }

  await scanDir(dirPath);
  return files.sort();
}

/**
 * Find all supported video files in a directory
 */
export async function findVideoFiles(
  dirPath: string,
  recursive: boolean = false
): Promise<string[]> {
  const files: string[] = [];
  const videoExtensions = new Set(VIDEO_EXTENSIONS as readonly string[]);

  async function scanDir(dir: string) {
    const entries = await fs.readdir(dir, { withFileTypes: true });

    for (const entry of entries) {
      const fullPath = path.join(dir, entry.name);

      if (entry.isDirectory() && recursive) {
        await scanDir(fullPath);
      } else if (entry.isFile()) {
        const ext = path.extname(entry.name).slice(1).toLowerCase();
        if (videoExtensions.has(ext)) {
          files.push(fullPath);
        }
      }
    }
  }

  await scanDir(dirPath);
  return files.sort();
}
