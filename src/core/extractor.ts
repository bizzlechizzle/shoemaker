/**
 * Preview Extractor
 *
 * Extracts embedded previews from RAW files using ExifTool.
 * Analyzes available previews and selects the best one.
 * Includes LRU cache for preview metadata to avoid repeated analysis.
 */

import { exiftool } from 'exiftool-vendored';
import type { PreviewAnalysis, PreviewInfo } from '../schemas/index.js';
import { DEFAULT_MIN_PREVIEW_SIZE, RAW_EXTENSIONS, DECODED_EXTENSIONS, VIDEO_EXTENSIONS } from '../schemas/index.js';
import { ShoemakerError, ErrorCode, wrapError } from './errors.js';

// Preview tag priority order (best to worst)
const PREVIEW_TAGS = ['JpgFromRaw', 'PreviewImage', 'OtherImage', 'ThumbnailImage'] as const;
type PreviewTag = typeof PREVIEW_TAGS[number];

/**
 * Convert EXIF Orientation tag value to Sharp rotation angle.
 * EXIF Orientation values:
 *   1 = Normal (0°)
 *   2 = Mirrored horizontal
 *   3 = Rotated 180°
 *   4 = Mirrored vertical
 *   5 = Mirrored horizontal + rotated 270° CW
 *   6 = Rotated 90° CW (270° CCW)
 *   7 = Mirrored horizontal + rotated 90° CW
 *   8 = Rotated 270° CW (90° CCW)
 *
 * Sharp .rotate() accepts: 0, 90, 180, 270
 * Note: Sharp handles mirroring separately with .flip()/.flop()
 * For simplicity, we only handle rotation (1, 3, 6, 8 are most common)
 */
export function orientationToRotation(orientation: number | string | undefined): number {
  const o = typeof orientation === 'string' ? parseInt(orientation, 10) : orientation;
  switch (o) {
    case 3: return 180;
    case 6: return 90;  // 90° CW to correct
    case 8: return 270; // 270° CW (90° CCW) to correct
    default: return 0;  // Normal or unknown
  }
}

// LRU cache for preview analysis results
const PREVIEW_CACHE_MAX_SIZE = 1000;
const previewCache = new Map<string, { analysis: PreviewAnalysis; timestamp: number }>();

/**
 * Get cached preview analysis or null if not cached
 */
function getCachedAnalysis(filePath: string): PreviewAnalysis | null {
  const cached = previewCache.get(filePath);
  if (cached) {
    // Check if cache is still valid (15 minutes)
    if (Date.now() - cached.timestamp < 15 * 60 * 1000) {
      return cached.analysis;
    }
    // Expired, remove from cache
    previewCache.delete(filePath);
  }
  return null;
}

/**
 * Cache a preview analysis result
 */
function cacheAnalysis(filePath: string, analysis: PreviewAnalysis): void {
  // Evict oldest entries if cache is full
  if (previewCache.size >= PREVIEW_CACHE_MAX_SIZE) {
    const oldestKey = previewCache.keys().next().value;
    if (oldestKey) {
      previewCache.delete(oldestKey);
    }
  }
  previewCache.set(filePath, { analysis, timestamp: Date.now() });
}

/**
 * Clear the preview cache (useful for testing or memory management)
 */
export function clearPreviewCache(): void {
  previewCache.clear();
}

/**
 * Type guard: check if value is a valid number
 */
function isNumber(value: unknown): value is number {
  return typeof value === 'number' && !isNaN(value) && isFinite(value);
}

/**
 * Analyze embedded previews in an image file
 * Uses cache to avoid repeated ExifTool calls for the same file
 */
export async function analyzeEmbeddedPreviews(filePath: string): Promise<PreviewAnalysis> {
  // Check cache first
  const cached = getCachedAnalysis(filePath);
  if (cached) {
    return cached;
  }

  try {
    const tags = await exiftool.read(filePath);

    const jpgFromRaw = extractPreviewInfo(tags, 'JpgFromRaw');
    const previewImage = extractPreviewInfo(tags, 'PreviewImage');
    const otherImage = extractPreviewInfo(tags, 'OtherImage');
    const thumbnailImage = extractPreviewInfo(tags, 'ThumbnailImage');

    // Find best preview (largest that exists)
    // Two-pass: first try previews with known dimensions, then fall back to those with only length
    let bestPreview: PreviewAnalysis['bestPreview'] = null;
    const previewMap = { JpgFromRaw: jpgFromRaw, PreviewImage: previewImage, OtherImage: otherImage, ThumbnailImage: thumbnailImage };

    // Pass 1: Prefer previews with dimensions (can compare by width)
    for (const tag of PREVIEW_TAGS) {
      const info = previewMap[tag];
      if (info.exists && info.width && info.height) {
        if (!bestPreview || (info.width > bestPreview.width)) {
          bestPreview = { type: tag, width: info.width, height: info.height };
        }
      }
    }

    // Pass 2: If no dimensioned previews, use length as quality proxy (many RAW files omit dimensions)
    // JpgFromRaw typically has dimensions after extraction, so still prioritize by tag order
    if (!bestPreview) {
      let bestLength = 0;
      for (const tag of PREVIEW_TAGS) {
        const info = previewMap[tag];
        // Exists with length > reasonable minimum (10KB) suggests a usable preview
        if (info.exists && info.length && info.length > 10240 && info.length > bestLength) {
          bestLength = info.length;
          // Use 0 for width/height to indicate "dimensions unknown - extract to measure"
          bestPreview = { type: tag, width: 0, height: 0 };
        }
      }
    }

    // Determine if RAW decode is needed (best preview < minimum size)
    const needsRawDecode = !bestPreview || bestPreview.width < DEFAULT_MIN_PREVIEW_SIZE;

    const analysis: PreviewAnalysis = {
      filePath,
      jpgFromRaw,
      previewImage,
      otherImage,
      thumbnailImage,
      bestPreview,
      needsRawDecode,
    };

    // Cache the result
    cacheAnalysis(filePath, analysis);

    return analysis;
  } catch (err) {
    throw wrapError(err, filePath);
  }
}

/**
 * Extract preview info for a specific tag with type-safe value extraction
 */
function extractPreviewInfo(tags: unknown, tagName: string): PreviewInfo {
  // Guard against invalid input
  if (!tags || typeof tags !== 'object') {
    return { exists: false };
  }

  const tagsRecord = tags as Record<string, unknown>;
  const data = tagsRecord[tagName];

  // Extract width with type checking
  const widthKey1 = `${tagName}Width`;
  const widthKey2 = `${tagName}ImageWidth`;
  const width = isNumber(tagsRecord[widthKey1])
    ? tagsRecord[widthKey1]
    : isNumber(tagsRecord[widthKey2])
    ? tagsRecord[widthKey2]
    : undefined;

  // Extract height with type checking
  const heightKey1 = `${tagName}Height`;
  const heightKey2 = `${tagName}ImageHeight`;
  const height = isNumber(tagsRecord[heightKey1])
    ? tagsRecord[heightKey1]
    : isNumber(tagsRecord[heightKey2])
    ? tagsRecord[heightKey2]
    : undefined;

  // Extract length with type checking
  const lengthKey = `${tagName}Length`;
  const length = isNumber(tagsRecord[lengthKey]) ? tagsRecord[lengthKey] : undefined;

  return {
    exists: data !== undefined && data !== null,
    width,
    height,
    length,
  };
}

/**
 * Extract the best available preview as a buffer
 * Also reads orientation from source file since extracted previews lose EXIF orientation
 */
export async function extractBestPreview(filePath: string, minSize?: number): Promise<{ buffer: Buffer; tag: PreviewTag; width: number; height: number; orientation: number }> {
  const sharp = (await import('sharp')).default;
  const analysis = await analyzeEmbeddedPreviews(filePath);

  if (!analysis.bestPreview) {
    throw new ShoemakerError(
      `No embedded preview found in ${filePath}`,
      ErrorCode.NO_PREVIEW,
      filePath
    );
  }

  // Read orientation from source file BEFORE extraction
  // (extracted previews lose EXIF orientation metadata)
  let orientation = 0;
  try {
    const tags = await exiftool.read(filePath);
    const rawOrientation = (tags as Record<string, unknown>).Orientation;
    // Orientation can be a number or a string like "Rotate 90 CW"
    if (typeof rawOrientation === 'number') {
      orientation = orientationToRotation(rawOrientation);
    } else if (typeof rawOrientation === 'string') {
      // Parse string orientation values like "Rotate 90 CW", "Horizontal (normal)"
      if (rawOrientation.includes('90') && rawOrientation.toLowerCase().includes('cw')) {
        orientation = 90;
      } else if (rawOrientation.includes('270') || (rawOrientation.includes('90') && rawOrientation.toLowerCase().includes('ccw'))) {
        orientation = 270;
      } else if (rawOrientation.includes('180')) {
        orientation = 180;
      }
    }
  } catch {
    // Ignore orientation read errors - will use 0 (no rotation)
  }

  // Extract the preview buffer
  const buffer = await extractPreviewBuffer(filePath, analysis.bestPreview.type);

  // Determine dimensions: use metadata if known, otherwise measure from buffer
  let width = analysis.bestPreview.width;
  let height = analysis.bestPreview.height;

  if (width === 0 || height === 0) {
    // Dimensions weren't in metadata - measure from extracted buffer
    const metadata = await sharp(buffer).metadata();
    width = metadata.width ?? 0;
    height = metadata.height ?? 0;

    if (width === 0 || height === 0) {
      throw new ShoemakerError(
        `Could not determine preview dimensions for ${filePath}`,
        ErrorCode.NO_PREVIEW,
        filePath
      );
    }
  }

  // Check if preview meets minimum size requirement
  if (minSize && width < minSize) {
    throw new ShoemakerError(
      `Best preview (${width}px) below minimum size (${minSize}px)`,
      ErrorCode.NO_PREVIEW,
      filePath
    );
  }

  return {
    buffer,
    tag: analysis.bestPreview.type,
    width,
    height,
    orientation,
  };
}

/**
 * Extract a specific preview tag as a buffer
 */
export async function extractPreviewBuffer(filePath: string, tag: PreviewTag): Promise<Buffer> {
  const fs = await import('fs/promises');
  const os = await import('os');
  const path = await import('path');
  const crypto = await import('crypto');

  // Validate temp directory is accessible
  const tempDir = os.tmpdir();
  try {
    await fs.access(tempDir, fs.constants.W_OK);
  } catch {
    throw new ShoemakerError(
      `Temp directory not writable: ${tempDir}`,
      ErrorCode.PERMISSION_DENIED,
      tempDir
    );
  }

  // Use crypto for unique temp filename to avoid race conditions
  const uniqueId = crypto.randomBytes(8).toString('hex');
  const tempFile = path.join(tempDir, `shoemaker-preview-${uniqueId}.jpg`);

  try {
    await exiftool.extractBinaryTag(tag, filePath, tempFile);

    // Read the extracted file
    const buffer = await fs.readFile(tempFile);

    if (buffer.length === 0) {
      throw new ShoemakerError(
        `Failed to extract ${tag} from ${filePath}`,
        ErrorCode.NO_PREVIEW,
        filePath
      );
    }

    return buffer;
  } catch (err) {
    if (err instanceof ShoemakerError) throw err;
    throw wrapError(err, filePath);
  } finally {
    // Clean up temp file
    try {
      await fs.unlink(tempFile);
    } catch {
      // Ignore cleanup errors
    }
  }
}

/**
 * Check if a file is a RAW format that needs preview extraction
 */
export function isRawFormat(filePath: string): boolean {
  const ext = filePath.toLowerCase().split('.').pop() ?? '';
  return (RAW_EXTENSIONS as readonly string[]).includes(ext);
}

/**
 * Check if a file is already a decoded format (JPEG, PNG, etc)
 */
export function isDecodedFormat(filePath: string): boolean {
  const ext = filePath.toLowerCase().split('.').pop() ?? '';
  return (DECODED_EXTENSIONS as readonly string[]).includes(ext);
}

/**
 * Check if a file is a video format
 */
export function isVideoFormat(filePath: string): boolean {
  const ext = filePath.toLowerCase().split('.').pop() ?? '';
  return (VIDEO_EXTENSIONS as readonly string[]).includes(ext);
}

/**
 * Shutdown ExifTool process (call on app exit)
 */
export async function shutdownExiftool(): Promise<void> {
  await exiftool.end();
}
