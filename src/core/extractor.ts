/**
 * Preview Extractor
 *
 * Extracts embedded previews from RAW files using ExifTool.
 * Analyzes available previews and selects the best one.
 */

import { exiftool } from 'exiftool-vendored';
import type { PreviewAnalysis, PreviewInfo } from '../schemas/index.js';
import { DEFAULT_MIN_PREVIEW_SIZE, RAW_EXTENSIONS, DECODED_EXTENSIONS } from '../schemas/index.js';
import { ShoemakerError, ErrorCode, wrapError } from './errors.js';

// Preview tag priority order (best to worst)
const PREVIEW_TAGS = ['JpgFromRaw', 'PreviewImage', 'OtherImage', 'ThumbnailImage'] as const;
type PreviewTag = typeof PREVIEW_TAGS[number];

/**
 * Type guard: check if value is a valid number
 */
function isNumber(value: unknown): value is number {
  return typeof value === 'number' && !isNaN(value) && isFinite(value);
}

/**
 * Analyze embedded previews in an image file
 */
export async function analyzeEmbeddedPreviews(filePath: string): Promise<PreviewAnalysis> {
  try {
    const tags = await exiftool.read(filePath);

    const jpgFromRaw = extractPreviewInfo(tags, 'JpgFromRaw');
    const previewImage = extractPreviewInfo(tags, 'PreviewImage');
    const otherImage = extractPreviewInfo(tags, 'OtherImage');
    const thumbnailImage = extractPreviewInfo(tags, 'ThumbnailImage');

    // Find best preview (largest that exists)
    let bestPreview: PreviewAnalysis['bestPreview'] = null;
    for (const tag of PREVIEW_TAGS) {
      const info = { JpgFromRaw: jpgFromRaw, PreviewImage: previewImage, OtherImage: otherImage, ThumbnailImage: thumbnailImage }[tag];
      if (info.exists && info.width && info.height) {
        if (!bestPreview || (info.width > bestPreview.width)) {
          bestPreview = { type: tag, width: info.width, height: info.height };
        }
      }
    }

    // Determine if RAW decode is needed (best preview < minimum size)
    const needsRawDecode = !bestPreview || bestPreview.width < DEFAULT_MIN_PREVIEW_SIZE;

    return {
      filePath,
      jpgFromRaw,
      previewImage,
      otherImage,
      thumbnailImage,
      bestPreview,
      needsRawDecode,
    };
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
 */
export async function extractBestPreview(filePath: string, minSize?: number): Promise<{ buffer: Buffer; tag: PreviewTag; width: number; height: number }> {
  const analysis = await analyzeEmbeddedPreviews(filePath);

  if (!analysis.bestPreview) {
    throw new ShoemakerError(
      `No embedded preview found in ${filePath}`,
      ErrorCode.NO_PREVIEW,
      filePath
    );
  }

  // Check if preview meets minimum size requirement
  if (minSize && analysis.bestPreview.width < minSize) {
    throw new ShoemakerError(
      `Best preview (${analysis.bestPreview.width}px) below minimum size (${minSize}px)`,
      ErrorCode.NO_PREVIEW,
      filePath
    );
  }

  const buffer = await extractPreviewBuffer(filePath, analysis.bestPreview.type);

  return {
    buffer,
    tag: analysis.bestPreview.type,
    width: analysis.bestPreview.width,
    height: analysis.bestPreview.height,
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
 * Shutdown ExifTool process (call on app exit)
 */
export async function shutdownExiftool(): Promise<void> {
  await exiftool.end();
}
