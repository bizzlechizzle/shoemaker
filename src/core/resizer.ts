/**
 * Image Resizer
 *
 * Resizes images using Sharp with proper color space handling.
 * All outputs are converted to sRGB with embedded ICC profile.
 */

import sharp from 'sharp';
import path from 'path';
import fs from 'fs/promises';
import type { SizeConfig, ThumbnailResult } from '../schemas/index.js';
import { wrapError } from './errors.js';

export interface ResizeOptions {
  width: number;
  height?: number;
  format: 'webp' | 'jpeg' | 'png' | 'avif';
  quality: number;
  allowUpscale?: boolean;
  stripExif?: boolean;
  autoRotate?: boolean;
}

export interface ResizeResult {
  buffer: Buffer;
  width: number;
  height: number;
  format: string;
  bytes: number;
}

/**
 * Resize an image buffer to specified dimensions
 */
export async function resizeImage(
  input: Buffer | string,
  options: ResizeOptions
): Promise<ResizeResult> {
  try {
    let pipeline = sharp(input);

    // Auto-rotate based on EXIF orientation (default: true)
    if (options.autoRotate !== false) {
      pipeline = pipeline.rotate();
    }

    // Resize with aspect ratio preservation
    pipeline = pipeline.resize(options.width, options.height ?? options.width, {
      fit: 'inside',
      withoutEnlargement: options.allowUpscale !== true,
    });

    // Convert to sRGB color space
    pipeline = pipeline.toColorspace('srgb');

    // Set output format with quality
    switch (options.format) {
      case 'webp':
        pipeline = pipeline.webp({ quality: options.quality });
        break;
      case 'jpeg':
        pipeline = pipeline.jpeg({ quality: options.quality, mozjpeg: true });
        break;
      case 'png':
        pipeline = pipeline.png({ compressionLevel: 9 });
        break;
      case 'avif':
        pipeline = pipeline.avif({ quality: options.quality });
        break;
    }

    // Embed sRGB ICC profile, strip other metadata if requested
    pipeline = pipeline.withMetadata({
      icc: 'srgb',
    });

    // Process and get buffer + metadata
    const buffer = await pipeline.toBuffer();
    const metadata = await sharp(buffer).metadata();

    return {
      buffer,
      width: metadata.width ?? options.width,
      height: metadata.height ?? options.height ?? options.width,
      format: options.format,
      bytes: buffer.length,
    };
  } catch (err) {
    const filePath = typeof input === 'string' ? input : undefined;
    throw wrapError(err, filePath);
  }
}

/**
 * Generate a thumbnail and save to disk
 */
export async function generateThumbnail(
  input: Buffer | string,
  outputPath: string,
  options: ResizeOptions
): Promise<ThumbnailResult> {
  const result = await resizeImage(input, options);

  // Ensure output directory exists
  await fs.mkdir(path.dirname(outputPath), { recursive: true });

  // Write to disk
  await fs.writeFile(outputPath, result.buffer);

  return {
    size: '', // Will be set by caller
    width: result.width,
    height: result.height,
    format: result.format,
    path: outputPath,
    bytes: result.bytes,
  };
}

/**
 * Generate multiple thumbnail sizes from a single source
 * Uses Sharp .clone() to share the decoded input across all sizes for efficiency
 */
export async function generateThumbnails(
  input: Buffer | string,
  outputDir: string,
  stem: string,
  sizes: Record<string, SizeConfig>,
  options: { stripExif?: boolean; autoRotate?: boolean; rotationAngle?: number } = {}
): Promise<ThumbnailResult[]> {
  const results: ThumbnailResult[] = [];

  // Ensure output directory exists (once for all sizes)
  await fs.mkdir(outputDir, { recursive: true });

  // Create base pipeline with common operations (decode once, reuse for all sizes)
  let basePipeline = sharp(input);

  // Handle rotation:
  // 1. If explicit rotationAngle is provided (from source file's EXIF for extracted previews), use it
  // 2. Otherwise, if autoRotate is enabled, let Sharp read EXIF from the buffer
  // 3. If autoRotate is false, don't rotate
  if (typeof options.rotationAngle === 'number' && options.rotationAngle !== 0) {
    // Explicit rotation angle (for extracted previews that lost EXIF orientation)
    basePipeline = basePipeline.rotate(options.rotationAngle);
  } else if (options.autoRotate !== false) {
    // Let Sharp read EXIF orientation from buffer (for direct files like JPEG)
    basePipeline = basePipeline.rotate();
  }

  // Convert to sRGB color space (common for all outputs)
  basePipeline = basePipeline.toColorspace('srgb');

  // Process sizes in order (largest first for better quality)
  const sortedSizes = Object.entries(sizes).sort((a, b) => b[1].width - a[1].width);

  // Process all sizes using cloned pipelines (shares decoded input)
  const processPromises = sortedSizes.map(async ([sizeName, sizeConfig]) => {
    const ext = sizeConfig.format === 'jpeg' ? 'jpg' : sizeConfig.format;
    const outputPath = path.join(outputDir, `${stem}_${sizeName}_${sizeConfig.width}.${ext}`);

    // Clone the base pipeline (shares decoded input buffer)
    let pipeline = basePipeline.clone();

    // Resize with aspect ratio preservation
    pipeline = pipeline.resize(sizeConfig.width, sizeConfig.height ?? sizeConfig.width, {
      fit: 'inside',
      withoutEnlargement: sizeConfig.allowUpscale !== true,
    });

    // Set output format with quality
    switch (sizeConfig.format) {
      case 'webp':
        pipeline = pipeline.webp({ quality: sizeConfig.quality });
        break;
      case 'jpeg':
        pipeline = pipeline.jpeg({ quality: sizeConfig.quality, mozjpeg: true });
        break;
      case 'png':
        pipeline = pipeline.png({ compressionLevel: 9 });
        break;
      case 'avif':
        pipeline = pipeline.avif({ quality: sizeConfig.quality });
        break;
    }

    // Embed sRGB ICC profile
    pipeline = pipeline.withMetadata({ icc: 'srgb' });

    // Process and write to disk
    const buffer = await pipeline.toBuffer();
    await fs.writeFile(outputPath, buffer);

    const metadata = await sharp(buffer).metadata();

    return {
      size: sizeName,
      width: metadata.width ?? sizeConfig.width,
      height: metadata.height ?? sizeConfig.height ?? sizeConfig.width,
      format: sizeConfig.format,
      path: outputPath,
      bytes: buffer.length,
    } as ThumbnailResult;
  });

  // Wait for all sizes to complete
  const processedResults = await Promise.all(processPromises);

  // Sort results to match original order (largest first)
  for (const [sizeName] of sortedSizes) {
    const result = processedResults.find(r => r.size === sizeName);
    if (result) {
      results.push(result);
    }
  }

  return results;
}

/**
 * Get image metadata without processing
 */
export async function getImageMetadata(input: Buffer | string): Promise<{
  width: number;
  height: number;
  format: string;
  colorSpace: string;
  hasAlpha: boolean;
}> {
  try {
    const metadata = await sharp(input).metadata();
    return {
      width: metadata.width ?? 0,
      height: metadata.height ?? 0,
      format: metadata.format ?? 'unknown',
      colorSpace: metadata.space ?? 'unknown',
      hasAlpha: metadata.hasAlpha ?? false,
    };
  } catch (err) {
    const filePath = typeof input === 'string' ? input : undefined;
    throw wrapError(err, filePath);
  }
}

/**
 * Check Sharp/libvips capabilities
 */
export function getSharpCapabilities(): {
  formats: { input: string[]; output: string[] };
  version: string;
} {
  const vipsVersion = sharp.versions?.vips ?? 'unknown';

  return {
    formats: {
      input: ['jpeg', 'png', 'webp', 'gif', 'avif', 'tiff', 'heif', 'raw'],
      output: ['jpeg', 'png', 'webp', 'avif', 'tiff'],
    },
    version: vipsVersion,
  };
}
