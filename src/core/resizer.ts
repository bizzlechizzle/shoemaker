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
 */
export async function generateThumbnails(
  input: Buffer | string,
  outputDir: string,
  stem: string,
  sizes: Record<string, SizeConfig>,
  options: { stripExif?: boolean; autoRotate?: boolean } = {}
): Promise<ThumbnailResult[]> {
  const results: ThumbnailResult[] = [];

  // Process sizes in order (largest first for better quality)
  const sortedSizes = Object.entries(sizes).sort((a, b) => b[1].width - a[1].width);

  for (const [sizeName, sizeConfig] of sortedSizes) {
    const ext = sizeConfig.format === 'jpeg' ? 'jpg' : sizeConfig.format;
    const outputPath = path.join(outputDir, `${stem}_${sizeName}_${sizeConfig.width}.${ext}`);

    const result = await generateThumbnail(input, outputPath, {
      width: sizeConfig.width,
      height: sizeConfig.height,
      format: sizeConfig.format,
      quality: sizeConfig.quality,
      allowUpscale: sizeConfig.allowUpscale,
      stripExif: options.stripExif,
      autoRotate: options.autoRotate,
    });

    results.push({
      ...result,
      size: sizeName,
    });
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
