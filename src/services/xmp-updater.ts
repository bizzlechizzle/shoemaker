/**
 * XMP Sidecar Updater
 *
 * Updates XMP sidecar files with thumbnail metadata.
 * Uses a custom shoemaker namespace for thumbnail tracking.
 */

import fs from 'fs/promises';
import path from 'path';
import { exiftool } from 'exiftool-vendored';
import type { ThumbnailResult, VideoInfo } from '../schemas/index.js';
import { wrapError } from '../core/errors.js';

export interface XmpUpdateData {
  thumbnails: ThumbnailResult[];
  method: 'extracted' | 'decoded' | 'direct' | 'video';
  videoInfo?: VideoInfo;
}

/**
 * Update XMP sidecar with thumbnail information
 */
export async function updateXmpSidecar(
  imagePath: string,
  data: XmpUpdateData
): Promise<void> {
  const xmpPath = getXmpPath(imagePath);

  try {
    // Build XMP metadata
    const thumbnailsJson = JSON.stringify(data.thumbnails.map(t => ({
      size: t.size,
      resolution: t.width,
      format: t.format,
      path: path.basename(t.path),
      bytes: t.bytes,
    })));

    // Write using exiftool with raw args for custom namespace
    // exiftool-vendored doesn't support custom namespaces in typed API
    await exiftool.write(xmpPath, {}, [
      '-overwrite_original',
      `-XMP-xmp:Label=shoemaker-managed`,
      `-XMP-dc:Description=Thumbnails: ${data.thumbnails.map(t => t.size).join(', ')}`,
    ]);

    // Store thumbnail data in a standard XMP field as JSON
    // Using dc:source which accepts arbitrary text
    const metadata: Record<string, unknown> = {
      generated: true,
      generatedAt: new Date().toISOString(),
      method: data.method,
      thumbnails: JSON.parse(thumbnailsJson),
    };

    // Add video metadata if present
    if (data.videoInfo) {
      metadata.video = {
        duration: data.videoInfo.duration,
        resolution: `${data.videoInfo.width}x${data.videoInfo.height}`,
        frameRate: data.videoInfo.frameRate,
        codec: data.videoInfo.codec,
        bitrate: data.videoInfo.bitrate,
        isInterlaced: data.videoInfo.isInterlaced,
        isHdr: data.videoInfo.isHdr,
        rotation: data.videoInfo.rotation,
        audio: data.videoInfo.audio,
      };
    }

    await exiftool.write(xmpPath, {}, [
      '-overwrite_original',
      `-XMP-dc:Source=shoemaker:${JSON.stringify(metadata)}`,
    ]);
  } catch (err) {
    throw wrapError(err, xmpPath);
  }
}

/**
 * Check if XMP sidecar indicates thumbnails already exist
 */
export async function hasExistingThumbnails(imagePath: string): Promise<boolean> {
  const xmpPath = getXmpPath(imagePath);

  try {
    await fs.access(xmpPath);
    const tags = await exiftool.read(xmpPath);
    const tagsRecord = tags as Record<string, unknown>;
    // Check if our shoemaker metadata exists in dc:Source
    const source = tagsRecord['Source'] as string | undefined;
    return source?.startsWith('shoemaker:') ?? false;
  } catch {
    return false;
  }
}

/**
 * Read thumbnail information from XMP sidecar
 */
export async function readThumbnailInfo(imagePath: string): Promise<{
  exists: boolean;
  generatedAt?: string;
  method?: string;
  thumbnails?: Array<{
    size: string;
    resolution: number;
    format: string;
    path: string;
    bytes: number;
  }>;
}> {
  const xmpPath = getXmpPath(imagePath);

  try {
    await fs.access(xmpPath);
    const tags = await exiftool.read(xmpPath);
    const tagsRecord = tags as Record<string, unknown>;

    // Check for shoemaker metadata in dc:Source
    const source = tagsRecord['Source'] as string | undefined;
    if (!source?.startsWith('shoemaker:')) {
      return { exists: false };
    }

    try {
      const jsonStr = source.substring('shoemaker:'.length);
      const metadata = JSON.parse(jsonStr) as {
        generated: boolean;
        generatedAt: string;
        method: string;
        thumbnails: Array<{
          size: string;
          resolution: number;
          format: string;
          path: string;
          bytes: number;
        }>;
      };

      return {
        exists: metadata.generated,
        generatedAt: metadata.generatedAt,
        method: metadata.method,
        thumbnails: metadata.thumbnails,
      };
    } catch (parseErr) {
      // Parse error - metadata corrupted, log warning for debugging
      console.warn(`Warning: Failed to parse thumbnail metadata in ${xmpPath}: ${(parseErr as Error).message}`);
      return { exists: false };
    }
  } catch {
    return { exists: false };
  }
}

/**
 * Clear thumbnail information from XMP sidecar
 */
export async function clearThumbnailInfo(imagePath: string): Promise<void> {
  const xmpPath = getXmpPath(imagePath);

  try {
    await fs.access(xmpPath);

    // Clear shoemaker metadata by removing the dc:Source and dc:Description fields
    await exiftool.write(xmpPath, {}, [
      '-overwrite_original',
      '-XMP-dc:Source=',
      '-XMP-dc:Description=',
      '-XMP-xmp:Label=',
    ]);
  } catch (err) {
    if ((err as NodeJS.ErrnoException).code === 'ENOENT') {
      return; // No XMP file, nothing to clear
    }
    throw wrapError(err, xmpPath);
  }
}

/**
 * Get XMP sidecar path for an image
 */
export function getXmpPath(imagePath: string): string {
  return `${imagePath}.xmp`;
}

/**
 * Check if XMP sidecar exists
 */
export async function xmpExists(imagePath: string): Promise<boolean> {
  try {
    await fs.access(getXmpPath(imagePath));
    return true;
  } catch {
    return false;
  }
}
