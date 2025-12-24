/**
 * XMP Sidecar Updater
 *
 * Updates XMP sidecar files with thumbnail metadata.
 * Uses a custom shoemaker namespace for thumbnail tracking.
 * Supports batch mode for efficient bulk updates.
 */

import fs from 'fs/promises';
import path from 'path';
import { exiftool } from 'exiftool-vendored';
import type { ThumbnailResult, VideoInfo, ProxyResult } from '../schemas/index.js';
import { wrapError } from '../core/errors.js';

// Batch queue for collecting XMP updates
interface PendingXmpUpdate {
  imagePath: string;
  data: XmpUpdateData;
}

const pendingUpdates: PendingXmpUpdate[] = [];
const BATCH_SIZE = 50;
let batchTimer: ReturnType<typeof setTimeout> | null = null;
const BATCH_TIMEOUT_MS = 5000; // 5 seconds

/**
 * Queue an XMP update for batch processing
 */
export function queueXmpUpdate(imagePath: string, data: XmpUpdateData): void {
  pendingUpdates.push({ imagePath, data });

  // If we've hit the batch size, flush immediately
  if (pendingUpdates.length >= BATCH_SIZE) {
    flushXmpUpdates();
  } else if (!batchTimer) {
    // Start a timer to flush after timeout
    batchTimer = setTimeout(() => {
      flushXmpUpdates();
    }, BATCH_TIMEOUT_MS);
  }
}

/**
 * Flush all pending XMP updates
 */
export async function flushXmpUpdates(): Promise<void> {
  if (batchTimer) {
    clearTimeout(batchTimer);
    batchTimer = null;
  }

  if (pendingUpdates.length === 0) {
    return;
  }

  // Take all pending updates
  const updates = pendingUpdates.splice(0, pendingUpdates.length);

  // Process all updates concurrently (ExifTool handles its own queuing)
  await Promise.all(
    updates.map(({ imagePath, data }) =>
      updateXmpSidecar(imagePath, data).catch(() => {
        // Log but don't fail the batch for individual errors
      })
    )
  );
}

/**
 * Get count of pending XMP updates
 */
export function getPendingXmpCount(): number {
  return pendingUpdates.length;
}

export interface XmpUpdateData {
  thumbnails: ThumbnailResult[];
  method: 'extracted' | 'decoded' | 'direct' | 'video';
  videoInfo?: VideoInfo;
  proxies?: ProxyResult[];
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

    // Add proxy metadata if present
    if (data.proxies && data.proxies.length > 0) {
      metadata.proxies = data.proxies.map(p => ({
        size: p.size,
        resolution: `${p.width}x${p.height}`,
        codec: p.codec,
        format: p.format,
        path: path.basename(p.path),
        bytes: p.bytes,
        bitrate: p.bitrate,
      }));
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
    } catch {
      // Parse error - metadata corrupted or invalid format
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
