/**
 * XMP Sidecar Updater
 *
 * Updates XMP sidecar files with thumbnail metadata using proper `shoe:` namespace.
 * Preserves other namespaces (wnb:, vbuffet:, etc.) when writing.
 * Supports batch mode for efficient bulk updates.
 * Appends to shared custody chain for provenance tracking.
 */

import fs from 'fs/promises';
import path from 'path';
import { hostname, userInfo } from 'os';
import { exiftool } from 'exiftool-vendored';
import type { ThumbnailResult, VideoInfo, ProxyResult } from '../schemas/index.js';
import { wrapError } from '../core/errors.js';

// Namespace configuration
const NAMESPACE = 'shoe';
// Namespace URI for reference: http://shoemaker.dev/xmp/1.0/
const TOOL_NAME = 'shoemaker';
const SCHEMA_VERSION = 1;

// Get version from package.json
let TOOL_VERSION = '0.1.10';
try {
  // Will be set at build time
  TOOL_VERSION = process.env.npm_package_version || '0.1.10';
} catch {
  // Use default
}

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
 * Generate a unique event ID
 */
function generateEventId(): string {
  return `${Date.now().toString(36)}-${Math.random().toString(36).slice(2, 8)}`;
}

/**
 * Build custody event struct for exiftool
 */
function buildCustodyEvent(action: string, outcome: string, notes?: string): string {
  let eventStruct = `{EventID=${generateEventId()},` +
    `EventTimestamp=${new Date().toISOString()},` +
    `EventAction=${action},` +
    `EventOutcome=${outcome},` +
    `EventTool=${TOOL_NAME}/${TOOL_VERSION},` +
    `EventHost=${hostname()},` +
    `EventUser=${userInfo().username}`;

  if (notes) {
    // Escape special characters in notes
    const safeNotes = notes.replace(/[{}=,]/g, '_');
    eventStruct += `,EventNotes=${safeNotes}`;
  }
  eventStruct += '}';

  return eventStruct;
}

/**
 * Update XMP sidecar with thumbnail information using proper shoe: namespace
 */
export async function updateXmpSidecar(
  imagePath: string,
  data: XmpUpdateData
): Promise<void> {
  const xmpPath = getXmpPath(imagePath);

  try {
    // Read existing XMP to get current event count
    let eventCount = 0;
    try {
      const existing = await exiftool.read(xmpPath);
      const existingRecord = existing as Record<string, unknown>;
      eventCount = (existingRecord['EventCount'] as number) ?? 0;
    } catch {
      // XMP doesn't exist yet, that's fine
    }

    // Build thumbnail list for XMP
    const thumbnailsList = data.thumbnails.map(t =>
      `{Size=${t.size},Resolution=${t.width},Format=${t.format},` +
      `Path=${path.basename(t.path)},Bytes=${t.bytes}}`
    );

    // Build exiftool arguments for shoe: namespace
    const args: string[] = [
      '-overwrite_original',
      // Schema metadata
      `-XMP-${NAMESPACE}:SchemaVersion=${SCHEMA_VERSION}`,
      `-XMP-${NAMESPACE}:GeneratedAt=${new Date().toISOString()}`,
      `-XMP-${NAMESPACE}:Method=${data.method}`,
    ];

    // Add thumbnails (as structured list)
    for (const thumb of thumbnailsList) {
      args.push(`-XMP-${NAMESPACE}:Thumbnails+=${thumb}`);
    }

    // Add video metadata if present
    if (data.videoInfo) {
      args.push(
        `-XMP-${NAMESPACE}:VideoDuration=${data.videoInfo.duration}`,
        `-XMP-${NAMESPACE}:VideoResolution=${data.videoInfo.width}x${data.videoInfo.height}`,
        `-XMP-${NAMESPACE}:VideoFrameRate=${data.videoInfo.frameRate}`,
        `-XMP-${NAMESPACE}:VideoCodec=${data.videoInfo.codec}`,
      );
      if (data.videoInfo.bitrate) {
        args.push(`-XMP-${NAMESPACE}:VideoBitrate=${data.videoInfo.bitrate}`);
      }
      if (data.videoInfo.isHdr !== undefined) {
        args.push(`-XMP-${NAMESPACE}:VideoIsHdr=${data.videoInfo.isHdr}`);
      }
      if (data.videoInfo.isInterlaced !== undefined) {
        args.push(`-XMP-${NAMESPACE}:VideoIsInterlaced=${data.videoInfo.isInterlaced}`);
      }
      if (data.videoInfo.rotation) {
        args.push(`-XMP-${NAMESPACE}:VideoRotation=${data.videoInfo.rotation}`);
      }
    }

    // Add proxy metadata if present
    if (data.proxies && data.proxies.length > 0) {
      for (const proxy of data.proxies) {
        const proxyStruct = `{Size=${proxy.size},Resolution=${proxy.width}x${proxy.height},` +
          `Codec=${proxy.codec},Format=${proxy.format},` +
          `Path=${path.basename(proxy.path)},Bytes=${proxy.bytes},Bitrate=${proxy.bitrate}}`;
        args.push(`-XMP-${NAMESPACE}:Proxies+=${proxyStruct}`);
      }
    }

    // Add custody chain event (using shared wnb: namespace)
    const thumbnailSizes = data.thumbnails.map(t => t.size).join(', ');
    const custodyNotes = data.proxies?.length
      ? `Generated ${thumbnailSizes} thumbnails and ${data.proxies.length} proxies`
      : `Generated ${thumbnailSizes} thumbnails`;

    args.push(
      `-XMP-wnb:EventCount=${eventCount + 1}`,
      `-XMP-wnb:SidecarUpdated=${new Date().toISOString()}`,
      `-XMP-wnb:CustodyChain+=${buildCustodyEvent('thumbnail_generation', 'success', custodyNotes)}`,
    );

    // Also update dc:Description for compatibility with other tools
    args.push(`-XMP-dc:Description=Thumbnails: ${thumbnailSizes} (shoemaker/${TOOL_VERSION})`);

    await exiftool.write(xmpPath, {}, args);
  } catch (err) {
    throw wrapError(err, xmpPath);
  }
}

/**
 * Check if XMP sidecar indicates thumbnails already exist
 * Supports both old (dc:Source hack) and new (shoe: namespace) formats
 */
export async function hasExistingThumbnails(imagePath: string): Promise<boolean> {
  const xmpPath = getXmpPath(imagePath);

  try {
    await fs.access(xmpPath);
    const tags = await exiftool.read(xmpPath);
    const tagsRecord = tags as Record<string, unknown>;

    // Check new format first (shoe: namespace)
    if (tagsRecord['GeneratedAt'] !== undefined || tagsRecord['Method'] !== undefined) {
      return true;
    }

    // Fall back to old format (dc:Source hack) for backward compatibility
    const source = tagsRecord['Source'] as string | undefined;
    return source?.startsWith('shoemaker:') ?? false;
  } catch {
    return false;
  }
}

/**
 * Read thumbnail information from XMP sidecar
 * Supports both old (dc:Source hack) and new (shoe: namespace) formats
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
  format?: 'legacy' | 'v1';
}> {
  const xmpPath = getXmpPath(imagePath);

  try {
    await fs.access(xmpPath);
    const tags = await exiftool.read(xmpPath);
    const tagsRecord = tags as Record<string, unknown>;

    // Try new format first (shoe: namespace)
    const generatedAt = tagsRecord['GeneratedAt'] as string | undefined;
    const method = tagsRecord['Method'] as string | undefined;
    const thumbnails = tagsRecord['Thumbnails'] as unknown;

    if (generatedAt || method) {
      // Parse thumbnails from structured format
      let parsedThumbnails: Array<{
        size: string;
        resolution: number;
        format: string;
        path: string;
        bytes: number;
      }> = [];

      if (Array.isArray(thumbnails)) {
        parsedThumbnails = thumbnails.map((t: Record<string, unknown>) => ({
          size: String(t.Size || t.size || ''),
          resolution: Number(t.Resolution || t.resolution || 0),
          format: String(t.Format || t.format || ''),
          path: String(t.Path || t.path || ''),
          bytes: Number(t.Bytes || t.bytes || 0),
        }));
      }

      return {
        exists: true,
        generatedAt,
        method,
        thumbnails: parsedThumbnails,
        format: 'v1',
      };
    }

    // Fall back to old format (dc:Source hack)
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
        format: 'legacy',
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
 * Migrate from legacy format (dc:Source) to new format (shoe: namespace)
 */
export async function migrateToNewFormat(imagePath: string): Promise<boolean> {
  const info = await readThumbnailInfo(imagePath);

  if (!info.exists || info.format !== 'legacy') {
    return false; // Nothing to migrate
  }

  const xmpPath = getXmpPath(imagePath);

  try {
    // Read existing XMP to get event count
    const existing = await exiftool.read(xmpPath);
    const existingRecord = existing as Record<string, unknown>;
    const eventCount = (existingRecord['EventCount'] as number) ?? 0;

    // Build new format args
    const args: string[] = [
      '-overwrite_original',
      // Clear old format
      '-XMP-dc:Source=',
      // Write new format
      `-XMP-${NAMESPACE}:SchemaVersion=${SCHEMA_VERSION}`,
      `-XMP-${NAMESPACE}:GeneratedAt=${info.generatedAt || new Date().toISOString()}`,
      `-XMP-${NAMESPACE}:Method=${info.method || 'unknown'}`,
    ];

    // Add thumbnails in new format
    if (info.thumbnails) {
      for (const thumb of info.thumbnails) {
        const thumbStruct = `{Size=${thumb.size},Resolution=${thumb.resolution},` +
          `Format=${thumb.format},Path=${thumb.path},Bytes=${thumb.bytes}}`;
        args.push(`-XMP-${NAMESPACE}:Thumbnails+=${thumbStruct}`);
      }
    }

    // Add migration event to custody chain
    args.push(
      `-XMP-wnb:EventCount=${eventCount + 1}`,
      `-XMP-wnb:SidecarUpdated=${new Date().toISOString()}`,
      `-XMP-wnb:CustodyChain+=${buildCustodyEvent('metadata_migration', 'success', 'Migrated from dc:Source to shoe: namespace')}`,
    );

    await exiftool.write(xmpPath, {}, args);
    return true;
  } catch (err) {
    throw wrapError(err, xmpPath);
  }
}

/**
 * Clear thumbnail information from XMP sidecar
 * Clears both old and new formats
 */
export async function clearThumbnailInfo(imagePath: string): Promise<void> {
  const xmpPath = getXmpPath(imagePath);

  try {
    await fs.access(xmpPath);

    // Read existing XMP to get event count
    const existing = await exiftool.read(xmpPath);
    const existingRecord = existing as Record<string, unknown>;
    const eventCount = (existingRecord['EventCount'] as number) ?? 0;

    // Clear both old and new format fields
    await exiftool.write(xmpPath, {}, [
      '-overwrite_original',
      // Clear old format
      '-XMP-dc:Source=',
      '-XMP-xmp:Label=',
      // Clear new format
      `-XMP-${NAMESPACE}:SchemaVersion=`,
      `-XMP-${NAMESPACE}:GeneratedAt=`,
      `-XMP-${NAMESPACE}:Method=`,
      `-XMP-${NAMESPACE}:Thumbnails=`,
      `-XMP-${NAMESPACE}:VideoDuration=`,
      `-XMP-${NAMESPACE}:VideoResolution=`,
      `-XMP-${NAMESPACE}:VideoFrameRate=`,
      `-XMP-${NAMESPACE}:VideoCodec=`,
      `-XMP-${NAMESPACE}:VideoBitrate=`,
      `-XMP-${NAMESPACE}:VideoIsHdr=`,
      `-XMP-${NAMESPACE}:VideoIsInterlaced=`,
      `-XMP-${NAMESPACE}:VideoRotation=`,
      `-XMP-${NAMESPACE}:Proxies=`,
      // Add clear event to custody chain
      `-XMP-wnb:EventCount=${eventCount + 1}`,
      `-XMP-wnb:SidecarUpdated=${new Date().toISOString()}`,
      `-XMP-wnb:CustodyChain+=${buildCustodyEvent('thumbnail_removal', 'success', 'Cleared thumbnail metadata')}`,
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
