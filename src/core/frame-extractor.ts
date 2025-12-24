/**
 * Frame Extractor
 *
 * Extracts frames from video files using FFmpeg.
 * Handles smart frame selection, deinterlacing, rotation, and black frame detection.
 */

import { execFile } from 'child_process';
import { promisify } from 'util';
import fs from 'fs/promises';
import path from 'path';
import os from 'os';
import crypto from 'crypto';
import sharp from 'sharp';
import { probeVideo } from './ffprobe.js';
import { ShoemakerError, ErrorCode } from './errors.js';
import type { VideoInfo, VideoConfig } from '../schemas/index.js';

const execFileAsync = promisify(execFile);

// Frame extraction constants
/** Average luminance threshold for black frame detection (0-255) */
const BLACK_FRAME_THRESHOLD = 10;
/** Skip first/last percentage of video to avoid credits/intros */
const SAFE_ZONE_MARGIN = 0.05;
/** Offset step for black frame retry (percentage of duration) */
const BLACK_FRAME_OFFSET_STEP = 0.02;

export interface ExtractedFrame {
  buffer: Buffer;
  width: number;
  height: number;
  position: number;
  wasDeinterlaced: boolean;
  wasRotated: boolean;
}

export interface FrameExtractionOptions {
  deinterlace?: boolean;
  rotate?: boolean;
  skipBlackFrames?: boolean;
  hdrToneMap?: boolean;
}

interface FilterChain {
  filters: string[];
  wasDeinterlaced: boolean;
  wasRotated: boolean;
}

/**
 * Check if a frame is mostly black (for skipping)
 */
async function isBlackFrame(buffer: Buffer): Promise<boolean> {
  try {
    const stats = await sharp(buffer).stats();

    // Check if all channels are very dark
    const avgLuminance = stats.channels.reduce((sum, ch) => sum + ch.mean, 0) / stats.channels.length;

    // Consider black if average luminance is below threshold
    return avgLuminance < BLACK_FRAME_THRESHOLD;
  } catch {
    return false;
  }
}

/**
 * Build FFmpeg filter chain based on video info and options
 */
function buildFilterChain(
  videoInfo: VideoInfo,
  options: FrameExtractionOptions
): FilterChain {
  const filters: string[] = [];
  let wasDeinterlaced = false;
  let wasRotated = false;

  // Deinterlacing (must come first)
  if (options.deinterlace !== false && videoInfo.isInterlaced) {
    filters.push('yadif=mode=0');
    wasDeinterlaced = true;
  }

  // HDR tone mapping
  if (options.hdrToneMap !== false && videoInfo.isHdr) {
    filters.push(
      'zscale=t=linear:npl=100',
      'format=gbrpf32le',
      'zscale=p=bt709',
      'tonemap=hable',
      'zscale=t=bt709:m=bt709:r=tv',
      'format=yuv420p'
    );
  }

  // Rotation (FFmpeg usually auto-rotates, but we track it)
  if (options.rotate !== false && videoInfo.rotation) {
    // FFmpeg handles rotation automatically with -autorotate (default)
    // We just track that it happened
    wasRotated = videoInfo.rotation !== 0;
  }

  return { filters, wasDeinterlaced, wasRotated };
}

/**
 * Calculate seek time from percentage
 */
function calculateSeekTime(duration: number, percentage: number): number {
  // Skip first/last portion of video to avoid credits/intros
  const safeStart = duration * SAFE_ZONE_MARGIN;
  const safeEnd = duration * (1 - SAFE_ZONE_MARGIN);
  const safeRange = safeEnd - safeStart;

  // Map percentage to safe range
  const normalizedPct = percentage / 100;
  return safeStart + (safeRange * normalizedPct);
}

/**
 * Extract a single frame from a video at a specific time
 */
export async function extractFrame(
  videoPath: string,
  timeSeconds: number,
  options: FrameExtractionOptions = {}
): Promise<ExtractedFrame> {
  const videoInfo = await probeVideo(videoPath);
  const filterChain = buildFilterChain(videoInfo, options);

  // Build FFmpeg command
  const args: string[] = [
    '-ss', String(timeSeconds),
    '-i', videoPath,
    '-vframes', '1',
  ];

  // Add filter chain if needed
  if (filterChain.filters.length > 0) {
    args.push('-vf', filterChain.filters.join(','));
  }

  // Output as JPEG to stdout
  args.push(
    '-f', 'image2pipe',
    '-vcodec', 'mjpeg',
    '-q:v', '2',
    'pipe:1'
  );

  try {
    const { stdout } = await execFileAsync('ffmpeg', args, {
      encoding: 'buffer',
      maxBuffer: 50 * 1024 * 1024,
      timeout: 30000,
    });

    if (!stdout || stdout.length === 0) {
      throw new ShoemakerError(
        `No frame extracted at ${timeSeconds}s`,
        ErrorCode.DECODE_FAILED,
        videoPath
      );
    }

    // Get frame dimensions
    const metadata = await sharp(stdout).metadata();

    return {
      buffer: stdout,
      width: metadata.width ?? videoInfo.width,
      height: metadata.height ?? videoInfo.height,
      position: timeSeconds,
      wasDeinterlaced: filterChain.wasDeinterlaced,
      wasRotated: filterChain.wasRotated,
    };
  } catch (err) {
    if (err instanceof ShoemakerError) throw err;

    throw new ShoemakerError(
      `Failed to extract frame: ${(err as Error).message}`,
      ErrorCode.DECODE_FAILED,
      videoPath
    );
  }
}

/**
 * Extract a frame at a percentage through the video
 */
export async function extractFrameAtPercent(
  videoPath: string,
  percentage: number,
  options: FrameExtractionOptions = {}
): Promise<ExtractedFrame> {
  const videoInfo = await probeVideo(videoPath);
  const seekTime = calculateSeekTime(videoInfo.duration, percentage);
  return extractFrame(videoPath, seekTime, options);
}

/**
 * Extract multiple frames at specified percentages
 * Uses batch extraction with FFmpeg for efficiency when possible
 */
export async function extractMultipleFrames(
  videoPath: string,
  percentages: number[],
  options: FrameExtractionOptions = {},
  videoInfo?: VideoInfo
): Promise<ExtractedFrame[]> {
  // Use provided videoInfo or probe (allows caller to avoid re-probe)
  const info = videoInfo ?? await probeVideo(videoPath);
  const filterChain = buildFilterChain(info, options);

  // Calculate seek times for each percentage
  const seekTimes = percentages.map(pct => calculateSeekTime(info.duration, pct));

  // Create temp directory for output
  const tempDir = path.join(os.tmpdir(), `shoemaker-${crypto.randomBytes(8).toString('hex')}`);
  await fs.mkdir(tempDir, { recursive: true });

  const frames: ExtractedFrame[] = [];

  try {
    // Try batch extraction first (faster), fall back to sequential if it fails
    const batchSuccess = await tryBatchExtraction(
      videoPath, seekTimes, tempDir, filterChain, info, options
    );

    if (batchSuccess) {
      // Read batch-extracted frames
      for (let i = 0; i < seekTimes.length; i++) {
        const outputPath = path.join(tempDir, `frame_${i.toString().padStart(3, '0')}.jpg`);
        try {
          const buffer = await fs.readFile(outputPath);
          if (buffer.length > 0) {
            const metadata = await sharp(buffer).metadata();
            frames.push({
              buffer,
              width: metadata.width ?? info.width,
              height: metadata.height ?? info.height,
              position: seekTimes[i] ?? 0,
              wasDeinterlaced: filterChain.wasDeinterlaced,
              wasRotated: filterChain.wasRotated,
            });
          }
        } catch {
          // Frame not extracted, will be handled below
        }
      }
    }

    // Fill in any missing frames with sequential extraction
    if (frames.length < seekTimes.length) {
      const extractedPositions = new Set(frames.map(f => f.position));
      for (let i = 0; i < seekTimes.length; i++) {
        const seekTime = seekTimes[i] ?? 0;
        if (extractedPositions.has(seekTime)) continue;

        const frame = await extractFrameWithBlackSkip(
          videoPath, seekTime, tempDir, i, filterChain, info, options
        );
        if (frame) {
          frames.push(frame);
        }
      }
    }

    // Sort frames by position to maintain order
    frames.sort((a, b) => a.position - b.position);

    return frames;
  } finally {
    // Cleanup temp directory
    try {
      await fs.rm(tempDir, { recursive: true, force: true });
    } catch {
      // Ignore cleanup errors
    }
  }
}

/**
 * Try batch extraction using FFmpeg select filter (faster but less reliable)
 */
async function tryBatchExtraction(
  videoPath: string,
  seekTimes: number[],
  tempDir: string,
  filterChain: FilterChain,
  videoInfo: VideoInfo,
  _options: FrameExtractionOptions
): Promise<boolean> {
  try {
    // Build select filter expression for all timestamps
    // Using time-based selection: select='gte(t,T1)*lt(t,T1+0.1)+gte(t,T2)*lt(t,T2+0.1)+...'
    const selectParts = seekTimes.map(t => `gte(t,${t.toFixed(3)})*lt(t,${t.toFixed(3) + 0.1})`);
    const selectExpr = selectParts.join('+');

    // Build complete filter chain
    const filters: string[] = [];

    // Add existing filters (deinterlace, HDR, etc.)
    if (filterChain.filters.length > 0) {
      filters.push(...filterChain.filters);
    }

    // Add frame selection
    filters.push(`select='${selectExpr}'`);
    filters.push('setpts=N/FRAME_RATE/TB');

    const args: string[] = [
      '-i', videoPath,
      '-vf', filters.join(','),
      '-vsync', '0',
      '-q:v', '2',
      '-frame_pts', '1',
      path.join(tempDir, 'frame_%03d.jpg'),
    ];

    // Use adaptive timeout based on video duration
    const timeout = Math.max(30000, Math.min(videoInfo.duration * 500, 120000));

    await execFileAsync('ffmpeg', args, {
      timeout,
      maxBuffer: 50 * 1024 * 1024,
    });

    // Verify we got at least some frames
    const files = await fs.readdir(tempDir);
    return files.filter(f => f.startsWith('frame_')).length > 0;
  } catch {
    // Batch extraction failed, will fall back to sequential
    return false;
  }
}

/**
 * Extract a single frame with black frame skip logic
 */
async function extractFrameWithBlackSkip(
  videoPath: string,
  seekTime: number,
  tempDir: string,
  index: number,
  filterChain: FilterChain,
  videoInfo: VideoInfo,
  options: FrameExtractionOptions
): Promise<ExtractedFrame | null> {
  const outputPath = path.join(tempDir, `frame_seq_${index.toString().padStart(3, '0')}.jpg`);

  // Use keyframe-aware seeking for faster extraction
  const args: string[] = [
    '-ss', String(seekTime),
    '-i', videoPath,
    '-vframes', '1',
  ];

  if (filterChain.filters.length > 0) {
    args.push('-vf', filterChain.filters.join(','));
  }

  args.push('-q:v', '2', '-y', outputPath);

  try {
    await execFileAsync('ffmpeg', args, { timeout: 30000 });

    let buffer = await fs.readFile(outputPath);
    let actualPosition = seekTime;

    // Check for black frame and try adjacent positions
    if (options.skipBlackFrames !== false) {
      let attempts = 0;
      const maxAttempts = 3;
      const offsetStep = videoInfo.duration * BLACK_FRAME_OFFSET_STEP;

      while (await isBlackFrame(buffer) && attempts < maxAttempts) {
        attempts++;
        const newTime = seekTime + (offsetStep * attempts);

        if (newTime >= videoInfo.duration * (1 - SAFE_ZONE_MARGIN)) break;

        await execFileAsync('ffmpeg', [
          '-ss', String(newTime),
          '-i', videoPath,
          '-vframes', '1',
          ...(filterChain.filters.length > 0 ? ['-vf', filterChain.filters.join(',')] : []),
          '-q:v', '2',
          '-y',
          outputPath,
        ], { timeout: 30000 });

        buffer = await fs.readFile(outputPath);
        actualPosition = newTime;
      }
    }

    const metadata = await sharp(buffer).metadata();

    return {
      buffer,
      width: metadata.width ?? videoInfo.width,
      height: metadata.height ?? videoInfo.height,
      position: actualPosition,
      wasDeinterlaced: filterChain.wasDeinterlaced,
      wasRotated: filterChain.wasRotated,
    };
  } catch {
    return null;
  }
}

/**
 * Generate a timeline strip (horizontal concatenation of frames)
 * Accepts optional videoInfo to avoid re-probing
 */
export async function generateTimelineStrip(
  videoPath: string,
  frameCount: number,
  frameHeight: number,
  options: FrameExtractionOptions = {},
  videoInfo?: VideoInfo
): Promise<Buffer> {
  // Calculate evenly distributed percentages
  const percentages: number[] = [];
  for (let i = 0; i < frameCount; i++) {
    percentages.push((i / (frameCount - 1)) * 100);
  }

  // Extract all frames (pass videoInfo to avoid re-probe)
  const frames = await extractMultipleFrames(videoPath, percentages, options, videoInfo);

  if (frames.length === 0) {
    throw new ShoemakerError(
      'No frames extracted for timeline',
      ErrorCode.DECODE_FAILED,
      videoPath
    );
  }

  // Process frames sequentially to optimize memory
  // Calculate dimensions first, then process one at a time
  const frameDimensions: { width: number }[] = [];
  const resizedBuffers: Buffer[] = [];

  for (const frame of frames) {
    const aspectRatio = frame.width / frame.height;
    const newWidth = Math.round(frameHeight * aspectRatio);
    frameDimensions.push({ width: newWidth });

    const resized = await sharp(frame.buffer)
      .resize(newWidth, frameHeight, { fit: 'fill' })
      .jpeg({ quality: 85 })
      .toBuffer();

    resizedBuffers.push(resized);

    // Release original frame buffer immediately for GC
    frame.buffer = Buffer.alloc(0);
  }

  // Calculate total strip dimensions
  const totalWidth = frameDimensions.reduce((sum, f) => sum + f.width, 0);

  // Create composite image with all frames
  const composites: sharp.OverlayOptions[] = [];
  let xOffset = 0;

  for (let i = 0; i < resizedBuffers.length; i++) {
    composites.push({
      input: resizedBuffers[i],
      left: xOffset,
      top: 0,
    });
    xOffset += frameDimensions[i]?.width ?? 0;
  }

  // Create blank canvas and composite frames
  const strip = await sharp({
    create: {
      width: totalWidth,
      height: frameHeight,
      channels: 3,
      background: { r: 0, g: 0, b: 0 },
    },
  })
    .composite(composites)
    .jpeg({ quality: 85 })
    .toBuffer();

  return strip;
}

/**
 * Extract poster frame (best single frame for thumbnail)
 * Accepts optional videoInfo to avoid re-probing
 */
export async function extractPosterFrame(
  videoPath: string,
  config: VideoConfig,
  options: FrameExtractionOptions = {},
  videoInfo?: VideoInfo
): Promise<ExtractedFrame> {
  const info = videoInfo ?? await probeVideo(videoPath);
  const seekTime = calculateSeekTime(info.duration, config.posterPosition);
  const filterChain = buildFilterChain(info, {
    deinterlace: config.autoDeinterlace,
    rotate: config.autoRotate,
    skipBlackFrames: config.skipBlackFrames,
    hdrToneMap: config.hdrToneMap,
    ...options,
  });

  // Build FFmpeg command
  const args: string[] = [
    '-ss', String(seekTime),
    '-i', videoPath,
    '-vframes', '1',
  ];

  if (filterChain.filters.length > 0) {
    args.push('-vf', filterChain.filters.join(','));
  }

  args.push('-f', 'image2pipe', '-vcodec', 'mjpeg', '-q:v', '2', 'pipe:1');

  try {
    const { stdout } = await execFileAsync('ffmpeg', args, {
      encoding: 'buffer',
      maxBuffer: 50 * 1024 * 1024,
      timeout: 30000,
    });

    if (!stdout || stdout.length === 0) {
      throw new ShoemakerError(
        `No poster frame extracted at ${seekTime}s`,
        ErrorCode.DECODE_FAILED,
        videoPath
      );
    }

    const metadata = await sharp(stdout).metadata();

    return {
      buffer: stdout,
      width: metadata.width ?? info.width,
      height: metadata.height ?? info.height,
      position: seekTime,
      wasDeinterlaced: filterChain.wasDeinterlaced,
      wasRotated: filterChain.wasRotated,
    };
  } catch (err) {
    if (err instanceof ShoemakerError) throw err;
    throw new ShoemakerError(
      `Failed to extract poster frame: ${(err as Error).message}`,
      ErrorCode.DECODE_FAILED,
      videoPath
    );
  }
}

/**
 * Extract preview frame (larger single frame for detail view)
 * Accepts optional videoInfo to avoid re-probing
 */
export async function extractPreviewFrame(
  videoPath: string,
  config: VideoConfig,
  options: FrameExtractionOptions = {},
  videoInfo?: VideoInfo
): Promise<ExtractedFrame> {
  const info = videoInfo ?? await probeVideo(videoPath);
  const seekTime = calculateSeekTime(info.duration, config.previewPosition);
  const filterChain = buildFilterChain(info, {
    deinterlace: config.autoDeinterlace,
    rotate: config.autoRotate,
    skipBlackFrames: config.skipBlackFrames,
    hdrToneMap: config.hdrToneMap,
    ...options,
  });

  // Build FFmpeg command
  const args: string[] = [
    '-ss', String(seekTime),
    '-i', videoPath,
    '-vframes', '1',
  ];

  if (filterChain.filters.length > 0) {
    args.push('-vf', filterChain.filters.join(','));
  }

  args.push('-f', 'image2pipe', '-vcodec', 'mjpeg', '-q:v', '2', 'pipe:1');

  try {
    const { stdout } = await execFileAsync('ffmpeg', args, {
      encoding: 'buffer',
      maxBuffer: 50 * 1024 * 1024,
      timeout: 30000,
    });

    if (!stdout || stdout.length === 0) {
      throw new ShoemakerError(
        `No preview frame extracted at ${seekTime}s`,
        ErrorCode.DECODE_FAILED,
        videoPath
      );
    }

    const metadata = await sharp(stdout).metadata();

    return {
      buffer: stdout,
      width: metadata.width ?? info.width,
      height: metadata.height ?? info.height,
      position: seekTime,
      wasDeinterlaced: filterChain.wasDeinterlaced,
      wasRotated: filterChain.wasRotated,
    };
  } catch (err) {
    if (err instanceof ShoemakerError) throw err;
    throw new ShoemakerError(
      `Failed to extract preview frame: ${(err as Error).message}`,
      ErrorCode.DECODE_FAILED,
      videoPath
    );
  }
}
