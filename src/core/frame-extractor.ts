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

    // Consider black if average luminance is below 10 (out of 255)
    return avgLuminance < 10;
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
  // Skip first 5% and last 5% of video
  const safeStart = duration * 0.05;
  const safeEnd = duration * 0.95;
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
 * Uses a single FFmpeg call with select filter for efficiency
 */
export async function extractMultipleFrames(
  videoPath: string,
  percentages: number[],
  options: FrameExtractionOptions = {}
): Promise<ExtractedFrame[]> {
  const videoInfo = await probeVideo(videoPath);
  const filterChain = buildFilterChain(videoInfo, options);

  // Calculate frame numbers for each percentage
  const seekTimes = percentages.map(pct => calculateSeekTime(videoInfo.duration, pct));

  // Create temp directory for output
  const tempDir = path.join(os.tmpdir(), `shoemaker-${crypto.randomBytes(8).toString('hex')}`);
  await fs.mkdir(tempDir, { recursive: true });

  const frames: ExtractedFrame[] = [];

  try {
    // Extract frames one at a time for reliability
    // (batch extraction with select filter is faster but less reliable for seeking)
    for (let i = 0; i < seekTimes.length; i++) {
      const seekTime = seekTimes[i] ?? 0;
      const outputPath = path.join(tempDir, `frame_${i.toString().padStart(3, '0')}.jpg`);

      const args: string[] = [
        '-ss', String(seekTime),
        '-i', videoPath,
        '-vframes', '1',
      ];

      if (filterChain.filters.length > 0) {
        args.push('-vf', filterChain.filters.join(','));
      }

      args.push(
        '-q:v', '2',
        '-y',
        outputPath
      );

      await execFileAsync('ffmpeg', args, {
        timeout: 30000,
      });

      let buffer = await fs.readFile(outputPath);
      let actualPosition = seekTime;

      // Check for black frame and try adjacent positions
      if (options.skipBlackFrames !== false) {
        let attempts = 0;
        const maxAttempts = 3;
        const offsetStep = videoInfo.duration * 0.02; // 2% offset

        while (await isBlackFrame(buffer) && attempts < maxAttempts) {
          attempts++;
          const newTime = seekTime + (offsetStep * attempts);

          if (newTime >= videoInfo.duration * 0.95) break;

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

      frames.push({
        buffer,
        width: metadata.width ?? videoInfo.width,
        height: metadata.height ?? videoInfo.height,
        position: actualPosition,
        wasDeinterlaced: filterChain.wasDeinterlaced,
        wasRotated: filterChain.wasRotated,
      });
    }

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
 * Generate a timeline strip (horizontal concatenation of frames)
 */
export async function generateTimelineStrip(
  videoPath: string,
  frameCount: number,
  frameHeight: number,
  options: FrameExtractionOptions = {}
): Promise<Buffer> {
  // Calculate evenly distributed percentages
  const percentages: number[] = [];
  for (let i = 0; i < frameCount; i++) {
    percentages.push((i / (frameCount - 1)) * 100);
  }

  // Extract all frames
  const frames = await extractMultipleFrames(videoPath, percentages, options);

  if (frames.length === 0) {
    throw new ShoemakerError(
      'No frames extracted for timeline',
      ErrorCode.DECODE_FAILED,
      videoPath
    );
  }

  // Resize all frames to consistent height, maintaining aspect ratio
  const resizedFrames: { buffer: Buffer; width: number }[] = [];

  for (const frame of frames) {
    const aspectRatio = frame.width / frame.height;
    const newWidth = Math.round(frameHeight * aspectRatio);

    const resized = await sharp(frame.buffer)
      .resize(newWidth, frameHeight, { fit: 'fill' })
      .jpeg({ quality: 85 })
      .toBuffer();

    resizedFrames.push({ buffer: resized, width: newWidth });
  }

  // Calculate total strip dimensions
  const totalWidth = resizedFrames.reduce((sum, f) => sum + f.width, 0);

  // Create composite image
  const composites: sharp.OverlayOptions[] = [];
  let xOffset = 0;

  for (const frame of resizedFrames) {
    composites.push({
      input: frame.buffer,
      left: xOffset,
      top: 0,
    });
    xOffset += frame.width;
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
 */
export async function extractPosterFrame(
  videoPath: string,
  config: VideoConfig,
  options: FrameExtractionOptions = {}
): Promise<ExtractedFrame> {
  return extractFrameAtPercent(videoPath, config.posterPosition, {
    deinterlace: config.autoDeinterlace,
    rotate: config.autoRotate,
    skipBlackFrames: config.skipBlackFrames,
    hdrToneMap: config.hdrToneMap,
    ...options,
  });
}

/**
 * Extract preview frame (larger single frame for detail view)
 */
export async function extractPreviewFrame(
  videoPath: string,
  config: VideoConfig,
  options: FrameExtractionOptions = {}
): Promise<ExtractedFrame> {
  return extractFrameAtPercent(videoPath, config.previewPosition, {
    deinterlace: config.autoDeinterlace,
    rotate: config.autoRotate,
    skipBlackFrames: config.skipBlackFrames,
    hdrToneMap: config.hdrToneMap,
    ...options,
  });
}
