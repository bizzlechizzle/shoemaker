/**
 * FFprobe Service
 *
 * Extracts video metadata using FFprobe.
 * Provides duration, resolution, codec, and other video information.
 */

import { execFile } from 'child_process';
import { promisify } from 'util';
import type { VideoInfo } from '../schemas/index.js';
import { ShoemakerError, ErrorCode } from './errors.js';

const execFileAsync = promisify(execFile);

// Cache ffprobe availability
let ffprobeAvailable: boolean | null = null;
let ffprobeVersion: string | null = null;

/**
 * Check if FFprobe is available on the system
 */
export async function checkFfprobeAvailable(): Promise<boolean> {
  if (ffprobeAvailable !== null) {
    return ffprobeAvailable;
  }

  try {
    const { stdout, stderr } = await execFileAsync('ffprobe', ['-version'], {
      timeout: 5000,
    });
    const versionMatch = (stdout || stderr).match(/ffprobe version (\S+)/);
    ffprobeVersion = versionMatch?.[1] ?? 'unknown';
    ffprobeAvailable = true;
    return true;
  } catch {
    ffprobeAvailable = false;
    return false;
  }
}

/**
 * Get FFprobe version
 */
export async function getFfprobeVersion(): Promise<string | null> {
  if (ffprobeVersion !== null) {
    return ffprobeVersion;
  }
  await checkFfprobeAvailable();
  return ffprobeVersion;
}

/**
 * Check if FFmpeg is available on the system
 */
export async function checkFfmpegAvailable(): Promise<boolean> {
  try {
    await execFileAsync('ffmpeg', ['-version'], { timeout: 5000 });
    return true;
  } catch {
    return false;
  }
}

/**
 * Get FFmpeg version
 */
export async function getFfmpegVersion(): Promise<string | null> {
  try {
    const { stdout, stderr } = await execFileAsync('ffmpeg', ['-version'], {
      timeout: 5000,
    });
    const versionMatch = (stdout || stderr).match(/ffmpeg version (\S+)/);
    return versionMatch?.[1] ?? 'unknown';
  } catch {
    return null;
  }
}

interface FFprobeStream {
  codec_type?: string;
  codec_name?: string;
  width?: number;
  height?: number;
  r_frame_rate?: string;
  avg_frame_rate?: string;
  bit_rate?: string;
  channels?: number;
  sample_rate?: string;
  field_order?: string;
  color_transfer?: string;
  side_data_list?: Array<{ rotation?: number }>;
}

interface FFprobeFormat {
  duration?: string;
  bit_rate?: string;
  tags?: {
    creation_time?: string;
    rotate?: string;
  };
}

interface FFprobeOutput {
  streams?: FFprobeStream[];
  format?: FFprobeFormat;
}

/**
 * Probe a video file for metadata
 */
export async function probeVideo(filePath: string): Promise<VideoInfo> {
  const available = await checkFfprobeAvailable();
  if (!available) {
    throw new ShoemakerError(
      'FFprobe not found. Install FFmpeg to process video files.',
      ErrorCode.DECODER_NOT_AVAILABLE,
      filePath
    );
  }

  try {
    const { stdout } = await execFileAsync('ffprobe', [
      '-v', 'quiet',
      '-print_format', 'json',
      '-show_format',
      '-show_streams',
      filePath,
    ], {
      timeout: 30000,
      maxBuffer: 10 * 1024 * 1024,
    });

    const data: FFprobeOutput = JSON.parse(stdout);
    return parseFFprobeOutput(data, filePath);
  } catch (err) {
    if (err instanceof ShoemakerError) throw err;

    const message = err instanceof Error ? err.message : String(err);
    if (message.includes('ENOENT')) {
      throw new ShoemakerError(
        `Video file not found: ${filePath}`,
        ErrorCode.FILE_NOT_FOUND,
        filePath
      );
    }

    throw new ShoemakerError(
      `Failed to probe video: ${message}`,
      ErrorCode.DECODE_FAILED,
      filePath
    );
  }
}

/**
 * Parse FFprobe JSON output into VideoInfo
 */
function parseFFprobeOutput(data: FFprobeOutput, filePath: string): VideoInfo {
  const videoStream = data.streams?.find(s => s.codec_type === 'video');
  const audioStream = data.streams?.find(s => s.codec_type === 'audio');
  const format = data.format;

  if (!videoStream) {
    throw new ShoemakerError(
      'No video stream found in file',
      ErrorCode.CORRUPT_FILE,
      filePath
    );
  }

  // Parse frame rate (handles "30/1", "30000/1001", etc.)
  const frameRate = parseFrameRate(videoStream.avg_frame_rate ?? videoStream.r_frame_rate);

  // Parse duration from format or calculate from stream
  const duration = parseDuration(format?.duration);

  // Detect interlacing
  const isInterlaced = detectInterlacing(videoStream);

  // Detect HDR
  const isHdr = detectHdr(videoStream);

  // Get rotation from side data or tags
  const rotation = getRotation(videoStream, format);

  // Parse bitrate
  const bitrate = parseBitrate(videoStream.bit_rate ?? format?.bit_rate);

  const info: VideoInfo = {
    duration,
    width: videoStream.width ?? 0,
    height: videoStream.height ?? 0,
    frameRate,
    codec: videoStream.codec_name ?? 'unknown',
    bitrate,
    rotation,
    isInterlaced,
    isHdr,
    creationTime: format?.tags?.creation_time,
  };

  // Add audio info if present
  if (audioStream) {
    info.audio = {
      codec: audioStream.codec_name ?? 'unknown',
      channels: audioStream.channels ?? 2,
      sampleRate: parseInt(audioStream.sample_rate ?? '48000', 10),
    };
  }

  return info;
}

/**
 * Parse frame rate string to number
 */
function parseFrameRate(rateStr?: string): number {
  if (!rateStr || rateStr === '0/0') return 0;

  if (rateStr.includes('/')) {
    const parts = rateStr.split('/').map(Number);
    const num = parts[0] ?? 0;
    const den = parts[1] ?? 1;
    if (den === 0) return 0;
    return num / den;
  }

  return parseFloat(rateStr) || 0;
}

/**
 * Parse duration string to seconds
 */
function parseDuration(durationStr?: string): number {
  if (!durationStr) return 0;

  // Handle H:MM:SS format
  if (durationStr.includes(':')) {
    const parts = durationStr.split(':').map(Number);
    if (parts.length === 3) {
      const h = parts[0] ?? 0;
      const m = parts[1] ?? 0;
      const s = parts[2] ?? 0;
      return h * 3600 + m * 60 + s;
    }
    if (parts.length === 2) {
      const m = parts[0] ?? 0;
      const s = parts[1] ?? 0;
      return m * 60 + s;
    }
  }

  return parseFloat(durationStr) || 0;
}

/**
 * Parse bitrate string to number
 */
function parseBitrate(bitrateStr?: string): number | undefined {
  if (!bitrateStr) return undefined;
  const bitrate = parseInt(bitrateStr, 10);
  return isNaN(bitrate) ? undefined : bitrate;
}

/**
 * Detect if video is interlaced
 */
function detectInterlacing(stream: FFprobeStream): boolean {
  const fieldOrder = stream.field_order;
  if (!fieldOrder) return false;

  // Progressive video
  if (fieldOrder === 'progressive') return false;

  // Interlaced indicators
  return ['tt', 'bb', 'tb', 'bt'].includes(fieldOrder);
}

/**
 * Detect if video is HDR
 */
function detectHdr(stream: FFprobeStream): boolean {
  const colorTransfer = stream.color_transfer;
  if (!colorTransfer) return false;

  // HDR transfer functions
  return ['smpte2084', 'arib-std-b67', 'smpte428'].includes(colorTransfer);
}

/**
 * Get rotation angle from video metadata
 */
function getRotation(stream: FFprobeStream, format?: FFprobeFormat): number | undefined {
  // Check side data first (more reliable)
  const sideDataRotation = stream.side_data_list?.find(s => s.rotation !== undefined)?.rotation;
  if (sideDataRotation !== undefined) {
    return Math.abs(sideDataRotation);
  }

  // Check format tags
  if (format?.tags?.rotate) {
    return parseInt(format.tags.rotate, 10);
  }

  return undefined;
}

/**
 * Get video duration in seconds
 */
export async function getVideoDuration(filePath: string): Promise<number> {
  const info = await probeVideo(filePath);
  return info.duration;
}

/**
 * Check if a video has audio
 */
export async function hasAudio(filePath: string): Promise<boolean> {
  const info = await probeVideo(filePath);
  return info.audio !== undefined;
}

/**
 * Clear the FFprobe cache (useful for testing)
 */
export function clearFfprobeCache(): void {
  ffprobeAvailable = null;
  ffprobeVersion = null;
}
