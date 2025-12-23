/**
 * Video Proxy Generator
 *
 * Generates lower-resolution proxy videos for editing workflows.
 * Supports hardware acceleration on macOS (VideoToolbox), NVIDIA (NVENC),
 * AMD/Intel Linux (VAAPI), and Intel QuickSync (QSV).
 */

import { execFile, spawn } from 'child_process';
import { promisify } from 'util';
import fs from 'fs/promises';
import path from 'path';
import { probeVideo } from './ffprobe.js';
import { ShoemakerError, ErrorCode } from './errors.js';
import type { ProxyConfig, ProxySizeConfig, ProxyResult, VideoInfo } from '../schemas/index.js';

const execFileAsync = promisify(execFile);

// Hardware acceleration encoder mappings
const HW_ENCODERS: Record<string, Record<string, string>> = {
  videotoolbox: {
    h264: 'h264_videotoolbox',
    h265: 'hevc_videotoolbox',
    prores: 'prores_videotoolbox',
  },
  nvenc: {
    h264: 'h264_nvenc',
    h265: 'hevc_nvenc',
  },
  vaapi: {
    h264: 'h264_vaapi',
    h265: 'hevc_vaapi',
  },
  qsv: {
    h264: 'h264_qsv',
    h265: 'hevc_qsv',
  },
  none: {
    h264: 'libx264',
    h265: 'libx265',
    prores: 'prores_ks',
  },
};

// Cache for available encoders
let encoderCache: Set<string> | null = null;

export interface ProxyGeneratorOptions {
  config: ProxyConfig;
  videoInfo: VideoInfo;
  outputDir: string;
  stem: string;
  onProgress?: (percent: number, fps: number, size: string) => void;
}

/**
 * Detect available hardware encoders
 */
export async function detectAvailableEncoders(): Promise<Set<string>> {
  if (encoderCache) {
    return encoderCache;
  }

  const available = new Set<string>();

  try {
    const { stdout } = await execFileAsync('ffmpeg', ['-encoders'], {
      timeout: 10000,
    });

    // Parse encoder list
    const lines = stdout.split('\n');
    for (const line of lines) {
      const match = line.match(/^\s*V[.F]*[.S]*[.X]*[.B]*[.D]*\s+(\S+)/);
      if (match?.[1]) {
        available.add(match[1]);
      }
    }
  } catch {
    // If detection fails, assume basic encoders
    available.add('libx264');
    available.add('libx265');
  }

  encoderCache = available;
  return available;
}

/**
 * Select best encoder based on config and availability
 */
export async function selectEncoder(
  codec: 'h264' | 'h265' | 'prores',
  hwAccel: ProxyConfig['hwAccel']
): Promise<{ encoder: string; hwType: string }> {
  const available = await detectAvailableEncoders();

  // If auto, try hardware encoders in order of preference
  const hwPriority = hwAccel === 'auto'
    ? ['videotoolbox', 'nvenc', 'qsv', 'vaapi', 'none']
    : [hwAccel];

  for (const hw of hwPriority) {
    const encoderMap = HW_ENCODERS[hw];
    if (!encoderMap) continue;

    const encoder = encoderMap[codec];
    if (encoder && available.has(encoder)) {
      return { encoder, hwType: hw };
    }
  }

  // Fallback to software
  const softwareEncoders = HW_ENCODERS.none;
  const softwareEncoder = softwareEncoders?.[codec];
  if (softwareEncoder && available.has(softwareEncoder)) {
    return { encoder: softwareEncoder, hwType: 'none' };
  }

  throw new ShoemakerError(
    `No encoder available for codec: ${codec}`,
    ErrorCode.DECODER_NOT_AVAILABLE
  );
}

/**
 * Get oriented dimensions (swap width/height for 90°/270° rotation)
 * Mobile devices record portrait as landscape + rotation metadata
 */
function getOrientedDimensions(
  width: number,
  height: number,
  rotation: number | undefined
): { width: number; height: number } {
  const rot = Math.abs(rotation ?? 0) % 360;
  if (rot === 90 || rot === 270) {
    return { width: height, height: width };
  }
  return { width, height };
}

/**
 * Build FFmpeg filter chain for proxy generation
 */
function buildFilterChain(
  videoInfo: VideoInfo,
  targetHeight: number,
  config: ProxyConfig,
  maxWidth?: number
): string {
  const filters: string[] = [];

  // Deinterlace if needed (must come first)
  if (config.deinterlace && videoInfo.isInterlaced) {
    filters.push('yadif=mode=0');
  }

  // HDR tone mapping (if HDR source, convert to SDR for proxy)
  if (videoInfo.isHdr) {
    filters.push(
      'zscale=t=linear:npl=100',
      'format=gbrpf32le',
      'zscale=p=bt709',
      'tonemap=hable:desat=0',
      'zscale=t=bt709:m=bt709:r=tv',
      'format=yuv420p'
    );
  }

  // Get oriented dimensions for proper scaling calculation
  const oriented = getOrientedDimensions(videoInfo.width, videoInfo.height, videoInfo.rotation);

  // Scale to target height, preserving aspect ratio
  // Use oriented dimensions to calculate correct scale
  // -2 ensures even dimensions (required for h264)
  if (oriented.height <= targetHeight) {
    // Source is smaller than target - don't upscale
    filters.push('scale=trunc(iw/2)*2:trunc(ih/2)*2'); // Just ensure even dimensions
  } else if (maxWidth && oriented.width > maxWidth) {
    // Source is wider than max - constrain by width
    filters.push(`scale=${maxWidth}:-2`);
  } else {
    // Normal case - scale by height
    filters.push(`scale=-2:${targetHeight}`);
  }

  // Apply LUT for color grading (after scaling, before output)
  if (config.lutPath) {
    // Escape path for FFmpeg filter
    const escapedPath = config.lutPath.replace(/'/g, "'\\''");
    filters.push(`lut3d='${escapedPath}'`);
  }

  return filters.length > 0 ? filters.join(',') : '';
}

/**
 * Generate a single proxy file
 */
export async function generateProxy(
  inputPath: string,
  outputPath: string,
  sizeName: string,
  sizeConfig: ProxySizeConfig,
  options: ProxyGeneratorOptions
): Promise<ProxyResult> {
  const startTime = Date.now();
  const { config, videoInfo, onProgress } = options;

  // Select encoder
  const { encoder, hwType } = await selectEncoder(config.codec, config.hwAccel);

  // Build FFmpeg arguments
  const args: string[] = [
    '-y', // Overwrite output
    '-i', inputPath,
    '-map_metadata', '0', // Preserve source metadata (timecode, creation date, etc.)
  ];

  // Hardware acceleration input (if using hw encoder)
  if (hwType === 'videotoolbox') {
    args.unshift('-hwaccel', 'videotoolbox');
  } else if (hwType === 'nvenc') {
    args.unshift('-hwaccel', 'cuda');
  } else if (hwType === 'vaapi') {
    args.unshift('-hwaccel', 'vaapi', '-hwaccel_output_format', 'vaapi');
  } else if (hwType === 'qsv') {
    args.unshift('-hwaccel', 'qsv');
  }

  // Video codec
  args.push('-c:v', encoder);

  // Filter chain
  const filterChain = buildFilterChain(videoInfo, sizeConfig.height, config, sizeConfig.maxWidth);
  if (filterChain) {
    args.push('-vf', filterChain);
  }

  // Quality settings
  if (sizeConfig.bitrate) {
    args.push('-b:v', sizeConfig.bitrate);
  } else {
    // Use CRF for quality-based encoding
    if (encoder.includes('nvenc')) {
      args.push('-cq', String(sizeConfig.crf));
    } else if (encoder.includes('videotoolbox')) {
      // VideoToolbox uses different quality parameter
      args.push('-q:v', String(Math.round(sizeConfig.crf * 1.5)));
    } else {
      args.push('-crf', String(sizeConfig.crf));
    }
  }

  // Encoding preset (not for hardware encoders that don't support it)
  if (!encoder.includes('videotoolbox') && !encoder.includes('prores')) {
    args.push('-preset', config.preset);
  }

  // Pixel format for compatibility
  if (!encoder.includes('prores')) {
    args.push('-pix_fmt', 'yuv420p');
  }

  // Keyframe interval for smooth NLE scrubbing (~1 second)
  const gopSize = Math.round(videoInfo.frameRate);
  args.push('-g', String(gopSize));

  // VFR to CFR conversion (variable frame rate breaks editing software)
  args.push('-vsync', 'cfr');

  // Color space tagging for proper display (bt709 for SDR content)
  if (!videoInfo.isHdr) {
    args.push('-colorspace', 'bt709', '-color_primaries', 'bt709', '-color_trc', 'bt709');
  }

  // Embed source filename for relinking in NLEs
  const sourceFilename = path.basename(inputPath);
  args.push('-metadata', `comment=Source: ${sourceFilename}`);

  // Audio handling
  if (config.audioCodec === 'none' || !videoInfo.audio) {
    args.push('-an'); // No audio
  } else if (config.audioCodec === 'copy') {
    args.push('-c:a', 'copy');
  } else {
    args.push('-c:a', 'aac', '-b:a', config.audioBitrate);
  }

  // Fast start for MP4 (moves moov atom to beginning for streaming)
  if (config.fastStart && config.format === 'mp4') {
    args.push('-movflags', '+faststart');
  }

  // Output file
  args.push(outputPath);

  // Run FFmpeg with progress tracking
  return new Promise((resolve, reject) => {
    const ffmpeg = spawn('ffmpeg', args, {
      stdio: ['pipe', 'pipe', 'pipe'],
    });

    let stderr = '';

    ffmpeg.stderr?.on('data', (data: Buffer) => {
      stderr += data.toString();

      // Parse progress from FFmpeg output
      if (onProgress) {
        const timeMatch = stderr.match(/time=(\d+):(\d+):(\d+\.\d+)/);
        const fpsMatch = stderr.match(/fps=\s*(\d+)/);
        const sizeMatch = stderr.match(/size=\s*(\d+\w+)/);

        if (timeMatch) {
          const hours = parseInt(timeMatch[1] ?? '0', 10);
          const minutes = parseInt(timeMatch[2] ?? '0', 10);
          const seconds = parseFloat(timeMatch[3] ?? '0');
          const currentTime = hours * 3600 + minutes * 60 + seconds;
          const percent = Math.min(100, (currentTime / videoInfo.duration) * 100);
          const fps = parseInt(fpsMatch?.[1] ?? '0', 10);
          const size = sizeMatch?.[1] ?? '0KB';
          onProgress(percent, fps, size);
        }
      }
    });

    ffmpeg.on('close', async (code) => {
      if (code !== 0) {
        // Clean up partial output file on failure
        try {
          await fs.unlink(outputPath);
        } catch {
          // Ignore cleanup errors
        }
        reject(new ShoemakerError(
          `FFmpeg encoding failed: ${stderr.slice(-500)}`,
          ErrorCode.DECODE_FAILED,
          inputPath
        ));
        return;
      }

      try {
        // Get output file stats
        const stats = await fs.stat(outputPath);

        // Probe output to get actual dimensions
        const outputInfo = await probeVideo(outputPath);

        resolve({
          size: sizeName,
          width: outputInfo.width,
          height: outputInfo.height,
          codec: config.codec,
          format: config.format,
          path: outputPath,
          bytes: stats.size,
          duration: Date.now() - startTime,
          bitrate: outputInfo.bitrate,
        });
      } catch (err) {
        reject(new ShoemakerError(
          `Failed to verify proxy output: ${(err as Error).message}`,
          ErrorCode.DECODE_FAILED,
          outputPath
        ));
      }
    });

    ffmpeg.on('error', async (err) => {
      // Clean up partial output file on spawn failure
      try {
        await fs.unlink(outputPath);
      } catch {
        // Ignore cleanup errors
      }
      reject(new ShoemakerError(
        `FFmpeg spawn failed: ${err.message}`,
        ErrorCode.DECODER_NOT_AVAILABLE,
        inputPath
      ));
    });
  });
}

/**
 * Generate all configured proxy sizes for a video
 */
export async function generateProxies(
  inputPath: string,
  options: ProxyGeneratorOptions
): Promise<ProxyResult[]> {
  const { config, outputDir, stem } = options;
  const results: ProxyResult[] = [];

  // Create proxies subdirectory
  const proxyDir = path.join(outputDir, 'proxies');
  await fs.mkdir(proxyDir, { recursive: true });

  // Generate each configured size
  for (const [sizeName, sizeConfig] of Object.entries(config.sizes)) {
    // Skip if target height is larger than source
    if (sizeConfig.height >= options.videoInfo.height) {
      continue;
    }

    const outputPath = path.join(
      proxyDir,
      `${stem}_proxy_${sizeConfig.height}p.${config.format}`
    );

    const result = await generateProxy(
      inputPath,
      outputPath,
      sizeName,
      sizeConfig,
      options
    );

    results.push(result);
  }

  return results;
}

/**
 * Check if FFmpeg supports video encoding
 */
export async function checkFfmpegEncodingSupport(): Promise<boolean> {
  try {
    const available = await detectAvailableEncoders();
    return available.has('libx264') || available.has('h264_videotoolbox');
  } catch {
    return false;
  }
}

/**
 * Get info about available hardware acceleration
 */
export async function getHardwareAccelInfo(): Promise<{
  available: string[];
  recommended: string;
}> {
  const available: string[] = [];
  const encoders = await detectAvailableEncoders();

  if (encoders.has('h264_videotoolbox')) available.push('videotoolbox');
  if (encoders.has('h264_nvenc')) available.push('nvenc');
  if (encoders.has('h264_vaapi')) available.push('vaapi');
  if (encoders.has('h264_qsv')) available.push('qsv');
  available.push('none'); // Software always available

  // Recommend first available hardware, or software
  const recommended = available[0] ?? 'none';

  return { available, recommended };
}
