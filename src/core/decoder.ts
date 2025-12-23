/**
 * RAW Decoder Module
 *
 * Provides RAW file decoding using multiple backends:
 * - embedded: Extract embedded preview (fastest, used by default)
 * - sharp: Use sharp's libvips for basic RAW support
 * - rawtherapee: Use RawTherapee CLI for high-quality decoding
 * - darktable: Use darktable CLI as alternative
 * - dcraw: Use dcraw as legacy fallback
 *
 * Note: libraw WASM integration is deferred - it requires significant
 * additional dependencies and is better suited as an optional peer dependency.
 */

import { execFile } from 'child_process';
import { promisify } from 'util';
import fs from 'fs/promises';
import path from 'path';
import os from 'os';
import sharp from 'sharp';
import { ShoemakerError, ErrorCode, wrapError } from './errors.js';
import { extractBestPreview } from './extractor.js';

const execFileAsync = promisify(execFile);

export type DecoderType = 'embedded' | 'sharp' | 'rawtherapee' | 'darktable' | 'dcraw';

export interface DecodeOptions {
  /** Decoder to use */
  decoder?: DecoderType;
  /** Fallback decoder if primary fails */
  fallbackDecoder?: DecoderType;
  /** Target width for output (optional, for efficiency) */
  targetWidth?: number;
  /** RawTherapee processing profile path */
  profile?: string;
  /** JPEG quality for CLI output (1-100) */
  quality?: number;
}

interface DecoderInfo {
  name: DecoderType;
  available: boolean;
  path?: string;
  version?: string;
}

// Security: whitelist of allowed decoder commands
const ALLOWED_COMMANDS = new Set(['rawtherapee-cli', 'darktable-cli', 'dcraw', 'convert']);

// Cache decoder availability
let decoderCache: Map<DecoderType, DecoderInfo> | null = null;

/**
 * Detect which decoders are available on the system
 */
export async function detectAvailableDecoders(): Promise<Map<DecoderType, DecoderInfo>> {
  if (decoderCache) {
    return decoderCache;
  }

  const decoders = new Map<DecoderType, DecoderInfo>();

  // Embedded and sharp are always available
  decoders.set('embedded', { name: 'embedded', available: true, version: 'built-in' });
  decoders.set('sharp', { name: 'sharp', available: true, version: sharp.versions.sharp });

  // Check CLI decoders
  const cliDecoders: Array<{ name: DecoderType; command: string; versionArg: string }> = [
    { name: 'rawtherapee', command: 'rawtherapee-cli', versionArg: '--version' },
    { name: 'darktable', command: 'darktable-cli', versionArg: '--version' },
    { name: 'dcraw', command: 'dcraw', versionArg: '' },
  ];

  for (const decoder of cliDecoders) {
    if (!ALLOWED_COMMANDS.has(decoder.command)) {
      continue;
    }

    try {
      const { stdout } = await execFileAsync('which', [decoder.command]);
      const toolPath = stdout.trim();

      let version = 'unknown';
      if (decoder.versionArg) {
        try {
          const { stdout: vOut, stderr: vErr } = await execFileAsync(decoder.command, [decoder.versionArg]);
          const versionMatch = (vOut || vErr).match(/(\d+\.\d+(?:\.\d+)?)/);
          if (versionMatch?.[1]) {
            version = versionMatch[1];
          }
        } catch {
          // Version check failed but tool exists
        }
      }

      decoders.set(decoder.name, {
        name: decoder.name,
        available: true,
        path: toolPath,
        version,
      });
    } catch {
      decoders.set(decoder.name, { name: decoder.name, available: false });
    }
  }

  decoderCache = decoders;
  return decoders;
}

/**
 * Clear the decoder cache (useful for testing)
 */
export function clearDecoderCache(): void {
  decoderCache = null;
}

/**
 * Get the best available decoder based on preferences
 */
export async function selectDecoder(
  preferred?: DecoderType,
  fallback?: DecoderType
): Promise<DecoderType> {
  const available = await detectAvailableDecoders();

  // Try preferred decoder first
  if (preferred && available.get(preferred)?.available) {
    return preferred;
  }

  // Try fallback decoder
  if (fallback && available.get(fallback)?.available) {
    return fallback;
  }

  // Default priority order
  const priority: DecoderType[] = ['embedded', 'sharp', 'rawtherapee', 'darktable', 'dcraw'];
  for (const decoder of priority) {
    if (available.get(decoder)?.available) {
      return decoder;
    }
  }

  // Should never happen since embedded is always available
  return 'embedded';
}

/**
 * Decode a RAW file to a buffer using the specified decoder
 */
export async function decodeRawFile(
  filePath: string,
  options: DecodeOptions = {}
): Promise<Buffer> {
  const decoder = await selectDecoder(options.decoder, options.fallbackDecoder);

  try {
    switch (decoder) {
      case 'embedded':
        return await decodeWithEmbedded(filePath);

      case 'sharp':
        return await decodeWithSharp(filePath, options);

      case 'rawtherapee':
        return await decodeWithRawTherapee(filePath, options);

      case 'darktable':
        return await decodeWithDarktable(filePath, options);

      case 'dcraw':
        return await decodeWithDcraw(filePath, options);

      default:
        throw new ShoemakerError(
          `Unknown decoder: ${decoder}`,
          ErrorCode.DECODER_NOT_AVAILABLE,
          filePath
        );
    }
  } catch (err) {
    // If primary decoder fails and we have a fallback, try it
    if (options.fallbackDecoder && options.decoder !== options.fallbackDecoder) {
      const fallbackDecoder = await selectDecoder(options.fallbackDecoder);
      if (fallbackDecoder !== options.decoder) {
        return decodeRawFile(filePath, {
          ...options,
          decoder: fallbackDecoder,
          fallbackDecoder: undefined, // Don't recurse infinitely
        });
      }
    }
    throw err;
  }
}

/**
 * Decode using embedded preview extraction
 */
async function decodeWithEmbedded(filePath: string): Promise<Buffer> {
  try {
    const result = await extractBestPreview(filePath);
    return result.buffer;
  } catch (err) {
    throw wrapError(err, filePath);
  }
}

/**
 * Decode using sharp's built-in RAW support (via libvips)
 * Note: Sharp's RAW support is limited but works for basic cases
 */
async function decodeWithSharp(filePath: string, options: DecodeOptions): Promise<Buffer> {
  try {
    const image = sharp(filePath, {
      // Enable raw format processing
      raw: undefined, // Let sharp auto-detect
      failOn: 'error',
    });

    // Apply target width if specified for efficiency
    if (options.targetWidth) {
      image.resize(options.targetWidth, options.targetWidth, {
        fit: 'inside',
        withoutEnlargement: true,
      });
    }

    // Convert to sRGB and output as JPEG buffer
    return await image
      .toColorspace('srgb')
      .jpeg({ quality: options.quality ?? 95 })
      .toBuffer();
  } catch (err) {
    throw wrapError(err, filePath);
  }
}

/**
 * Decode using RawTherapee CLI (highest quality)
 */
async function decodeWithRawTherapee(filePath: string, options: DecodeOptions): Promise<Buffer> {
  const tmpDir = os.tmpdir();
  const tmpFile = path.join(tmpDir, `shoemaker-${Date.now()}-${Math.random().toString(36).slice(2)}.jpg`);

  try {
    const args = [
      '-o', tmpFile,
      '-j', String(options.quality ?? 95),
      '-Y', // Overwrite existing
      '-c', // Use camera profiles
    ];

    // Add processing profile if specified
    if (options.profile) {
      args.push('-p', options.profile);
    }

    args.push(filePath);

    await execFileAsync('rawtherapee-cli', args, {
      timeout: 120000, // 2 minute timeout for large files
    });

    // Read the output file
    const buffer = await fs.readFile(tmpFile);
    return buffer;
  } catch (err) {
    throw new ShoemakerError(
      `RawTherapee decode failed: ${(err as Error).message}`,
      ErrorCode.DECODE_FAILED,
      filePath
    );
  } finally {
    // Clean up temp file
    try {
      await fs.unlink(tmpFile);
    } catch {
      // Ignore cleanup errors
    }
  }
}

/**
 * Decode using darktable CLI
 */
async function decodeWithDarktable(filePath: string, options: DecodeOptions): Promise<Buffer> {
  const tmpDir = os.tmpdir();
  const tmpFile = path.join(tmpDir, `shoemaker-${Date.now()}-${Math.random().toString(36).slice(2)}.jpg`);

  try {
    const args = [filePath, tmpFile];

    // Add target width if specified
    if (options.targetWidth) {
      args.push('--width', String(options.targetWidth));
      args.push('--height', String(options.targetWidth)); // Square bounding box
    }

    // Enable high quality mode
    args.push('--hq', 'true');

    await execFileAsync('darktable-cli', args, {
      timeout: 180000, // 3 minute timeout
    });

    // Read the output file
    const buffer = await fs.readFile(tmpFile);
    return buffer;
  } catch (err) {
    throw new ShoemakerError(
      `darktable decode failed: ${(err as Error).message}`,
      ErrorCode.DECODE_FAILED,
      filePath
    );
  } finally {
    // Clean up temp file
    try {
      await fs.unlink(tmpFile);
    } catch {
      // Ignore cleanup errors
    }
  }
}

/**
 * Decode using dcraw (basic, fast)
 * dcraw outputs PPM, which we convert to JPEG via sharp
 */
async function decodeWithDcraw(filePath: string, options: DecodeOptions): Promise<Buffer> {
  try {
    // dcraw -c outputs to stdout as PPM
    // -w: use camera white balance
    // -W: don't automatically brighten
    // -q 3: high-quality interpolation
    const { stdout } = await execFileAsync('dcraw', ['-c', '-w', '-W', '-q', '3', filePath], {
      maxBuffer: 100 * 1024 * 1024, // 100MB buffer for large images
      encoding: 'buffer',
      timeout: 60000, // 1 minute timeout
    });

    // Convert PPM to JPEG using sharp
    let image = sharp(stdout);

    // Apply target width if specified
    if (options.targetWidth) {
      image = image.resize(options.targetWidth, options.targetWidth, {
        fit: 'inside',
        withoutEnlargement: true,
      });
    }

    return await image
      .toColorspace('srgb')
      .jpeg({ quality: options.quality ?? 95 })
      .toBuffer();
  } catch (err) {
    throw new ShoemakerError(
      `dcraw decode failed: ${(err as Error).message}`,
      ErrorCode.DECODE_FAILED,
      filePath
    );
  }
}

/**
 * Check if a specific decoder is available
 */
export async function isDecoderAvailable(decoder: DecoderType): Promise<boolean> {
  const available = await detectAvailableDecoders();
  return available.get(decoder)?.available ?? false;
}

/**
 * Get information about all available decoders
 */
export async function getDecoderInfo(): Promise<DecoderInfo[]> {
  const available = await detectAvailableDecoders();
  return Array.from(available.values());
}
