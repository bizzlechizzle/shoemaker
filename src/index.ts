/**
 * Shoemaker - A Library That Makes Thumbnails
 *
 * This is the library entry point for programmatic use.
 * Import this module to use Shoemaker in your own applications.
 *
 * @example
 * ```typescript
 * import { generateThumbnails, loadConfig, loadPreset } from 'shoemaker';
 *
 * const config = await loadConfig();
 * const preset = await loadPreset('fast');
 *
 * const result = await generateThumbnails('/path/to/image.arw', {
 *   config,
 *   preset,
 * });
 * ```
 */

// Core modules
export {
  analyzeEmbeddedPreviews,
  extractBestPreview,
  extractPreviewBuffer,
  isRawFormat,
  isDecodedFormat,
  isVideoFormat,
  shutdownExiftool,
} from './core/extractor.js';

// Video modules
export {
  probeVideo,
  getVideoDuration,
  hasAudio,
  checkFfprobeAvailable,
  checkFfmpegAvailable,
  getFfprobeVersion,
  getFfmpegVersion,
  clearFfprobeCache,
} from './core/ffprobe.js';

export {
  extractFrame,
  extractFrameAtPercent,
  extractMultipleFrames,
  generateTimelineStrip,
  extractPosterFrame,
  extractPreviewFrame,
  type ExtractedFrame,
  type FrameExtractionOptions,
} from './core/frame-extractor.js';

export {
  resizeImage,
  generateThumbnail,
  generateThumbnails as resizeThumbnails,
  getImageMetadata,
  getSharpCapabilities,
  type ResizeOptions,
  type ResizeResult,
} from './core/resizer.js';

export {
  loadConfig,
  loadPreset,
  applyPreset,
  getBehavior,
  expandPath,
  getConfigPaths,
} from './core/config.js';

export {
  ShoemakerError,
  ErrorCode,
  wrapError,
  isRecoverable,
  shouldStopBatch,
  shouldReduceConcurrency,
} from './core/errors.js';

export {
  decodeRawFile,
  detectAvailableDecoders,
  selectDecoder,
  isDecoderAvailable,
  getDecoderInfo,
  clearDecoderCache,
  type DecoderType,
  type DecodeOptions,
} from './core/decoder.js';

// Services
export {
  generateForFile,
  generateForBatch,
  findImageFiles,
  findVideoFiles,
  type GenerateOptions,
  type ProgressInfo,
} from './services/thumbnail-generator.js';

export {
  updateXmpSidecar,
  hasExistingThumbnails,
  readThumbnailInfo,
  clearThumbnailInfo,
  getXmpPath,
  xmpExists,
  type XmpUpdateData,
} from './services/xmp-updater.js';

// Schemas and types
export {
  ConfigSchema,
  PresetSchema,
  PreviewAnalysisSchema,
  GenerationResultSchema,
  BatchResultSchema,
  VideoInfoSchema,
  VideoConfigSchema,
  VIDEO_EXTENSIONS,
  type Config,
  type Preset,
  type SizeConfig,
  type BehaviorConfig,
  type VideoConfig,
  type VideoInfo,
  type PreviewAnalysis,
  type PreviewInfo,
  type ThumbnailResult,
  type GenerationResult,
  type BatchResult,
} from './schemas/index.js';

// Re-export types for convenience function
import type { ProgressInfo } from './services/thumbnail-generator.js';
import type { GenerationResult } from './schemas/index.js';

// Convenience function for simple use cases
export async function generateThumbnails(
  inputPath: string,
  options?: {
    preset?: string;
    force?: boolean;
    onProgress?: (info: ProgressInfo) => void;
  }
): Promise<GenerationResult> {
  const { loadConfig, loadPreset, applyPreset } = await import('./core/config.js');
  const { generateForFile } = await import('./services/thumbnail-generator.js');

  const config = await loadConfig();
  const preset = await loadPreset(options?.preset ?? 'fast', config);
  const finalConfig = applyPreset(config, preset);

  return generateForFile(inputPath, {
    config: finalConfig,
    preset,
    force: options?.force,
    onProgress: options?.onProgress,
  });
}
