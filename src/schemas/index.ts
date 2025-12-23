/**
 * Shoemaker Schemas
 *
 * Zod schemas for configuration, results, and validation.
 */

import { z } from 'zod';

// ============================================================================
// Constants - Centralized magic values
// ============================================================================

/** Default minimum preview size in pixels for thumbnail extraction */
export const DEFAULT_MIN_PREVIEW_SIZE = 2560;

/** Maximum concurrent file processing workers */
export const MAX_CONCURRENCY = 32;

/** Sidecar folder template. Supports: {stem} = filename without extension */
export const DEFAULT_SIDECAR_FOLDER = '{stem}_thumbs';

/** Cache directory path. Supports ~ for home directory expansion */
export const DEFAULT_CACHE_DIR = '~/.cache/shoemaker';

/** Output filename template. Supports: {stem}, {size}, {width}, {format} */
export const DEFAULT_NAMING_PATTERN = '{stem}_{size}_{width}.{format}';

/** Maximum errors to display in CLI output */
export const MAX_ERRORS_TO_DISPLAY = 10;

/** Maximum pending files to display in status command */
export const MAX_PENDING_TO_DISPLAY = 20;

/** RAW file extensions that require preview extraction or decoding */
export const RAW_EXTENSIONS = [
  'arw', 'cr2', 'cr3', 'nef', 'raf', 'rw2', 'orf', 'pef', 'dng',
  'srw', 'x3f', 'erf', 'mrw', 'dcr', 'kdc', 'rwl', 'raw', '3fr',
  'ari', 'srf', 'sr2', 'bay', 'crw', 'iiq',
] as const;

/** Pre-decoded image formats that can be processed directly */
export const DECODED_EXTENSIONS = [
  'jpg', 'jpeg', 'png', 'tif', 'tiff', 'webp', 'avif', 'heic', 'heif',
] as const;

/** Allowed RAW decoder commands (security whitelist) */
export const ALLOWED_DECODER_COMMANDS = [
  'rawtherapee-cli', 'darktable-cli', 'dcraw',
] as const;

/** Allowed metadata tool commands (security whitelist) */
export const ALLOWED_METADATA_COMMANDS = ['exiv2'] as const;

// Size configuration for a single thumbnail tier
export const SizeConfigSchema = z.object({
  width: z.number().int().positive(),
  height: z.number().int().positive().optional(),
  format: z.enum(['webp', 'jpeg', 'png', 'avif']),
  quality: z.number().int().min(1).max(100),
  allowUpscale: z.boolean().default(false),
});

// Output configuration
export const OutputConfigSchema = z.object({
  location: z.enum(['sidecar', 'cache']).default('sidecar'),
  sidecarFolder: z.string().default(DEFAULT_SIDECAR_FOLDER),
  cacheDir: z.string().default(DEFAULT_CACHE_DIR),
  namingPattern: z.string().default(DEFAULT_NAMING_PATTERN),
});

// Processing configuration
export const ProcessingConfigSchema = z.object({
  concurrency: z.number().int().min(1).max(MAX_CONCURRENCY).default(4),
  minPreviewSize: z.number().int().positive().default(DEFAULT_MIN_PREVIEW_SIZE),
  skipExisting: z.boolean().default(true),
  autoRotate: z.boolean().default(true),
  stripExif: z.boolean().default(true),
});

// Behavior configuration (preset-specific)
export const BehaviorConfigSchema = z.object({
  fallbackToRaw: z.boolean().default(false),
  useLargestAvailable: z.boolean().default(true),
  skipIfInsufficient: z.boolean().default(false),
  decoder: z.enum(['embedded', 'libraw', 'rawtherapee', 'darktable', 'dcraw', 'vips']).optional(),
  fallbackDecoder: z.enum(['embedded', 'libraw', 'rawtherapee', 'darktable', 'dcraw', 'vips']).optional(),
  profile: z.string().optional(),
});

// File type configuration
export const FileTypesConfigSchema = z.object({
  include: z.array(z.string()).default([
    'arw', 'cr2', 'cr3', 'nef', 'raf', 'rw2', 'orf', 'pef', 'dng',
    'jpg', 'jpeg', 'png', 'tif', 'tiff', 'heic', 'heif', 'webp',
  ]),
  exclude: z.array(z.string()).default(['xmp', 'json', 'txt', 'md']),
  noPreviewAction: z.enum(['decode', 'skip', 'warn']).default('decode'),
});

// XMP configuration
export const XmpConfigSchema = z.object({
  updateSidecars: z.boolean().default(true),
  createIfMissing: z.boolean().default(false),
  namespace: z.string().default('http://shoemaker.local/1.0/'),
});

// Logging configuration
export const LoggingConfigSchema = z.object({
  level: z.enum(['error', 'warn', 'info', 'debug']).default('info'),
  file: z.string().default(''),
  progress: z.boolean().default(true),
});

// Full configuration schema
export const ConfigSchema = z.object({
  defaultPreset: z.string().default('fast'),
  presetDir: z.string().default('~/.config/shoemaker/presets'),
  output: OutputConfigSchema.default({}),
  processing: ProcessingConfigSchema.default({}),
  sizes: z.record(z.string(), SizeConfigSchema).default({
    thumb: { width: 300, format: 'webp', quality: 80 },
    preview: { width: 1600, format: 'webp', quality: 85 },
    ml: { width: 2560, format: 'jpeg', quality: 90 },
  }),
  filetypes: FileTypesConfigSchema.default({}),
  xmp: XmpConfigSchema.default({}),
  logging: LoggingConfigSchema.default({}),
});

// Preset schema (subset of config with behavior)
export const PresetSchema = z.object({
  sizes: z.record(z.string(), z.object({
    width: z.number().optional(),
    format: z.string().optional(),
    quality: z.number().optional(),
  })).optional(),
  formats: z.record(z.string(), z.string()).optional(),
  quality: z.record(z.string(), z.number()).optional(),
  behavior: BehaviorConfigSchema.default({}),
});

// Preview analysis result
export const PreviewInfoSchema = z.object({
  exists: z.boolean(),
  width: z.number().optional(),
  height: z.number().optional(),
  length: z.number().optional(),
});

export const PreviewAnalysisSchema = z.object({
  filePath: z.string(),
  jpgFromRaw: PreviewInfoSchema,
  previewImage: PreviewInfoSchema,
  otherImage: PreviewInfoSchema,
  thumbnailImage: PreviewInfoSchema,
  bestPreview: z.object({
    type: z.enum(['JpgFromRaw', 'PreviewImage', 'OtherImage', 'ThumbnailImage']),
    width: z.number(),
    height: z.number(),
  }).nullable(),
  needsRawDecode: z.boolean(),
});

// Thumbnail result
export const ThumbnailResultSchema = z.object({
  size: z.string(),
  width: z.number(),
  height: z.number(),
  format: z.string(),
  path: z.string(),
  bytes: z.number(),
});

// Generation result
export const GenerationResultSchema = z.object({
  source: z.string(),
  method: z.enum(['extracted', 'decoded', 'direct']),
  thumbnails: z.array(ThumbnailResultSchema),
  warnings: z.array(z.string()).default([]),
  duration: z.number(),
});

// Batch result
export const BatchResultSchema = z.object({
  total: z.number(),
  succeeded: z.number(),
  failed: z.number(),
  skipped: z.number(),
  warnings: z.number(),
  duration: z.number(),
  errors: z.array(z.object({
    file: z.string(),
    code: z.string(),
    message: z.string(),
  })),
});

// Type exports
export type SizeConfig = z.infer<typeof SizeConfigSchema>;
export type OutputConfig = z.infer<typeof OutputConfigSchema>;
export type ProcessingConfig = z.infer<typeof ProcessingConfigSchema>;
export type BehaviorConfig = z.infer<typeof BehaviorConfigSchema>;
export type FileTypesConfig = z.infer<typeof FileTypesConfigSchema>;
export type XmpConfig = z.infer<typeof XmpConfigSchema>;
export type LoggingConfig = z.infer<typeof LoggingConfigSchema>;
export type Config = z.infer<typeof ConfigSchema>;
export type Preset = z.infer<typeof PresetSchema>;
export type PreviewInfo = z.infer<typeof PreviewInfoSchema>;
export type PreviewAnalysis = z.infer<typeof PreviewAnalysisSchema>;
export type ThumbnailResult = z.infer<typeof ThumbnailResultSchema>;
export type GenerationResult = z.infer<typeof GenerationResultSchema>;
export type BatchResult = z.infer<typeof BatchResultSchema>;
