/**
 * Schema and constants tests
 */

import { describe, it, expect } from 'vitest';
import {
  DEFAULT_MIN_PREVIEW_SIZE,
  MAX_CONCURRENCY,
  DEFAULT_SIDECAR_FOLDER,
  DEFAULT_CACHE_DIR,
  DEFAULT_NAMING_PATTERN,
  MAX_ERRORS_TO_DISPLAY,
  MAX_PENDING_TO_DISPLAY,
  RAW_EXTENSIONS,
  DECODED_EXTENSIONS,
  ALLOWED_DECODER_COMMANDS,
  ALLOWED_METADATA_COMMANDS,
  ConfigSchema,
  PresetSchema,
  SizeConfigSchema,
} from '../../src/schemas/index.js';

describe('Constants', () => {
  it('should have valid DEFAULT_MIN_PREVIEW_SIZE', () => {
    expect(DEFAULT_MIN_PREVIEW_SIZE).toBe(2560);
    expect(typeof DEFAULT_MIN_PREVIEW_SIZE).toBe('number');
    expect(DEFAULT_MIN_PREVIEW_SIZE).toBeGreaterThan(0);
  });

  it('should have valid MAX_CONCURRENCY', () => {
    expect(MAX_CONCURRENCY).toBe(32);
    expect(typeof MAX_CONCURRENCY).toBe('number');
    expect(MAX_CONCURRENCY).toBeGreaterThan(0);
  });

  it('should have valid display limits', () => {
    expect(MAX_ERRORS_TO_DISPLAY).toBe(10);
    expect(MAX_PENDING_TO_DISPLAY).toBe(20);
    expect(MAX_ERRORS_TO_DISPLAY).toBeGreaterThan(0);
    expect(MAX_PENDING_TO_DISPLAY).toBeGreaterThan(0);
  });

  it('should have valid default paths', () => {
    expect(DEFAULT_SIDECAR_FOLDER).toBe('{stem}_thumbs');
    expect(DEFAULT_CACHE_DIR).toBe('~/.cache/shoemaker');
    expect(DEFAULT_NAMING_PATTERN).toBe('{stem}_{size}_{width}.{format}');
  });
});

describe('RAW_EXTENSIONS', () => {
  it('should include common RAW formats', () => {
    expect(RAW_EXTENSIONS).toContain('arw'); // Sony
    expect(RAW_EXTENSIONS).toContain('cr2'); // Canon
    expect(RAW_EXTENSIONS).toContain('cr3'); // Canon
    expect(RAW_EXTENSIONS).toContain('nef'); // Nikon
    expect(RAW_EXTENSIONS).toContain('raf'); // Fuji
    expect(RAW_EXTENSIONS).toContain('dng'); // Adobe
  });

  it('should not include decoded formats', () => {
    expect(RAW_EXTENSIONS).not.toContain('jpg');
    expect(RAW_EXTENSIONS).not.toContain('jpeg');
    expect(RAW_EXTENSIONS).not.toContain('png');
  });

  it('should be lowercase', () => {
    for (const ext of RAW_EXTENSIONS) {
      expect(ext).toBe(ext.toLowerCase());
    }
  });
});

describe('DECODED_EXTENSIONS', () => {
  it('should include common decoded formats', () => {
    expect(DECODED_EXTENSIONS).toContain('jpg');
    expect(DECODED_EXTENSIONS).toContain('jpeg');
    expect(DECODED_EXTENSIONS).toContain('png');
    expect(DECODED_EXTENSIONS).toContain('webp');
    expect(DECODED_EXTENSIONS).toContain('heic');
  });

  it('should not include RAW formats', () => {
    expect(DECODED_EXTENSIONS).not.toContain('arw');
    expect(DECODED_EXTENSIONS).not.toContain('cr2');
    expect(DECODED_EXTENSIONS).not.toContain('nef');
  });
});

describe('Security whitelists', () => {
  it('should have valid decoder commands', () => {
    expect(ALLOWED_DECODER_COMMANDS).toContain('rawtherapee-cli');
    expect(ALLOWED_DECODER_COMMANDS).toContain('darktable-cli');
    expect(ALLOWED_DECODER_COMMANDS).toContain('dcraw');
    expect(ALLOWED_DECODER_COMMANDS.length).toBe(3);
  });

  it('should have valid metadata commands', () => {
    expect(ALLOWED_METADATA_COMMANDS).toContain('exiv2');
    expect(ALLOWED_METADATA_COMMANDS.length).toBe(1);
  });
});

describe('SizeConfigSchema', () => {
  it('should validate valid size config', () => {
    const result = SizeConfigSchema.safeParse({
      width: 300,
      format: 'webp',
      quality: 80,
    });
    expect(result.success).toBe(true);
  });

  it('should reject invalid width', () => {
    const result = SizeConfigSchema.safeParse({
      width: -100,
      format: 'webp',
      quality: 80,
    });
    expect(result.success).toBe(false);
  });

  it('should reject invalid format', () => {
    const result = SizeConfigSchema.safeParse({
      width: 300,
      format: 'gif',
      quality: 80,
    });
    expect(result.success).toBe(false);
  });

  it('should reject invalid quality', () => {
    const result = SizeConfigSchema.safeParse({
      width: 300,
      format: 'webp',
      quality: 150,
    });
    expect(result.success).toBe(false);
  });

  it('should default allowUpscale to false', () => {
    const result = SizeConfigSchema.parse({
      width: 300,
      format: 'webp',
      quality: 80,
    });
    expect(result.allowUpscale).toBe(false);
  });
});

describe('ConfigSchema', () => {
  it('should provide defaults for empty config', () => {
    const result = ConfigSchema.parse({});
    expect(result.defaultPreset).toBe('fast');
    expect(result.output.location).toBe('sidecar');
    expect(result.output.sidecarFolder).toBe(DEFAULT_SIDECAR_FOLDER);
    expect(result.processing.concurrency).toBe(4);
    expect(result.processing.minPreviewSize).toBe(DEFAULT_MIN_PREVIEW_SIZE);
  });

  it('should validate concurrency bounds', () => {
    // Valid bounds
    const valid1 = ConfigSchema.safeParse({ processing: { concurrency: 1 } });
    expect(valid1.success).toBe(true);

    const valid32 = ConfigSchema.safeParse({ processing: { concurrency: 32 } });
    expect(valid32.success).toBe(true);

    // Invalid bounds
    const invalid0 = ConfigSchema.safeParse({ processing: { concurrency: 0 } });
    expect(invalid0.success).toBe(false);

    const invalid33 = ConfigSchema.safeParse({ processing: { concurrency: 33 } });
    expect(invalid33.success).toBe(false);
  });
});

describe('PresetSchema', () => {
  it('should validate empty preset with defaults', () => {
    const result = PresetSchema.parse({});
    expect(result.behavior.fallbackToRaw).toBe(false);
    expect(result.behavior.useLargestAvailable).toBe(true);
  });

  it('should validate preset with behavior overrides', () => {
    const result = PresetSchema.parse({
      behavior: {
        fallbackToRaw: true,
        decoder: 'rawtherapee',
      },
    });
    expect(result.behavior.fallbackToRaw).toBe(true);
    expect(result.behavior.decoder).toBe('rawtherapee');
  });
});
