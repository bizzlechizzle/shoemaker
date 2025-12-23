/**
 * Integration tests for the full thumbnail generation pipeline
 *
 * These tests require actual image files in tests/fixtures/.
 * Run `npm run test:download-fixtures` first to set up test files.
 *
 * Tests will be skipped if fixtures are not available.
 */

import { describe, it, expect, beforeAll, afterAll } from 'vitest';
import fs from 'fs/promises';
import path from 'path';
import { fileURLToPath } from 'url';
import sharp from 'sharp';

const __dirname = path.dirname(fileURLToPath(import.meta.url));
const FIXTURES_DIR = path.join(__dirname, '..', 'fixtures');
const IMAGES_DIR = path.join(FIXTURES_DIR, 'images');
const RAW_DIR = path.join(FIXTURES_DIR, 'raw');
const OUTPUT_DIR = path.join(__dirname, '..', 'output');

async function fileExists(filePath: string): Promise<boolean> {
  try {
    await fs.access(filePath);
    return true;
  } catch {
    return false;
  }
}

async function hasTestFixtures(): Promise<boolean> {
  const sampleJpg = path.join(IMAGES_DIR, 'sample.jpg');
  return fileExists(sampleJpg);
}

describe('Integration: Full Pipeline', () => {
  let fixturesAvailable = false;

  beforeAll(async () => {
    fixturesAvailable = await hasTestFixtures();

    if (fixturesAvailable) {
      // Clean output directory
      try {
        await fs.rm(OUTPUT_DIR, { recursive: true });
      } catch {
        // Ignore if doesn't exist
      }
      await fs.mkdir(OUTPUT_DIR, { recursive: true });
    }
  });

  afterAll(async () => {
    // Cleanup output directory after tests
    try {
      await fs.rm(OUTPUT_DIR, { recursive: true });
    } catch {
      // Ignore errors
    }
  });

  describe('JPEG Processing', () => {
    it.skipIf(!fixturesAvailable)('should process JPEG file end-to-end', async () => {
      const { generateForFile } = await import('../../src/services/thumbnail-generator.js');
      const { loadConfig, loadPreset, applyPreset } = await import('../../src/core/config.js');

      const config = await loadConfig();
      const preset = await loadPreset('fast', config);
      let finalConfig = applyPreset(config, preset);

      // Override output to test directory
      finalConfig = {
        ...finalConfig,
        output: {
          ...finalConfig.output,
          location: 'cache' as const,
          cacheDir: OUTPUT_DIR,
        },
        xmp: {
          ...finalConfig.xmp,
          updateSidecars: false, // Don't write XMP in tests
        },
      };

      const inputPath = path.join(IMAGES_DIR, 'sample.jpg');
      const result = await generateForFile(inputPath, {
        config: finalConfig,
        preset,
        force: true,
      });

      expect(result.method).toBe('direct');
      expect(result.thumbnails.length).toBeGreaterThan(0);
      expect(result.duration).toBeDefined();

      // Verify thumbnails were created
      for (const thumb of result.thumbnails) {
        const exists = await fileExists(thumb.path);
        expect(exists).toBe(true);

        // Verify thumbnail dimensions
        const meta = await sharp(thumb.path).metadata();
        expect(meta.width).toBeLessThanOrEqual(thumb.width + 1); // Allow 1px tolerance
      }
    });

    it.skipIf(!fixturesAvailable)('should handle dry-run mode', async () => {
      const { generateForFile } = await import('../../src/services/thumbnail-generator.js');
      const { loadConfig, loadPreset, applyPreset } = await import('../../src/core/config.js');

      const config = await loadConfig();
      const preset = await loadPreset('fast', config);
      const finalConfig = applyPreset(config, preset);

      const inputPath = path.join(IMAGES_DIR, 'sample.jpg');
      const result = await generateForFile(inputPath, {
        config: finalConfig,
        preset,
        dryRun: true,
      });

      expect(result.thumbnails).toEqual([]);
      expect(result.warnings).toContain('Dry run: no files written');
    });
  });

  describe('Batch Processing', () => {
    it.skipIf(!fixturesAvailable)('should process batch of files', async () => {
      const { generateForBatch, findImageFiles } = await import('../../src/services/thumbnail-generator.js');
      const { loadConfig, loadPreset, applyPreset } = await import('../../src/core/config.js');

      const config = await loadConfig();
      const preset = await loadPreset('fast', config);
      let finalConfig = applyPreset(config, preset);

      finalConfig = {
        ...finalConfig,
        output: {
          ...finalConfig.output,
          location: 'cache' as const,
          cacheDir: OUTPUT_DIR,
        },
        xmp: {
          ...finalConfig.xmp,
          updateSidecars: false,
        },
      };

      const files = await findImageFiles(IMAGES_DIR, finalConfig, false);

      const progressUpdates: Array<{ current: string; status: string }> = [];

      const result = await generateForBatch(files, {
        config: finalConfig,
        preset,
        force: true,
        onProgress: (info) => {
          progressUpdates.push({ current: info.current, status: info.status });
        },
      });

      expect(result.total).toBe(files.length);
      expect(result.succeeded).toBe(files.length);
      expect(result.failed).toBe(0);
      expect(progressUpdates.length).toBeGreaterThan(0);
    });
  });

  describe('Error Handling', () => {
    it('should handle non-existent file gracefully', async () => {
      const { generateForFile } = await import('../../src/services/thumbnail-generator.js');
      const { loadConfig, loadPreset, applyPreset } = await import('../../src/core/config.js');
      const { ShoemakerError, ErrorCode } = await import('../../src/core/errors.js');

      const config = await loadConfig();
      const preset = await loadPreset('fast', config);
      const finalConfig = applyPreset(config, preset);

      const inputPath = '/non/existent/file.jpg';

      try {
        await generateForFile(inputPath, {
          config: finalConfig,
          preset,
        });
        expect.fail('Should have thrown');
      } catch (err) {
        // Error may be raw ENOENT or wrapped ShoemakerError depending on path
        if (err instanceof ShoemakerError) {
          expect(err.code).toBe(ErrorCode.FILE_NOT_FOUND);
        } else {
          // Raw error should be ENOENT
          expect((err as NodeJS.ErrnoException).code).toBe('ENOENT');
        }
      }
    });
  });
});

describe('Integration: Decoder Module', () => {
  it('should detect available decoders', async () => {
    const { detectAvailableDecoders } = await import('../../src/core/decoder.js');

    const decoders = await detectAvailableDecoders();

    // These should always be available
    expect(decoders.get('embedded')?.available).toBe(true);
    expect(decoders.get('sharp')?.available).toBe(true);
  });

  it('should select best decoder based on availability', async () => {
    const { selectDecoder } = await import('../../src/core/decoder.js');

    const decoder = await selectDecoder('embedded');
    expect(decoder).toBe('embedded');

    const sharpDecoder = await selectDecoder('sharp');
    expect(sharpDecoder).toBe('sharp');
  });
});

describe('Integration: Sharp Capabilities', () => {
  it('should report correct format support', async () => {
    const { getSharpCapabilities } = await import('../../src/core/resizer.js');

    const caps = getSharpCapabilities();

    // WebP should be supported for thumbnails
    expect(caps.formats.output).toContain('webp');
    expect(caps.formats.output).toContain('jpeg');

    // JPEG input should be supported
    expect(caps.formats.input).toContain('jpeg');
    expect(caps.formats.input).toContain('png');
  });
});
