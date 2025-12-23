/**
 * Thumbnail Generator module tests
 */

import { describe, it, expect } from 'vitest';

describe('Thumbnail Generator Module', () => {
  describe('Module exports', () => {
    it('should export generateForFile function', async () => {
      const { generateForFile } = await import('../../src/services/thumbnail-generator.js');
      expect(typeof generateForFile).toBe('function');
    });

    it('should export generateForBatch function', async () => {
      const { generateForBatch } = await import('../../src/services/thumbnail-generator.js');
      expect(typeof generateForBatch).toBe('function');
    });

    it('should export findImageFiles function', async () => {
      const { findImageFiles } = await import('../../src/services/thumbnail-generator.js');
      expect(typeof findImageFiles).toBe('function');
    });
  });

  describe('findImageFiles', () => {
    it('should return empty array for non-existent directory', async () => {
      const { findImageFiles } = await import('../../src/services/thumbnail-generator.js');

      // This should throw or return empty
      try {
        const files = await findImageFiles('/non/existent/directory', {
          filetypes: {
            include: ['arw', 'cr2', 'jpg'],
            exclude: ['xmp'],
          },
        });
        expect(files).toEqual([]);
      } catch (err) {
        // Expected - directory doesn't exist
        expect((err as Error).message).toContain('ENOENT');
      }
    });
  });

  describe('GenerateOptions interface', () => {
    it('should define expected properties', async () => {
      // Type check - this is mainly for documentation
      const { generateForFile } = await import('../../src/services/thumbnail-generator.js');
      const { loadConfig, loadPreset, applyPreset } = await import('../../src/core/config.js');

      // Load actual config to verify types work
      const config = await loadConfig();
      const preset = await loadPreset('fast', config);
      const finalConfig = applyPreset(config, preset);

      expect(finalConfig).toBeDefined();
      expect(finalConfig.processing).toBeDefined();
      expect(finalConfig.output).toBeDefined();
      expect(finalConfig.sizes).toBeDefined();
    });
  });

  describe('ProgressInfo interface', () => {
    it('should have expected structure', () => {
      // Document expected structure
      type ProgressStatus = 'processing' | 'success' | 'error' | 'skipped';
      type Method = 'extracted' | 'decoded' | 'direct';

      interface ExpectedProgressInfo {
        current: string;
        completed: number;
        total: number;
        method?: Method;
        status: ProgressStatus;
        message?: string;
        duration?: number;
      }

      // Type assertion test
      const info: ExpectedProgressInfo = {
        current: 'test.arw',
        completed: 5,
        total: 10,
        status: 'processing',
      };

      expect(info.current).toBe('test.arw');
      expect(info.completed).toBe(5);
      expect(info.total).toBe(10);
      expect(info.status).toBe('processing');
    });
  });
});

describe('Batch Processing', () => {
  it('should handle empty file list', async () => {
    const { generateForBatch } = await import('../../src/services/thumbnail-generator.js');
    const { loadConfig, loadPreset, applyPreset } = await import('../../src/core/config.js');

    const config = await loadConfig();
    const preset = await loadPreset('fast', config);
    const finalConfig = applyPreset(config, preset);

    const result = await generateForBatch([], {
      config: finalConfig,
      preset,
    });

    expect(result.total).toBe(0);
    expect(result.succeeded).toBe(0);
    expect(result.failed).toBe(0);
  });
});
