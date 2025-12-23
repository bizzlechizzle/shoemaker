/**
 * Resizer module tests
 */

import { describe, it, expect } from 'vitest';
import { getSharpCapabilities } from '../../src/core/resizer.js';

describe('Resizer Module', () => {
  describe('getSharpCapabilities', () => {
    it('should return sharp version', () => {
      const caps = getSharpCapabilities();

      expect(caps.version).toBeDefined();
      expect(typeof caps.version).toBe('string');
      // Version should be semver-like
      expect(caps.version).toMatch(/^\d+\.\d+/);
    });

    it('should report supported input formats', () => {
      const caps = getSharpCapabilities();

      expect(caps.formats.input).toBeDefined();
      expect(Array.isArray(caps.formats.input)).toBe(true);
      expect(caps.formats.input).toContain('jpeg');
      expect(caps.formats.input).toContain('png');
    });

    it('should report supported output formats', () => {
      const caps = getSharpCapabilities();

      expect(caps.formats.output).toBeDefined();
      expect(Array.isArray(caps.formats.output)).toBe(true);
      expect(caps.formats.output).toContain('jpeg');
      expect(caps.formats.output).toContain('png');
      expect(caps.formats.output).toContain('webp');
    });

    it('should include webp in output formats', () => {
      const caps = getSharpCapabilities();
      expect(caps.formats.output).toContain('webp');
    });
  });
});

describe('Image Metadata', () => {
  // These tests would require actual image files
  // They serve as documentation of the expected API

  it('should export getImageMetadata function', async () => {
    const { getImageMetadata } = await import('../../src/core/resizer.js');
    expect(typeof getImageMetadata).toBe('function');
  });

  it('should export resizeImage function', async () => {
    const { resizeImage } = await import('../../src/core/resizer.js');
    expect(typeof resizeImage).toBe('function');
  });

  it('should export generateThumbnail function', async () => {
    const { generateThumbnail } = await import('../../src/core/resizer.js');
    expect(typeof generateThumbnail).toBe('function');
  });

  it('should export generateThumbnails function', async () => {
    const { generateThumbnails } = await import('../../src/core/resizer.js');
    expect(typeof generateThumbnails).toBe('function');
  });
});
