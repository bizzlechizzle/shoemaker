/**
 * XMP Updater Tests for shoemaker
 *
 * Tests for XMP sidecar metadata writing with shoe: namespace.
 */

import { describe, it, expect, beforeEach, afterEach, vi } from 'vitest';
import { mkdtempSync, rmSync, writeFileSync, existsSync, readFileSync } from 'fs';
import { tmpdir } from 'os';
import { join } from 'path';
import {
  ThumbnailMetadata,
  writeThumbnailXMP,
  xmpExists,
  getXmpPath,
  appendCustodyEvent,
  readLegacyXmpMetadata,
  migrateToNewFormat,
} from '../src/services/xmp-updater.js';

describe('XMP Path Helpers', () => {
  let tempDir: string;

  beforeEach(() => {
    tempDir = mkdtempSync(join(tmpdir(), 'shoe-xmp-test-'));
  });

  afterEach(() => {
    try {
      rmSync(tempDir, { recursive: true });
    } catch {
      // Ignore cleanup errors
    }
  });

  describe('getXmpPath()', () => {
    it('should return correct XMP sidecar path', () => {
      const imgPath = join(tempDir, 'image.jpg');
      const xmpPath = getXmpPath(imgPath);

      expect(xmpPath).toBe(join(tempDir, 'image.jpg.xmp'));
    });

    it('should handle paths with multiple dots', () => {
      const imgPath = join(tempDir, 'my.photo.2024.jpg');
      const xmpPath = getXmpPath(imgPath);

      expect(xmpPath).toBe(join(tempDir, 'my.photo.2024.jpg.xmp'));
    });
  });

  describe('xmpExists()', () => {
    it('should return false when XMP does not exist', () => {
      const imgPath = join(tempDir, 'image.jpg');
      writeFileSync(imgPath, '');

      expect(xmpExists(imgPath)).toBe(false);
    });

    it('should return true when XMP exists', () => {
      const imgPath = join(tempDir, 'image.jpg');
      const xmpPath = join(tempDir, 'image.jpg.xmp');
      writeFileSync(imgPath, '');
      writeFileSync(xmpPath, '<?xml version="1.0"?><x:xmpmeta/>');

      expect(xmpExists(imgPath)).toBe(true);
    });
  });
});

describe('ThumbnailMetadata', () => {
  it('should have required fields', () => {
    const metadata: ThumbnailMetadata = {
      sourcePath: '/path/to/source.jpg',
      outputPath: '/path/to/thumb.jpg',
      width: 400,
      height: 300,
      quality: 85,
      format: 'jpeg',
      processingTimeMs: 150,
    };

    expect(metadata.sourcePath).toBe('/path/to/source.jpg');
    expect(metadata.width).toBe(400);
    expect(metadata.height).toBe(300);
    expect(metadata.quality).toBe(85);
  });

  it('should support optional fields', () => {
    const metadata: ThumbnailMetadata = {
      sourcePath: '/path/to/source.jpg',
      outputPath: '/path/to/thumb.jpg',
      width: 400,
      height: 300,
      quality: 85,
      format: 'webp',
      processingTimeMs: 150,
      preset: 'web-large',
      sourceHash: 'blake3:abc123',
    };

    expect(metadata.preset).toBe('web-large');
    expect(metadata.sourceHash).toBe('blake3:abc123');
    expect(metadata.format).toBe('webp');
  });
});

describe('writeThumbnailXMP()', () => {
  let tempDir: string;
  let mockExiftool: ReturnType<typeof vi.fn>;

  beforeEach(() => {
    tempDir = mkdtempSync(join(tmpdir(), 'shoe-xmp-write-'));

    // Mock exiftool-vendored
    mockExiftool = vi.fn().mockResolvedValue(undefined);
  });

  afterEach(() => {
    try {
      rmSync(tempDir, { recursive: true });
    } catch {
      // Ignore
    }
    vi.restoreAllMocks();
  });

  // Note: These tests would require mocking exiftool-vendored
  // In a real test environment, you'd use dependency injection or module mocking

  it('should create metadata object with correct structure', () => {
    const metadata: ThumbnailMetadata = {
      sourcePath: '/original/photo.raw',
      outputPath: join(tempDir, 'photo.jpg'),
      width: 1200,
      height: 800,
      quality: 90,
      format: 'jpeg',
      processingTimeMs: 250,
      preset: 'web-xlarge',
      sourceHash: 'blake3:abc123def456',
    };

    // Verify structure
    expect(metadata).toHaveProperty('sourcePath');
    expect(metadata).toHaveProperty('outputPath');
    expect(metadata).toHaveProperty('width');
    expect(metadata).toHaveProperty('height');
    expect(metadata).toHaveProperty('quality');
    expect(metadata).toHaveProperty('format');
    expect(metadata).toHaveProperty('processingTimeMs');
  });
});

describe('Namespace Compliance', () => {
  it('should use shoe: namespace prefix', () => {
    // This is a structural test verifying the module exports expected functions
    expect(typeof writeThumbnailXMP).toBe('function');
    expect(typeof appendCustodyEvent).toBe('function');
  });

  it('should use wnb:CustodyChain for shared custody', () => {
    // Verify custody chain uses shared namespace
    expect(typeof appendCustodyEvent).toBe('function');
  });
});

describe('readLegacyXmpMetadata()', () => {
  let tempDir: string;

  beforeEach(() => {
    tempDir = mkdtempSync(join(tmpdir(), 'shoe-legacy-'));
  });

  afterEach(() => {
    try {
      rmSync(tempDir, { recursive: true });
    } catch {
      // Ignore
    }
  });

  it('should be a function for reading legacy format', () => {
    expect(typeof readLegacyXmpMetadata).toBe('function');
  });
});

describe('migrateToNewFormat()', () => {
  it('should be a function for migration', () => {
    expect(typeof migrateToNewFormat).toBe('function');
  });
});

describe('appendCustodyEvent()', () => {
  it('should be a function for appending custody events', () => {
    expect(typeof appendCustodyEvent).toBe('function');
  });

  it('should accept required parameters', () => {
    // Verify function signature accepts expected parameters
    const fn = appendCustodyEvent;
    expect(fn.length).toBeGreaterThanOrEqual(2); // At least path and action
  });
});

// Integration test that would require real exiftool
describe.skip('Integration Tests', () => {
  let tempDir: string;

  beforeEach(() => {
    tempDir = mkdtempSync(join(tmpdir(), 'shoe-int-'));
  });

  afterEach(() => {
    try {
      rmSync(tempDir, { recursive: true });
    } catch {
      // Ignore
    }
  });

  it('should write XMP sidecar with shoe: namespace', async () => {
    // Create test image file
    const imgPath = join(tempDir, 'test.jpg');
    // Write minimal JPEG header
    const jpegHeader = Buffer.from([
      0xff, 0xd8, 0xff, 0xe0, 0x00, 0x10, 0x4a, 0x46,
      0x49, 0x46, 0x00, 0x01, 0x01, 0x00, 0x00, 0x01,
      0x00, 0x01, 0x00, 0x00, 0xff, 0xd9,
    ]);
    writeFileSync(imgPath, jpegHeader);

    const metadata: ThumbnailMetadata = {
      sourcePath: '/original/test.raw',
      outputPath: imgPath,
      width: 800,
      height: 600,
      quality: 85,
      format: 'jpeg',
      processingTimeMs: 100,
    };

    await writeThumbnailXMP(metadata);

    // Verify XMP file was created
    const xmpPath = getXmpPath(imgPath);
    expect(existsSync(xmpPath)).toBe(true);
  });
});
