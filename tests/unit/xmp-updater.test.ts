/**
 * XMP Updater module tests
 */

import { describe, it, expect } from 'vitest';
import { getXmpPath } from '../../src/services/xmp-updater.js';

describe('XMP Updater Module', () => {
  describe('getXmpPath', () => {
    it('should append .xmp to image path', () => {
      expect(getXmpPath('/path/to/image.arw')).toBe('/path/to/image.arw.xmp');
    });

    it('should work with various extensions', () => {
      expect(getXmpPath('/path/to/image.ARW')).toBe('/path/to/image.ARW.xmp');
      expect(getXmpPath('/path/to/image.cr2')).toBe('/path/to/image.cr2.xmp');
      expect(getXmpPath('/path/to/image.jpg')).toBe('/path/to/image.jpg.xmp');
    });

    it('should work with paths containing spaces', () => {
      expect(getXmpPath('/path/with spaces/image.arw')).toBe('/path/with spaces/image.arw.xmp');
    });

    it('should work with relative paths', () => {
      expect(getXmpPath('image.arw')).toBe('image.arw.xmp');
      expect(getXmpPath('./image.arw')).toBe('./image.arw.xmp');
    });
  });

  describe('Module exports', () => {
    it('should export updateXmpSidecar function', async () => {
      const { updateXmpSidecar } = await import('../../src/services/xmp-updater.js');
      expect(typeof updateXmpSidecar).toBe('function');
    });

    it('should export hasExistingThumbnails function', async () => {
      const { hasExistingThumbnails } = await import('../../src/services/xmp-updater.js');
      expect(typeof hasExistingThumbnails).toBe('function');
    });

    it('should export readThumbnailInfo function', async () => {
      const { readThumbnailInfo } = await import('../../src/services/xmp-updater.js');
      expect(typeof readThumbnailInfo).toBe('function');
    });

    it('should export clearThumbnailInfo function', async () => {
      const { clearThumbnailInfo } = await import('../../src/services/xmp-updater.js');
      expect(typeof clearThumbnailInfo).toBe('function');
    });

    it('should export xmpExists function', async () => {
      const { xmpExists } = await import('../../src/services/xmp-updater.js');
      expect(typeof xmpExists).toBe('function');
    });
  });

  describe('hasExistingThumbnails', () => {
    it('should return false for non-existent file', async () => {
      const { hasExistingThumbnails } = await import('../../src/services/xmp-updater.js');
      const result = await hasExistingThumbnails('/non/existent/path.arw');
      expect(result).toBe(false);
    });
  });

  describe('readThumbnailInfo', () => {
    it('should return exists: false for non-existent file', async () => {
      const { readThumbnailInfo } = await import('../../src/services/xmp-updater.js');
      const result = await readThumbnailInfo('/non/existent/path.arw');
      expect(result.exists).toBe(false);
    });
  });

  describe('xmpExists', () => {
    it('should return false for non-existent xmp', async () => {
      const { xmpExists } = await import('../../src/services/xmp-updater.js');
      const result = await xmpExists('/non/existent/path.arw');
      expect(result).toBe(false);
    });
  });
});
