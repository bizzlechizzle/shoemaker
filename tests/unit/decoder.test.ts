/**
 * Decoder module tests
 */

import { describe, it, expect, beforeEach, afterEach, vi } from 'vitest';
import {
  detectAvailableDecoders,
  selectDecoder,
  isDecoderAvailable,
  getDecoderInfo,
  clearDecoderCache,
  type DecoderType,
} from '../../src/core/decoder.js';

describe('Decoder Module', () => {
  beforeEach(() => {
    clearDecoderCache();
  });

  afterEach(() => {
    clearDecoderCache();
  });

  describe('detectAvailableDecoders', () => {
    it('should always report embedded as available', async () => {
      const decoders = await detectAvailableDecoders();
      const embedded = decoders.get('embedded');

      expect(embedded).toBeDefined();
      expect(embedded?.available).toBe(true);
      expect(embedded?.version).toBe('built-in');
    });

    it('should always report sharp as available', async () => {
      const decoders = await detectAvailableDecoders();
      const sharp = decoders.get('sharp');

      expect(sharp).toBeDefined();
      expect(sharp?.available).toBe(true);
      expect(sharp?.version).toBeDefined();
    });

    it('should cache decoder detection results', async () => {
      const first = await detectAvailableDecoders();
      const second = await detectAvailableDecoders();

      // Should return the same cached instance
      expect(first).toBe(second);
    });
  });

  describe('selectDecoder', () => {
    it('should return embedded when no preference specified', async () => {
      const decoder = await selectDecoder();
      expect(decoder).toBe('embedded');
    });

    it('should return preferred decoder if available', async () => {
      const decoder = await selectDecoder('embedded');
      expect(decoder).toBe('embedded');
    });

    it('should return sharp when preferred', async () => {
      const decoder = await selectDecoder('sharp');
      expect(decoder).toBe('sharp');
    });

    it('should use fallback if preferred not available', async () => {
      // rawtherapee may not be installed
      const decoder = await selectDecoder('rawtherapee', 'embedded');
      // If rawtherapee is installed, it returns rawtherapee; otherwise embedded
      expect(['rawtherapee', 'embedded']).toContain(decoder);
    });
  });

  describe('isDecoderAvailable', () => {
    it('should return true for embedded', async () => {
      expect(await isDecoderAvailable('embedded')).toBe(true);
    });

    it('should return true for sharp', async () => {
      expect(await isDecoderAvailable('sharp')).toBe(true);
    });
  });

  describe('getDecoderInfo', () => {
    it('should return info for all decoder types', async () => {
      const info = await getDecoderInfo();

      expect(info.length).toBeGreaterThanOrEqual(2); // At least embedded and sharp
      expect(info.some(d => d.name === 'embedded')).toBe(true);
      expect(info.some(d => d.name === 'sharp')).toBe(true);
    });

    it('should include version information', async () => {
      const info = await getDecoderInfo();
      const embedded = info.find(d => d.name === 'embedded');
      const sharp = info.find(d => d.name === 'sharp');

      expect(embedded?.version).toBe('built-in');
      expect(sharp?.version).toBeDefined();
    });
  });

  describe('clearDecoderCache', () => {
    it('should force re-detection after clearing', async () => {
      const first = await detectAvailableDecoders();
      clearDecoderCache();
      const second = await detectAvailableDecoders();

      // Should be different instances
      expect(first).not.toBe(second);
      // But same content
      expect(first.size).toBe(second.size);
    });
  });
});
