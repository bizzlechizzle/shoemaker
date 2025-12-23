/**
 * Proxy Generator Tests
 *
 * Tests for video proxy generation functionality.
 */

import { describe, it, expect, beforeEach, vi } from 'vitest';
import {
  detectAvailableEncoders,
  selectEncoder,
  getHardwareAccelInfo,
  checkFfmpegEncodingSupport,
} from '../../src/core/proxy-generator.js';
import { ProxyConfigSchema, ProxySizeConfigSchema, VideoInfoSchema } from '../../src/schemas/index.js';

describe('Proxy Generator', () => {
  describe('Schema Validation', () => {
    describe('ProxySizeConfigSchema', () => {
      it('should validate valid proxy size config', () => {
        const config = { height: 720, crf: 23 };
        const result = ProxySizeConfigSchema.safeParse(config);
        expect(result.success).toBe(true);
      });

      it('should reject height below minimum', () => {
        const config = { height: 100, crf: 23 };
        const result = ProxySizeConfigSchema.safeParse(config);
        expect(result.success).toBe(false);
      });

      it('should reject height above maximum', () => {
        const config = { height: 4320, crf: 23 };
        const result = ProxySizeConfigSchema.safeParse(config);
        expect(result.success).toBe(false);
      });

      it('should reject invalid CRF values', () => {
        expect(ProxySizeConfigSchema.safeParse({ height: 720, crf: -1 }).success).toBe(false);
        expect(ProxySizeConfigSchema.safeParse({ height: 720, crf: 52 }).success).toBe(false);
      });

      it('should accept optional bitrate', () => {
        const config = { height: 720, crf: 23, bitrate: '5M' };
        const result = ProxySizeConfigSchema.safeParse(config);
        expect(result.success).toBe(true);
        if (result.success) {
          expect(result.data.bitrate).toBe('5M');
        }
      });

      it('should accept optional maxWidth', () => {
        const config = { height: 720, crf: 23, maxWidth: 1920 };
        const result = ProxySizeConfigSchema.safeParse(config);
        expect(result.success).toBe(true);
        if (result.success) {
          expect(result.data.maxWidth).toBe(1920);
        }
      });
    });

    describe('ProxyConfigSchema', () => {
      it('should provide sensible defaults', () => {
        const result = ProxyConfigSchema.safeParse({});
        expect(result.success).toBe(true);
        if (result.success) {
          expect(result.data.enabled).toBe(false);
          expect(result.data.codec).toBe('h264');
          expect(result.data.format).toBe('mp4');
          expect(result.data.preset).toBe('fast');
          expect(result.data.hwAccel).toBe('auto');
          expect(result.data.deinterlace).toBe(true);
          expect(result.data.fastStart).toBe(true);
        }
      });

      it('should validate codec options', () => {
        expect(ProxyConfigSchema.safeParse({ codec: 'h264' }).success).toBe(true);
        expect(ProxyConfigSchema.safeParse({ codec: 'h265' }).success).toBe(true);
        expect(ProxyConfigSchema.safeParse({ codec: 'prores' }).success).toBe(true);
        expect(ProxyConfigSchema.safeParse({ codec: 'invalid' }).success).toBe(false);
      });

      it('should validate format options', () => {
        expect(ProxyConfigSchema.safeParse({ format: 'mp4' }).success).toBe(true);
        expect(ProxyConfigSchema.safeParse({ format: 'mov' }).success).toBe(true);
        expect(ProxyConfigSchema.safeParse({ format: 'avi' }).success).toBe(false);
      });

      it('should validate hwAccel options', () => {
        expect(ProxyConfigSchema.safeParse({ hwAccel: 'auto' }).success).toBe(true);
        expect(ProxyConfigSchema.safeParse({ hwAccel: 'none' }).success).toBe(true);
        expect(ProxyConfigSchema.safeParse({ hwAccel: 'videotoolbox' }).success).toBe(true);
        expect(ProxyConfigSchema.safeParse({ hwAccel: 'nvenc' }).success).toBe(true);
        expect(ProxyConfigSchema.safeParse({ hwAccel: 'vaapi' }).success).toBe(true);
        expect(ProxyConfigSchema.safeParse({ hwAccel: 'qsv' }).success).toBe(true);
        expect(ProxyConfigSchema.safeParse({ hwAccel: 'invalid' }).success).toBe(false);
      });

      it('should validate preset options', () => {
        const validPresets = ['ultrafast', 'superfast', 'veryfast', 'faster', 'fast', 'medium', 'slow'];
        for (const preset of validPresets) {
          expect(ProxyConfigSchema.safeParse({ preset }).success).toBe(true);
        }
        expect(ProxyConfigSchema.safeParse({ preset: 'invalid' }).success).toBe(false);
      });

      it('should validate audioCodec options', () => {
        expect(ProxyConfigSchema.safeParse({ audioCodec: 'aac' }).success).toBe(true);
        expect(ProxyConfigSchema.safeParse({ audioCodec: 'copy' }).success).toBe(true);
        expect(ProxyConfigSchema.safeParse({ audioCodec: 'none' }).success).toBe(true);
        expect(ProxyConfigSchema.safeParse({ audioCodec: 'mp3' }).success).toBe(false);
      });

      it('should have default proxy sizes', () => {
        const result = ProxyConfigSchema.safeParse({});
        expect(result.success).toBe(true);
        if (result.success) {
          expect(result.data.sizes).toHaveProperty('small');
          expect(result.data.sizes).toHaveProperty('medium');
          expect(result.data.sizes).toHaveProperty('large');
          expect(result.data.sizes.small?.height).toBe(540);
          expect(result.data.sizes.medium?.height).toBe(720);
          expect(result.data.sizes.large?.height).toBe(1080);
        }
      });

      it('should accept custom LUT path', () => {
        const result = ProxyConfigSchema.safeParse({ lutPath: '/path/to/lut.cube' });
        expect(result.success).toBe(true);
        if (result.success) {
          expect(result.data.lutPath).toBe('/path/to/lut.cube');
        }
      });
    });
  });

  describe('Encoder Detection', () => {
    it('should detect available encoders', async () => {
      const encoders = await detectAvailableEncoders();
      expect(encoders).toBeInstanceOf(Set);
      // At minimum, software encoders should be detected
      expect(encoders.size).toBeGreaterThan(0);
    });

    it('should cache encoder detection results', async () => {
      const encoders1 = await detectAvailableEncoders();
      const encoders2 = await detectAvailableEncoders();
      // Should be the same cached instance
      expect(encoders1).toBe(encoders2);
    });
  });

  describe('Encoder Selection', () => {
    it('should select h264 encoder', async () => {
      const result = await selectEncoder('h264', 'auto');
      expect(result.encoder).toBeDefined();
      expect(result.hwType).toBeDefined();
      // Should be either hardware or software encoder
      expect(['h264_videotoolbox', 'h264_nvenc', 'h264_qsv', 'h264_vaapi', 'libx264']).toContain(result.encoder);
    });

    it('should fallback to software when hardware not available', async () => {
      // Force software encoding
      const result = await selectEncoder('h264', 'none');
      expect(result.encoder).toBe('libx264');
      expect(result.hwType).toBe('none');
    });

    it('should select h265 encoder', async () => {
      const result = await selectEncoder('h265', 'none');
      expect(result.encoder).toBe('libx265');
      expect(result.hwType).toBe('none');
    });
  });

  describe('Hardware Acceleration Info', () => {
    it('should return available acceleration options', async () => {
      const info = await getHardwareAccelInfo();
      expect(info.available).toBeInstanceOf(Array);
      expect(info.available.length).toBeGreaterThan(0);
      // Software is always available
      expect(info.available).toContain('none');
      expect(info.recommended).toBeDefined();
    });
  });

  describe('FFmpeg Encoding Support', () => {
    it('should check FFmpeg encoding support', async () => {
      const supported = await checkFfmpegEncodingSupport();
      expect(typeof supported).toBe('boolean');
    });
  });

  describe('Video Info for Proxy Generation', () => {
    it('should validate video info required for proxy generation', () => {
      const videoInfo = {
        duration: 120.5,
        width: 1920,
        height: 1080,
        frameRate: 29.97,
        codec: 'h264',
      };
      const result = VideoInfoSchema.safeParse(videoInfo);
      expect(result.success).toBe(true);
    });

    it('should include rotation info', () => {
      const videoInfo = {
        duration: 60,
        width: 1920,
        height: 1080,
        frameRate: 30,
        codec: 'h264',
        rotation: 90,
      };
      const result = VideoInfoSchema.safeParse(videoInfo);
      expect(result.success).toBe(true);
      if (result.success) {
        expect(result.data.rotation).toBe(90);
      }
    });

    it('should include HDR and interlacing info', () => {
      const videoInfo = {
        duration: 60,
        width: 3840,
        height: 2160,
        frameRate: 24,
        codec: 'hevc',
        isHdr: true,
        isInterlaced: false,
      };
      const result = VideoInfoSchema.safeParse(videoInfo);
      expect(result.success).toBe(true);
      if (result.success) {
        expect(result.data.isHdr).toBe(true);
        expect(result.data.isInterlaced).toBe(false);
      }
    });
  });
});
