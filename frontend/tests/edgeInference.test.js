import { describe, it, expect } from 'vitest';
import {
  calculateChromophoreRatio,
  calculateSpatialAutocorrelation,
  extractRetinalPixelStats,
  runEdgeInference,
  TARGET_CLASSES,
  CLINICAL_METADATA,
} from '../src/edgeInference';

describe('Edge Inference Engine & Optical Validator', () => {
  describe('Chromophore Ratio Calculation', () => {
    it('calculates expected ratio for typical fundus values', () => {
      const ratio = calculateChromophoreRatio(150, 50);
      expect(ratio).toBeCloseTo(151 / 51, 4);
    });

    it('prevents division by zero when blue channel is zero', () => {
      const ratio = calculateChromophoreRatio(100, 0);
      expect(ratio).toBe(101 / 1);
      expect(Number.isFinite(ratio)).toBe(true);
    });

    it('handles negative or NaN input gracefully', () => {
      const ratio = calculateChromophoreRatio(NaN, -10);
      expect(ratio).toBe(1.0);
      expect(Number.isFinite(ratio)).toBe(true);
    });
  });

  describe('Spatial Autocorrelation', () => {
    it('returns 0 for degenerate or undersized grids', () => {
      expect(calculateSpatialAutocorrelation(null, 0, 0)).toBe(0);
      expect(calculateSpatialAutocorrelation(new Float32Array([1]), 1, 1)).toBe(0);
    });

    it('returns 0 for completely flat zero-variance image without division by zero', () => {
      const flatGrid = new Float32Array(16).fill(128);
      const corr = calculateSpatialAutocorrelation(flatGrid, 4, 4);
      expect(corr).toBe(0.0);
      expect(Number.isFinite(corr)).toBe(true);
    });

    it('returns high autocorrelation for smooth spatial gradient', () => {
      const width = 16;
      const height = 16;
      const smoothGrid = new Float32Array(width * height);
      for (let y = 0; y < height; y++) {
        for (let x = 0; x < width; x++) {
          smoothGrid[y * width + x] = x * 10 + y * 10;
        }
      }
      const corr = calculateSpatialAutocorrelation(smoothGrid, width, height);
      expect(corr).toBeGreaterThan(0.85);
    });
  });

  describe('Pixel Statistics Extraction', () => {
    it('throws descriptive error on degenerate pixel array', () => {
      expect(() => extractRetinalPixelStats(new Uint8ClampedArray(0), 0, 0)).toThrow();
      expect(() => extractRetinalPixelStats(null, 10, 10)).toThrow();
    });

    it('clamps out-of-range RGB values to [0, 255]', () => {
      // 2x2 image = 4 pixels, 16 bytes
      const rawPixels = [
        300, -50, 999, 255,  // pixel 0
        100, 150, 200, 255,  // pixel 1
        50,  60,  70,  255,  // pixel 2
        80,  90,  100, 255,  // pixel 3
      ];
      const stats = extractRetinalPixelStats(rawPixels, 2, 2);
      expect(stats.meanRed).toBe((255 + 100 + 50 + 80) / 4);
      expect(stats.meanGreen).toBe((0 + 150 + 60 + 90) / 4);
      expect(stats.meanBlue).toBe((255 + 200 + 70 + 100) / 4);
      expect(stats.numPixels).toBe(4);
    });
  });

  describe('Edge Inference Boundary Handling', () => {
    it('returns structured failure for missing image input', async () => {
      const res = await runEdgeInference(null);
      expect(res.success).toBe(false);
      expect(res.error).toMatch(/no image source/i);
    });

    it('returns structured failure for empty (0-byte) Blob', async () => {
      const emptyBlob = new Blob([], { type: 'image/jpeg' });
      const res = await runEdgeInference(emptyBlob);
      expect(res.success).toBe(false);
      expect(res.error).toMatch(/empty/i);
    });

    it('provides clinical metadata for all target classes', () => {
      expect(TARGET_CLASSES.length).toBe(6);
      for (const cls of TARGET_CLASSES) {
        expect(CLINICAL_METADATA[cls]).toBeDefined();
        expect(CLINICAL_METADATA[cls].icd10).toBeDefined();
        expect(CLINICAL_METADATA[cls].snomed).toBeDefined();
        expect(CLINICAL_METADATA[cls].urgency).toBeDefined();
      }
    });
  });
});
