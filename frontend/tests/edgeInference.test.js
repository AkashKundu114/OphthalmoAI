import { describe, it, expect, vi, beforeEach, afterEach } from 'vitest';
import * as ort from 'onnxruntime-web';
import {
  loadEdgeModel,
  runEdgeInference,
  preprocessRetinalImage,
  softmax,
  resetEdgeModelSession,
  isEdgeModelLoaded,
  TARGET_CLASSES,
  CLINICAL_METADATA,
  calculateChromophoreRatio,
  calculateSpatialAutocorrelation,
  extractRetinalPixelStats,
} from '../src/edgeInference';
import { runHeuristicInference } from '../src/edgeHeuristic';

/**
 * Creates a synthetic HTMLCanvasElement with realistic retinal fundus pixels.
 * Uses red/orange dominance with vascular green elements and a smooth foveal gradient.
 */
function createSyntheticFundusCanvas(width = 224, height = 224) {
  const canvas = document.createElement('canvas');
  canvas.width = width;
  canvas.height = height;
  const ctx = canvas.getContext('2d');

  const imgData = ctx.createImageData(width, height);
  const data = imgData.data;
  const cx = width / 2;
  const cy = height / 2;
  const radius = width * 0.45;

  for (let y = 0; y < height; y++) {
    for (let x = 0; x < width; x++) {
      const idx = (y * width + x) * 4;
      const dist = Math.hypot(x - cx, y - cy);

      if (dist < radius) {
        // Typical fundus values: high red, moderate green, low blue
        const falloff = 1 - (dist / radius) * 0.3;
        data[idx] = Math.round(180 * falloff);     // Red
        data[idx + 1] = Math.round(90 * falloff);   // Green
        data[idx + 2] = Math.round(35 * falloff);   // Blue
        data[idx + 3] = 255;
      } else {
        // Black mask border
        data[idx] = 0;
        data[idx + 1] = 0;
        data[idx + 2] = 0;
        data[idx + 3] = 255;
      }
    }
  }

  ctx.putImageData(imgData, 0, 0);
  return canvas;
}

describe('Client-Side ONNX Edge Inference Engine', () => {
  beforeEach(() => {
    resetEdgeModelSession();
    vi.restoreAllMocks();
  });

  afterEach(() => {
    resetEdgeModelSession();
    vi.restoreAllMocks();
  });

  describe('1. ONNX Model Loading & Lifecycle', () => {
    it('loads ONNX model session successfully and caches instance', async () => {
      const mockSession = {
        inputNames: ['input'],
        outputNames: ['output'],
        run: vi.fn().mockResolvedValue({
          output: {
            data: new Float32Array([2.1, 0.4, 0.2, 0.1, 0.3, 0.1]),
            dims: [1, 6],
          },
        }),
      };

      const createSpy = vi.spyOn(ort.InferenceSession, 'create').mockResolvedValue(mockSession);

      const session1 = await loadEdgeModel('/models/edge_model.onnx');
      expect(session1).toBe(mockSession);
      expect(isEdgeModelLoaded()).toBe(true);
      expect(createSpy).toHaveBeenCalledTimes(1);

      // Verify memoization / caching: subsequent calls do not recreate session
      const session2 = await loadEdgeModel();
      expect(session2).toBe(mockSession);
      expect(createSpy).toHaveBeenCalledTimes(1);
    });

    it('handles ONNX model loading failures gracefully and resets state', async () => {
      vi.spyOn(ort.InferenceSession, 'create').mockRejectedValue(new Error('WASM binary missing or network error'));

      await expect(loadEdgeModel()).rejects.toThrow(/Failed to load ONNX edge model/i);
      expect(isEdgeModelLoaded()).toBe(false);
    });
  });

  describe('2. Ben Graham Image Preprocessing & Tensor Creation', () => {
    it('generates an ort.Tensor with exact NCHW shape [1, 3, 224, 224]', async () => {
      const canvas = createSyntheticFundusCanvas(300, 250);
      const { tensor, stats } = await preprocessRetinalImage(canvas, 224);

      expect(tensor).toBeDefined();
      expect(tensor.dims).toEqual([1, 3, 224, 224]);
      expect(tensor.type).toBe('float32');
      expect(tensor.data.length).toBe(1 * 3 * 224 * 224);

      // Verify optical statistics
      expect(stats.rawWidth).toBe(300);
      expect(stats.rawHeight).toBe(250);
      expect(stats.meanRed).toBeGreaterThan(stats.meanBlue);
    });

    it('normalizes pixel intensities with ImageNet statistics', async () => {
      const canvas = createSyntheticFundusCanvas(100, 100);
      const { tensor } = await preprocessRetinalImage(canvas, 224);
      const data = tensor.data;

      // Tensor values should be centered around zero, bounded roughly within [-3, 3]
      let minVal = Infinity;
      let maxVal = -Infinity;
      for (let i = 0; i < data.length; i++) {
        if (data[i] < minVal) minVal = data[i];
        if (data[i] > maxVal) maxVal = data[i];
      }

      expect(minVal).toBeGreaterThan(-3.5);
      expect(maxVal).toBeLessThan(3.5);
      expect(Number.isFinite(minVal)).toBe(true);
      expect(Number.isFinite(maxVal)).toBe(true);
    });
  });

  describe('3. Softmax & Probability Distribution', () => {
    it('produces probability distribution that sums to ~1.0', () => {
      const logits = [3.2, 1.1, -0.5, 0.2, -1.8, 2.0];
      const probs = softmax(logits);

      expect(probs.length).toBe(6);
      const sum = probs.reduce((a, b) => a + b, 0);
      expect(sum).toBeCloseTo(1.0, 5);

      // Highest logit should have the highest probability
      expect(probs[0]).toBeGreaterThan(probs[1]);
      expect(probs[0]).toBeGreaterThan(probs[5]);
    });

    it('safely handles extreme logits without NaN or infinite values', () => {
      const extremeLogits = [1000, 1005, 990, 800, -500, -1000];
      const probs = softmax(extremeLogits);

      expect(probs.length).toBe(6);
      for (const p of probs) {
        expect(Number.isFinite(p)).toBe(true);
        expect(p).toBeGreaterThanOrEqual(0.0);
        expect(p).toBeLessThanOrEqual(1.0);
      }
      const sum = probs.reduce((a, b) => a + b, 0);
      expect(sum).toBeCloseTo(1.0, 4);
    });
  });

  describe('4. ONNX Inference Execution & Latency Requirement', () => {
    it('runs inference through ONNX model, returns calibrated probabilities summing to ~1.0, and executes in < 200ms', async () => {
      // Mock fast ONNX session execution
      const mockSession = {
        inputNames: ['input'],
        outputNames: ['output'],
        run: vi.fn().mockImplementation(async () => {
          // Simulate realistic ONNX WASM execution delay (e.g., 15ms)
          await new Promise((r) => setTimeout(r, 15));
          return {
            output: {
              data: new Float32Array([3.5, 0.2, 0.1, 0.05, 0.1, 0.05]), // Strongly predicts "Normal"
              dims: [1, 6],
            },
          };
        }),
      };

      vi.spyOn(ort.InferenceSession, 'create').mockResolvedValue(mockSession);

      const canvas = createSyntheticFundusCanvas(224, 224);
      const t0 = performance.now();
      const result = await runEdgeInference(canvas);
      const elapsed = performance.now() - t0;

      expect(result.success).toBe(true);
      expect(result.isEdgeModel).toBe(true);
      expect(result.active_mode).toBe('onnx_wasm');
      expect(result.diagnosis).toBe('Normal');

      // Latency requirement: must complete in < 200ms
      expect(result.inferenceTimeMs).toBeLessThan(200);
      expect(elapsed).toBeLessThan(200);

      // Verify probability distribution sums to ~1.0
      expect(result.predictions).toBeDefined();
      expect(result.predictions.length).toBe(6);
      const probSum = result.predictions.reduce((acc, p) => acc + p.probability, 0);
      expect(probSum).toBeGreaterThanOrEqual(0.99);
      expect(probSum).toBeLessThanOrEqual(1.01);

      // Medical disclaimer verification
      expect(result.disclaimer).toMatch(/this is not a diagnosis/i);
      expect(result.disclaimer).toMatch(/edge screening - confirm with full pipeline/i);
      expect(result.warning).toMatch(/preliminary screening/i);
    });
  });

  describe('5. Graceful Heuristic Fallback when ONNX is Unavailable', () => {
    it('falls back to heuristic ONLY if ONNX model loading fails, showing active mode', async () => {
      // Simulate ONNX model missing (404 / network failure)
      vi.spyOn(ort.InferenceSession, 'create').mockRejectedValue(new Error('HTTP 404: Model not found'));

      const canvas = createSyntheticFundusCanvas(224, 224);
      const result = await runEdgeInference(canvas);

      expect(result.success).toBe(true);
      // Mode must be explicitly flagged as heuristic fallback, NOT real neural model
      expect(result.isEdgeModel).toBe(false);
      expect(result.isHeuristicFallback).toBe(true);
      expect(result.active_mode).toBe('heuristic_fallback');
      expect(result.engine).toMatch(/Legacy Heuristic Pre-Filter/i);

      // Predictions and probabilities should still be structured cleanly
      expect(result.diagnosis).toBeDefined();
      expect(result.probabilities).toBeDefined();
      expect(result.disclaimer).toMatch(/this is not a diagnosis/i);
      expect(result.warning).toMatch(/preliminary screening/i);
    });

    it('allows direct invocation of legacy heuristic via edgeHeuristic.js', async () => {
      const canvas = createSyntheticFundusCanvas(224, 224);
      const heuristicRes = await runHeuristicInference(canvas);

      expect(heuristicRes.success).toBe(true);
      expect(heuristicRes.isEdgeModel).toBe(false);
      expect(heuristicRes.isHeuristicFallback).toBe(true);
      expect(heuristicRes.active_mode).toBe('heuristic_fallback');
      expect(heuristicRes.engine).toMatch(/Legacy Heuristic/i);
    });
  });

  describe('6. Boundary Conditions & Quality Guardrails', () => {
    it('returns structured failure for missing image input', async () => {
      const res = await runEdgeInference(null);
      expect(res.success).toBe(false);
      expect(res.error).toMatch(/no image source/i);
    });

    it('rejects completely underexposed / black images', async () => {
      const blackCanvas = document.createElement('canvas');
      blackCanvas.width = 64;
      blackCanvas.height = 64;
      const ctx = blackCanvas.getContext('2d');
      ctx.fillStyle = '#000000';
      ctx.fillRect(0, 0, 64, 64);

      const res = await runEdgeInference(blackCanvas);
      expect(res.success).toBe(false);
      expect(res.error).toMatch(/underexposed or completely black/i);
    });

    it('provides clinical metadata for all 6 target classes', () => {
      expect(TARGET_CLASSES.length).toBe(6);
      for (const cls of TARGET_CLASSES) {
        expect(CLINICAL_METADATA[cls]).toBeDefined();
        expect(CLINICAL_METADATA[cls].icd10).toBeDefined();
        expect(CLINICAL_METADATA[cls].snomed).toBeDefined();
        expect(CLINICAL_METADATA[cls].urgency).toBeDefined();
      }
    });

    it('computes chromophore ratio and autocorrelation accurately', () => {
      const ratio = calculateChromophoreRatio(150, 50);
      expect(ratio).toBeCloseTo(151 / 51, 4);

      const flatGrid = new Float32Array(16).fill(100);
      expect(calculateSpatialAutocorrelation(flatGrid, 4, 4)).toBe(0.0);
    });
  });
});
