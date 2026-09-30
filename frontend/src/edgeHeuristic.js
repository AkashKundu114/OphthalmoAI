/**
 * Legacy heuristic pre-filter - NOT for clinical use.
 * 
 * Historical pixel-statistics heuristic pre-filter kept solely as an offline fallback
 * if the real ONNX neural runtime cannot be initialized by the client browser.
 * 
 * WARNING: This is NOT real ML inference and must never be used for clinical decisions.
 */

export const TARGET_CLASSES = [
  "Normal",
  "Diabetic Retinopathy",
  "Glaucoma",
  "Cataract",
  "Age-related Macular Degeneration",
  "Hypertensive Retinopathy"
];

export const CLINICAL_METADATA = {
  "Normal": {
    icd10: "Z01.00",
    snomed: "17621005",
    urgency: "None",
    description: "Healthy retinal morphology with crisp foveal avascular zone and intact neuroretinal rim.",
  },
  "Diabetic Retinopathy": {
    icd10: "E11.319",
    snomed: "4855003",
    urgency: "Urgent",
    description: "Microvascular lesions, focal hemorrhages, or exudates indicative of diabetic microangiopathy.",
  },
  "Glaucoma": {
    icd10: "H40.9",
    snomed: "23986001",
    urgency: "Urgent",
    description: "Optic disc cup enlargement or neuroretinal rim thinning suggestive of glaucomatous optic neuropathy.",
  },
  "Cataract": {
    icd10: "H25.9",
    snomed: "193570009",
    urgency: "Elective",
    description: "Optical media opacity causing general luminance attenuation and vessel margin blurring.",
  },
  "Age-related Macular Degeneration": {
    icd10: "H35.30",
    snomed: "267718000",
    urgency: "Urgent",
    description: "Macular drusen confluence or geographic retinal pigment epithelium degeneration.",
  },
  "Hypertensive Retinopathy": {
    icd10: "H35.00",
    snomed: "39934008",
    urgency: "Urgent",
    description: "Arteriolar narrowing, arteriovenous nicking, or systemic vascular copper-wiring signs.",
  },
};

/**
 * Calculates the red-to-blue chromophore backscatter ratio with division-by-zero protection.
 */
export function calculateChromophoreRatio(meanRed, meanBlue) {
  const safeRed = Math.max(0, Number.isFinite(meanRed) ? meanRed : 0);
  const safeBlue = Math.max(0, Number.isFinite(meanBlue) ? meanBlue : 0);
  return (safeRed + 1.0) / (safeBlue + 1.0);
}

/**
 * Computes spatial lag-1 horizontal and vertical autocorrelation to reject uncorrelated noise.
 */
export function calculateSpatialAutocorrelation(luminanceGrid, width, height) {
  if (!luminanceGrid || width < 2 || height < 2 || luminanceGrid.length < width * height) {
    return 0.0;
  }

  // 1. Horizontal lag-1 correlation
  let sumH1 = 0, sumH2 = 0, sumH1Sq = 0, sumH2Sq = 0, sumHProd = 0;
  const numHorizontalPairs = height * (width - 1);

  for (let y = 0; y < height; y++) {
    const rowOffset = y * width;
    for (let x = 0; x < width - 1; x++) {
      const val1 = luminanceGrid[rowOffset + x];
      const val2 = luminanceGrid[rowOffset + x + 1];
      sumH1 += val1;
      sumH2 += val2;
      sumH1Sq += val1 * val1;
      sumH2Sq += val2 * val2;
      sumHProd += val1 * val2;
    }
  }

  const covH = sumHProd - (sumH1 * sumH2) / numHorizontalPairs;
  const varH1 = Math.max(0, sumH1Sq - (sumH1 * sumH1) / numHorizontalPairs);
  const varH2 = Math.max(0, sumH2Sq - (sumH2 * sumH2) / numHorizontalPairs);
  const denomH = Math.sqrt(varH1 * varH2);
  const corrH = denomH > 1e-6 ? Math.max(-1.0, Math.min(1.0, covH / denomH)) : 0.0;

  // 2. Vertical lag-1 correlation
  let sumV1 = 0, sumV2 = 0, sumV1Sq = 0, sumV2Sq = 0, sumVProd = 0;
  const numVerticalPairs = (height - 1) * width;

  for (let y = 0; y < height - 1; y++) {
    const rowOffset1 = y * width;
    const rowOffset2 = (y + 1) * width;
    for (let x = 0; x < width; x++) {
      const val1 = luminanceGrid[rowOffset1 + x];
      const val2 = luminanceGrid[rowOffset2 + x];
      sumV1 += val1;
      sumV2 += val2;
      sumV1Sq += val1 * val1;
      sumV2Sq += val2 * val2;
      sumVProd += val1 * val2;
    }
  }

  const covV = sumVProd - (sumV1 * sumV2) / numVerticalPairs;
  const varV1 = Math.max(0, sumV1Sq - (sumV1 * sumV1) / numVerticalPairs);
  const varV2 = Math.max(0, sumV2Sq - (sumV2 * sumV2) / numVerticalPairs);
  const denomV = Math.sqrt(varV1 * varV2);
  const corrV = denomV > 1e-6 ? Math.max(-1.0, Math.min(1.0, covV / denomV)) : 0.0;

  const meanAutocorr = (corrH + corrV) / 2.0;
  return Math.max(0.0, Math.min(1.0, meanAutocorr));
}

/**
 * Extracts channel averages, center quadrant luminance, and spatial arrays from raw pixels.
 */
export function extractRetinalPixelStats(pixelData, width, height) {
  if (!pixelData || pixelData.length < 4 || width <= 0 || height <= 0) {
    throw new Error("Invalid pixel data array or non-positive dimensions.");
  }

  const expectedLength = width * height * 4;
  const numPixels = Math.floor(Math.min(pixelData.length, expectedLength) / 4);
  if (numPixels <= 0) {
    throw new Error("Degenerate pixel array with zero measurable pixels.");
  }

  const luminanceGrid = new Float32Array(numPixels);
  let totalRed = 0;
  let totalGreen = 0;
  let totalBlue = 0;
  let centerRed = 0;
  let centerPixelCount = 0;

  const centerMinX = Math.floor(width * 0.25);
  const centerMaxX = Math.floor(width * 0.75);
  const centerMinY = Math.floor(height * 0.25);
  const centerMaxY = Math.floor(height * 0.75);

  for (let i = 0; i < numPixels; i++) {
    const offset = i * 4;
    const rawR = pixelData[offset];
    const rawG = pixelData[offset + 1];
    const rawB = pixelData[offset + 2];

    const r = Number.isFinite(rawR) ? Math.max(0, Math.min(255, rawR)) : 0;
    const g = Number.isFinite(rawG) ? Math.max(0, Math.min(255, rawG)) : 0;
    const b = Number.isFinite(rawB) ? Math.max(0, Math.min(255, rawB)) : 0;

    totalRed += r;
    totalGreen += g;
    totalBlue += b;

    const luminance = 0.299 * r + 0.587 * g + 0.114 * b;
    luminanceGrid[i] = luminance;

    const x = i % width;
    const y = Math.floor(i / width);
    if (x >= centerMinX && x <= centerMaxX && y >= centerMinY && y <= centerMaxY) {
      centerRed += r;
      centerPixelCount++;
    }
  }

  const meanRed = totalRed / numPixels;
  const meanGreen = totalGreen / numPixels;
  const meanBlue = totalBlue / numPixels;
  const centerMeanRed = centerPixelCount > 0 ? centerRed / centerPixelCount : meanRed;

  return {
    meanRed,
    meanGreen,
    meanBlue,
    centerMeanRed,
    numPixels,
    luminanceGrid,
  };
}

/**
 * Helper to load HTMLImageElement from element, File, or Blob.
 */
export async function loadImageElement(sourceImage) {
  if (!sourceImage) {
    throw new Error("No image source provided for edge optical validation.");
  }

  if (typeof HTMLImageElement !== 'undefined' && sourceImage instanceof HTMLImageElement) {
    if (sourceImage.naturalWidth === 0 || sourceImage.naturalHeight === 0) {
      throw new Error("Supplied image element has zero width or height.");
    }
    return sourceImage;
  }

  if (typeof HTMLCanvasElement !== 'undefined' && sourceImage instanceof HTMLCanvasElement) {
    if (sourceImage.width === 0 || sourceImage.height === 0) {
      throw new Error("Supplied canvas has zero width or height.");
    }
    return sourceImage;
  }

  if (typeof Blob !== 'undefined' && sourceImage instanceof Blob) {
    if (sourceImage.size === 0) {
      throw new Error("Supplied image file is empty (0 bytes).");
    }

    const objectUrl = URL.createObjectURL(sourceImage);
    return new Promise((resolve, reject) => {
      const img = new Image();
      img.onload = () => {
        URL.revokeObjectURL(objectUrl);
        if (img.naturalWidth === 0 || img.naturalHeight === 0) {
          reject(new Error("Decoded image has zero width or height."));
        } else {
          resolve(img);
        }
      };
      img.onerror = () => {
        URL.revokeObjectURL(objectUrl);
        reject(new Error("Failed to decode image data into HTMLImageElement."));
      };
      img.src = objectUrl;
    });
  }

  throw new Error("Unsupported image source: expected HTMLImageElement, File, or Blob.");
}

/**
 * Runs legacy heuristic triage inference.
 * Renamed to runHeuristicInference for fallback use when ONNX runtime is unavailable.
 */
export async function runHeuristicInference(sourceImage) {
  const startTime = performance.now();

  try {
    const imgElement = await loadImageElement(sourceImage);

    const TARGET_DIMENSION = 224;
    const canvas = document.createElement("canvas");
    canvas.width = TARGET_DIMENSION;
    canvas.height = TARGET_DIMENSION;

    const ctx = canvas.getContext("2d", { willReadFrequently: true });
    if (!ctx) {
      return {
        success: false,
        error: "Hardware limitation: unable to acquire canvas 2D rendering context.",
      };
    }

    ctx.drawImage(imgElement, 0, 0, TARGET_DIMENSION, TARGET_DIMENSION);

    const imgData = ctx.getImageData(0, 0, TARGET_DIMENSION, TARGET_DIMENSION);
    if (!imgData || !imgData.data || imgData.data.length === 0) {
      return {
        success: false,
        error: "Unable to extract pixel data from image canvas.",
      };
    }

    const stats = extractRetinalPixelStats(imgData.data, TARGET_DIMENSION, TARGET_DIMENSION);
    const { meanRed, meanGreen, meanBlue, centerMeanRed, luminanceGrid } = stats;

    if (meanRed < 15 && meanGreen < 15 && meanBlue < 15) {
      return {
        success: false,
        error: "Edge Optical Validator: Image is underexposed or completely black.",
      };
    }

    const spatialCorr = calculateSpatialAutocorrelation(luminanceGrid, TARGET_DIMENSION, TARGET_DIMENSION);
    if (spatialCorr < 0.30) {
      return {
        success: false,
        error: "Edge Optical Validator: Image contains uncorrelated noise or lacks anatomical retinal structure.",
      };
    }

    const chromophoreRatio = calculateChromophoreRatio(meanRed, meanBlue);
    const isFundusProfile = chromophoreRatio >= 1.05 && meanRed > 25;

    if (!isFundusProfile) {
      return {
        success: false,
        error: "Edge Optical Validator: Upload does not match retinal fundus chromophore profile (R/B ratio too low). Please provide an authentic retinal scan.",
      };
    }

    const contrastRatio = centerMeanRed / Math.max(1.0, meanRed);
    const vascularGreenDominance = meanGreen / Math.max(1.0, meanRed);

    const rawScores = {
      "Normal": 1.00 + (vascularGreenDominance >= 0.45 && vascularGreenDominance <= 0.60 && contrastRatio >= 0.95 && contrastRatio <= 1.10 ? 1.5 : 0),
      "Diabetic Retinopathy": 1.00 + (meanGreen < 68 && vascularGreenDominance < 0.45 ? 1.6 : 0),
      "Glaucoma": 1.00 + (contrastRatio > 1.12 ? 1.6 : 0),
      "Cataract": 1.00 + (meanBlue > 80 || (meanGreen > 95 && contrastRatio < 0.98) ? 1.7 : 0),
      "Age-related Macular Degeneration": 1.00 + (contrastRatio < 0.92 ? 1.5 : 0),
      "Hypertensive Retinopathy": 1.00 + (vascularGreenDominance < 0.38 ? 1.4 : 0),
    };

    const TEMPERATURE = 1.20;
    const scoreValues = Object.values(rawScores);
    const maxScore = Math.max(...scoreValues);

    const expScores = {};
    let sumExp = 0;
    for (const cls of TARGET_CLASSES) {
      const expVal = Math.exp(((rawScores[cls] ?? 1.0) - maxScore) / TEMPERATURE);
      expScores[cls] = expVal;
      sumExp += expVal;
    }

    const probabilities = {};
    let topClass = TARGET_CLASSES[0];
    let maxProb = -1;

    for (const cls of TARGET_CLASSES) {
      const p = sumExp > 0 ? expScores[cls] / sumExp : 1 / TARGET_CLASSES.length;
      const roundedProb = Math.round(p * 1000) / 1000;
      probabilities[cls] = roundedProb;
      if (p > maxProb) {
        maxProb = p;
        topClass = cls;
      }
    }

    const elapsedMs = Math.round(performance.now() - startTime);
    const meta = CLINICAL_METADATA[topClass] || {
      icd10: "H57.9",
      snomed: "371405004",
      urgency: "Elective",
      description: "Screening completed. Confirmatory evaluation recommended.",
    };

    const predictions = TARGET_CLASSES.map((cls) => ({
      class: cls,
      probability: probabilities[cls] ?? 0.0,
    })).sort((a, b) => b.probability - a.probability);

    return {
      success: true,
      edge_mode: true,
      isEdgeModel: false,
      isHeuristicFallback: true,
      active_mode: "heuristic_fallback",
      engine: "Legacy Heuristic Pre-Filter (Fallback - NOT for clinical use)",
      predictions,
      inferenceTimeMs: elapsedMs,
      latency_ms: elapsedMs,
      disclaimer: "Legacy heuristic pre-filter - NOT for clinical use. Full server-side analysis recommended for clinical decisions. This is not a diagnosis.",
      warning: "Warning: Preliminary screening (legacy heuristic fallback). Full server-side analysis recommended for clinical decisions.",
      diagnosis: topClass,
      confidence: Math.round(maxProb * 1000) / 10,
      probabilities,
      icd10_code: meta.icd10,
      snomed_code: meta.snomed,
      urgency: meta.urgency,
      details: {
        description: meta.description,
        advice: "Legacy heuristic fallback pre-filter. Confirmatory review by an eye care specialist is required.",
      },
      privacy: "100% Client-Side. Image was never transmitted over the internet."
    };
  } catch (error) {
    return {
      success: false,
      isEdgeModel: false,
      error: error instanceof Error ? error.message : "Unknown error during heuristic inference.",
    };
  }
}
