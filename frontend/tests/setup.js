import '@testing-library/jest-dom';

// JSDOM Canvas 2D Mock for Vitest test environment
if (typeof HTMLCanvasElement !== 'undefined') {
  HTMLCanvasElement.prototype.getContext = function (contextType) {
    if (contextType === '2d') {
      if (!this._context2d) {
        this._context2d = {
          canvas: this,
          fillStyle: '#000000',
          strokeStyle: '#000000',
          fillRect: () => {},
          clearRect: () => {},
          beginPath: () => {},
          closePath: () => {},
          arc: () => {},
          fill: () => {},
          stroke: () => {},
          createImageData: (w, h) => ({
            width: w,
            height: h,
            data: new Uint8ClampedArray(w * h * 4),
          }),
          getImageData: function (x, y, w, h) {
            const width = w || this.canvas.width || 224;
            const height = h || this.canvas.height || 224;
            const size = width * height * 4;
            if (!this.canvas._pixelData || this.canvas._pixelData.length !== size) {
              this.canvas._pixelData = new Uint8ClampedArray(size);
            }
            return {
              width,
              height,
              data: this.canvas._pixelData,
            };
          },
          putImageData: function (imgData) {
            this.canvas._pixelData = new Uint8ClampedArray(imgData.data);
          },
          drawImage: function (src) {
            if (!src || !src._pixelData) return;
            const srcW = src.width || 224;
            const srcH = src.height || 224;
            const dstW = this.canvas.width || 224;
            const dstH = this.canvas.height || 224;
            const dstData = new Uint8ClampedArray(dstW * dstH * 4);
            const srcData = src._pixelData;
            for (let y = 0; y < dstH; y++) {
              const srcY = Math.min(srcH - 1, Math.floor((y / dstH) * srcH));
              for (let x = 0; x < dstW; x++) {
                const srcX = Math.min(srcW - 1, Math.floor((x / dstW) * srcW));
                const srcIdx = (srcY * srcW + srcX) * 4;
                const dstIdx = (y * dstW + x) * 4;
                dstData[dstIdx] = srcData[srcIdx];
                dstData[dstIdx + 1] = srcData[srcIdx + 1];
                dstData[dstIdx + 2] = srcData[srcIdx + 2];
                dstData[dstIdx + 3] = srcData[srcIdx + 3];
              }
            }
            this.canvas._pixelData = dstData;
          },
        };
      }
      return this._context2d;
    }
    return null;
  };
}
