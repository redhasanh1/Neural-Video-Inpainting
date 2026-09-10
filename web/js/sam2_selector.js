/**
 * SAM2 Interactive Object Selector
 * Fast click-to-select objects with instant mask preview
 * Ported from SELECT_OBJECT_FAST.py for web integration
 */

class SAM2Selector {
    constructor(videoElement, canvasElement, options = {}) {
        this.video = videoElement;
        this.canvas = canvasElement;
        this.ctx = canvasElement.getContext('2d');

        this.options = {
            onMaskUpdate: null,
            onPointsChange: null,
            maskColor: [0, 255, 0], // Green overlay
            maskAlpha: 0.5,
            ...options
        };

        // Multi-object selection state
        this.selections = []; // Array of {id, points, mask, color}
        this.currentSelectionId = 0;
        this.currentSelection = null; // Active selection being built with multiple clicks
        this.isLoading = false;
        this.enabled = false; // Starts disabled - only enabled when ✨ Auto tool is selected

        // NUCLEAR NEON palette - maximum saturation, eye-melting colors
        this.colorPalette = [
            [  0, 255, 255],   // 0  CYBER-CYAN        – electric blade
            [255,   0, 255],   // 1  PSYCHEDELIC PINK  – neon sign on steroids
            [  0, 255,   0],   // 2  NUCLEAR LIME      – toxic waste highlight
            [255, 255,   0],   // 3  BLINDING YELLOW   – solar flare
            [255,   0, 255],   // 4  MAGENTA BEAM      – violet laser
            [  0, 128, 255],   // 5  ARC-BLUE          – lightning bolt
            [255, 128,   0],   // 6  PLASMA ORANGE     – molten copper
            [128, 255,   0],   // 7  ACID CHARTREUSE   – acid-trip highlighter
            [255,   0,   0],   // 8  PURE HELL-RED     – stop-sign on fire
            [  0, 255, 128],   // 9  SPRING SHOCK-GREEN– radioactive mint
            [255,   0, 128],   //10  ULTRA HOT-PINK    – barbie in a microwave
            [255, 255, 255],   //11  STAR WHITE        – pure highlight
            [128,   0, 255],   //12  QUANTUM VIOLET    – galaxy edge
        ];
        this.colorIndex = 0; // Tracks which color to use next

        // Video metadata
        this.videoWidth = 0;
        this.videoHeight = 0;
        this.currentFrameIndex = 0;

        // Display state
        this.displayScale = 1.0;
        this.offsetX = 0;
        this.offsetY = 0;

        this.init();
    }

    /**
     * Initialize canvas and event listeners
     */
    init() {
        // Set canvas size to match container
        this.updateCanvasSize();

        // Attach mouse event listeners
        this.canvas.addEventListener('click', (e) => this.handleClick(e));
        this.canvas.addEventListener('contextmenu', (e) => {
            e.preventDefault();
            this.handleRightClick(e);
        });

        // Attach touch event listeners for mobile
        this.canvas.addEventListener('touchstart', (e) => {
            if (!this.enabled) return; // Don't interfere when disabled
            if (e.cancelable) e.preventDefault(); // Prevent zoom/scroll (only if cancelable)
            if (e.touches.length === 1) {
                this.handleTouch(e.touches[0]);
            }
        }, { passive: false });

        this.canvas.addEventListener('touchend', (e) => {
            if (!this.enabled) return; // Don't interfere when disabled
            if (e.cancelable) e.preventDefault(); // Prevent ghost clicks (only if cancelable)
        }, { passive: false });

        // Update on window resize
        window.addEventListener('resize', () => this.updateCanvasSize());

        console.log('[SAM2Selector] Initialized');
    }

    /**
     * Calculate video render bounds (Theater mode - simplified, no letterbox)
     */
    calculateVideoRenderBounds() {
        if (!this.video.videoWidth) return;

        // In theater mode, canvas matches video display exactly
        const rect = this.canvas.getBoundingClientRect();
        this.renderWidth = rect.width;
        this.renderHeight = rect.height;
        this.offsetX = 0;
        this.offsetY = 0;

        console.log(`[SAM2Selector] Theater mode: ${this.renderWidth.toFixed(1)}x${this.renderHeight.toFixed(1)}`);
    }

    /**
     * Update canvas size to match video element
     */
    updateCanvasSize() {
        if (!this.video.videoWidth) return;

        this.videoWidth = this.video.videoWidth;
        this.videoHeight = this.video.videoHeight;

        // Get current display dimensions for scale calculation
        const canvasRect = this.canvas.getBoundingClientRect();
        const displayWidth = canvasRect.width;
        const displayHeight = canvasRect.height;

        // Guard against invalid rect (canvas not yet laid out)
        if (displayWidth < 10 || displayHeight < 10) {
            console.log('[SAM2Selector] Rect invalid, scheduling retry');
            setTimeout(() => this.updateCanvasSize(), 50);
            return;
        }

        // Set canvas internal resolution to native video resolution
        this.canvas.width = this.videoWidth;
        this.canvas.height = this.videoHeight;

        // Canvas display size is handled by CSS (100% width/height of parent)
        // No need to set style.width/height - it auto-resizes responsively

        // Calculate scale for coordinate conversion
        this.displayScaleX = displayWidth / this.videoWidth;
        this.displayScaleY = displayHeight / this.videoHeight;

        console.log(`[SAM2Selector] Canvas: ${this.canvas.width}x${this.canvas.height}, Display: ${displayWidth.toFixed(0)}x${displayHeight.toFixed(0)}`);

        // Calculate video render bounds for letterbox compensation
        this.calculateVideoRenderBounds();

        // Redraw with current mask
        this.draw();
    }

    /**
     * Handle left click - add positive point
     */
    handleClick(e) {
        console.log('[SAM2] handleClick called, enabled =', this.enabled);
        if (!this.enabled) {
            console.log('[SAM2] Returning early - disabled');
            return;
        }
        if (this.isLoading) return;

        const point = this.getCanvasPoint(e);
        if (!point) return; // Ignore clicks in letterbox area
        this.addPoint(point.x, point.y, 1);
    }

    /**
     * Handle right click - add negative point
     */
    handleRightClick(e) {
        if (!this.enabled) return;
        if (this.isLoading) return;

        const point = this.getCanvasPoint(e);
        if (!point) return; // Ignore clicks in letterbox area
        this.addPoint(point.x, point.y, 0);
    }

    /**
     * Handle touch - add positive point (single tap = left click)
     */
    handleTouch(touch) {
        if (!this.enabled) return;
        if (this.isLoading) return;

        // Touch object has clientX/clientY like mouse events
        const point = this.getCanvasPoint(touch);
        if (!point) return;
        this.addPoint(point.x, point.y, 1);
    }

    /**
     * Convert mouse/touch event to canvas coordinates
     * Uses pre-calculated displayScaleX/Y from updateCanvasSize() for reliability
     */
    getCanvasPoint(e) {
        const rect = this.canvas.getBoundingClientRect();
        const clickX = e.clientX - rect.left;
        const clickY = e.clientY - rect.top;

        // Use pre-calculated scale (set in updateCanvasSize) - simpler and works on Android
        const videoX = Math.floor(clickX / this.displayScaleX);
        const videoY = Math.floor(clickY / this.displayScaleY);

        // Clamp to video bounds
        return {
            x: Math.max(0, Math.min(videoX, this.videoWidth - 1)),
            y: Math.max(0, Math.min(videoY, this.videoHeight - 1)),
            displayX: clickX,
            displayY: clickY
        };
    }

    /**
     * Add point as a new selection (each click = new object with new color)
     */
    async addPoint(x, y, label) {
        // Limit to 10 clicks per session
        const totalPoints = this.getAllPoints().length;
        if (totalPoints >= 10) {
            console.log('[SAM2Selector] Maximum 10 clicks reached');
            alert('Maximum 10 selection points allowed. Click Reset to start over.');
            return;
        }

        // Each click creates a NEW selection with its own color
        const color = this.colorPalette[this.colorIndex % this.colorPalette.length];
        this.colorIndex++;

        this.currentSelection = {
            id: this.currentSelectionId++,
            points: [{x, y, label}],  // Single point for this selection
            mask: null,
            color: color,
            frame_idx: this.currentFrameIndex  // Store frame at click time
        };

        console.log(`[SAM2Selector] New selection #${this.currentSelection.id} at (${x}, ${y}) with color ${this.colorIndex - 1}`);

        // Fire callback IMMEDIATELY when point is added (before waiting for mask)
        if (this.options.onPointsChange) {
            this.options.onPointsChange(this.getAllPoints());
        }

        // Request mask from server for this single point
        const mask = await this.requestMask(this.currentSelection.points);

        if (mask) {
            // Update current selection's mask
            this.currentSelection.mask = mask;

            // Always add as new selection (each click is separate)
            // Deep copy to avoid shared references
            this.selections.push({
                id: this.currentSelection.id,
                points: this.currentSelection.points.map(p => ({...p})),
                mask: this.currentSelection.mask,  // ImageData is immutable
                color: [...this.currentSelection.color],
                frame_idx: this.currentSelection.frame_idx
            });

            console.log(`[SAM2Selector] Added selection #${this.currentSelection.id} with color index ${this.colorIndex - 1}`);

            // Callback for selection changes
            if (this.options.onPointsChange) {
                this.options.onPointsChange(this.getAllPoints());
            }
        }

        // Redraw all masks
        this.draw();
    }

    /**
     * Request mask from server for given points
     */
    async requestMask(points) {
        if (points.length === 0) {
            return null;
        }

        this.isLoading = true;

        try {
            // Get current frame as base64
            const frameData = this.captureCurrentFrame();

            // Send to server
            const response = await fetch('/api/sam2/select-object', {
                method: 'POST',
                headers: {'Content-Type': 'application/json'},
                body: JSON.stringify({
                    frame_data: frameData,
                    frame_index: this.currentFrameIndex,
                    points: points,
                    video_width: this.videoWidth,
                    video_height: this.videoHeight
                })
            });

            const data = await response.json();

            if (data.status === 'success' && data.mask) {
                // Decode mask from base64
                const mask = await this.decodeMask(data.mask);
                console.log(`[SAM2Selector] Mask received (${mask.width}x${mask.height})`);
                return mask;
            } else {
                console.error('[SAM2Selector] Failed to get mask:', data.message);
                return null;
            }
        } catch (error) {
            console.error('[SAM2Selector] Mask request failed:', error);
            return null;
        } finally {
            this.isLoading = false;
        }
    }

    /**
     * Capture current video frame as base64
     */
    captureCurrentFrame() {
        const tempCanvas = document.createElement('canvas');
        tempCanvas.width = this.videoWidth;
        tempCanvas.height = this.videoHeight;
        const tempCtx = tempCanvas.getContext('2d');

        tempCtx.drawImage(this.video, 0, 0, this.videoWidth, this.videoHeight);

        // Use PNG for lossless quality (no JPEG compression artifacts)
        return tempCanvas.toDataURL('image/png').split(',')[1];
    }

    /**
     * Decode mask from base64 PNG
     */
    async decodeMask(base64Mask) {
        return new Promise((resolve, reject) => {
            const img = new Image();
            img.onload = () => {
                const tempCanvas = document.createElement('canvas');
                tempCanvas.width = img.width;
                tempCanvas.height = img.height;
                const tempCtx = tempCanvas.getContext('2d');
                tempCtx.drawImage(img, 0, 0);

                const imageData = tempCtx.getImageData(0, 0, img.width, img.height);
                resolve(imageData);
            };
            img.onerror = reject;
            img.src = 'data:image/png;base64,' + base64Mask;
        });
    }

    /**
     * Draw current state (video frame + mask overlay + points)
     */
    draw() {
        if (!this.videoWidth) return;

        // Clear canvas (transparent - video shows through underneath)
        this.ctx.clearRect(0, 0, this.canvas.width, this.canvas.height);

        // DON'T draw video - let <video> element handle it
        // Canvas is now overlay-only, positioned on top of video
        // This prevents sync issues when scrubbing to different frames

        // Draw all mask overlays
        this.drawAllMasks();

        // Draw all points
        this.drawAllPoints();

        // Draw loading indicator
        if (this.isLoading) {
            this.drawLoadingIndicator();
        }
    }

    /**
     * Draw all mask overlays with transparency
     */
    drawAllMasks() {
        if (this.selections.length === 0) return;

        // Create temporary canvas for the mask overlay
        const tempCanvas = document.createElement('canvas');
        tempCanvas.width = this.canvas.width;
        tempCanvas.height = this.canvas.height;
        const tempCtx = tempCanvas.getContext('2d');

        // Create ImageData to build mask
        const maskData = tempCtx.createImageData(this.canvas.width, this.canvas.height);

        // Loop through all selections and composite their masks with unique colors
        this.selections.forEach(selection => {
            const mask = selection.mask;
            if (!mask) return;

            // Get this selection's color (fallback to cyan if not set)
            const color = selection.color || [0, 212, 255];

            // Resize mask data to match canvas
            for (let y = 0; y < this.canvas.height; y++) {
                for (let x = 0; x < this.canvas.width; x++) {
                    // Map to mask coordinates
                    const maskX = Math.floor(x / this.canvas.width * mask.width);
                    const maskY = Math.floor(y / this.canvas.height * mask.height);
                    const maskIdx = (maskY * mask.width + maskX) * 4;

                    // If mask pixel is white (255), apply this selection's color
                    if (mask.data[maskIdx] > 127) {
                        const idx = (y * this.canvas.width + x) * 4;
                        maskData.data[idx] = color[0];     // R
                        maskData.data[idx + 1] = color[1]; // G
                        maskData.data[idx + 2] = color[2]; // B
                        maskData.data[idx + 3] = 255; // Full alpha on temp canvas
                    }
                }
            }
        });

        // Put mask data on temp canvas
        tempCtx.putImageData(maskData, 0, 0);

        // Draw overlay with transparency using globalAlpha (proper compositing)
        this.ctx.save();
        this.ctx.globalAlpha = this.options.maskAlpha;
        this.ctx.drawImage(tempCanvas, 0, 0);
        this.ctx.restore();
    }

    /**
     * Convert RGB array to hex color string
     */
    rgbToHex(rgb) {
        return '#' + rgb.map(c => c.toString(16).padStart(2, '0')).join('');
    }

    /**
     * Draw all selection points with matching colors
     */
    drawAllPoints() {
        this.selections.forEach(selection => {
            // Get selection's color (for positive points), red for negative
            const selectionColor = selection.color || [0, 212, 255];
            const positiveHex = this.rgbToHex(selectionColor);

            selection.points.forEach(point => {
                // Points are stored in canvas coordinates (native video resolution)
                // Draw directly since canvas is at native resolution
                const x = point.x;
                const y = point.y;
                // Positive points use selection color, negative always red
                const color = point.label ? positiveHex : '#FF0000';

                // Draw circle
                this.ctx.beginPath();
                this.ctx.arc(x, y, 6, 0, Math.PI * 2);
                this.ctx.fillStyle = color;
                this.ctx.fill();
                this.ctx.strokeStyle = '#FFFFFF';
                this.ctx.lineWidth = 2;
                this.ctx.stroke();

                // Draw crosshair
                this.ctx.strokeStyle = color;
                this.ctx.lineWidth = 2;
                this.ctx.beginPath();
                this.ctx.moveTo(x - 12, y);
                this.ctx.lineTo(x + 12, y);
                this.ctx.moveTo(x, y - 12);
                this.ctx.lineTo(x, y + 12);
                this.ctx.stroke();
            });
        });
    }

    /**
     * Draw loading indicator
     */
    drawLoadingIndicator() {
        this.ctx.fillStyle = 'rgba(0, 0, 0, 0.5)';
        this.ctx.fillRect(0, 0, this.canvas.width, this.canvas.height);

        this.ctx.fillStyle = '#FFFFFF';
        this.ctx.font = '16px Arial';
        this.ctx.textAlign = 'center';
        this.ctx.fillText('Generating mask...', this.canvas.width / 2, this.canvas.height / 2);
    }

    /**
     * Reset all selections
     */
    reset() {
        this.selections = [];
        this.currentSelection = null; // Clear current selection too
        this.currentSelectionId = 0;
        this.colorIndex = 0; // Reset color rotation
        this.draw();

        console.log('[SAM2Selector] Reset all selections');

        if (this.options.onPointsChange) {
            this.options.onPointsChange([]);
        }
    }

    /**
     * Get all points from all selections
     */
    getAllPoints() {
        const allPoints = [];
        this.selections.forEach(selection => {
            allPoints.push(...selection.points);
        });
        // Include current selection being built (before mask response)
        if (this.currentSelection && this.currentSelection.points) {
            allPoints.push(...this.currentSelection.points);
        }
        return allPoints;
    }

    /**
     * Update when video time changes
     */
    onVideoTimeUpdate() {
        // Estimate FPS (default 30 if unknown)
        const fps = this.videoFps || 30;
        const newFrameIndex = Math.floor(this.video.currentTime * fps);

        if (newFrameIndex !== this.currentFrameIndex) {
            this.currentFrameIndex = newFrameIndex;
            console.log(`[SAM2Selector] Frame changed to ${newFrameIndex}`);
            this.draw();
        }
    }

    /**
     * Set video FPS for accurate frame calculation
     */
    setVideoFps(fps) {
        this.videoFps = fps;
        console.log(`[SAM2Selector] Video FPS set to ${fps}`);
    }

    /**
     * Get current selection data for processing
     */
    getSelectionData() {
        return {
            points: this.getAllPoints(),
            selections: this.selections,
            frame_index: this.currentFrameIndex,
            video_width: this.videoWidth,
            video_height: this.videoHeight,
            has_mask: this.selections.length > 0
        };
    }

    /**
     * Enable touch/click selection (sets pointer-events: auto)
     */
    enable() {
        this.enabled = true;
        this.canvas.style.pointerEvents = 'auto';
        this.canvas.style.cursor = 'crosshair';
        console.log('[SAM2Selector] Enabled - touch/click active');
    }

    /**
     * Disable touch/click selection (sets pointer-events: none)
     */
    disable() {
        this.enabled = false;
        this.canvas.style.pointerEvents = 'none';
        console.log('[SAM2Selector] Disabled');
    }
}

// Export for use in other scripts
window.SAM2Selector = SAM2Selector;
