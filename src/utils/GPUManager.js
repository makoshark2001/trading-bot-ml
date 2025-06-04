const tf = require('@tensorflow/tfjs');

class GPUManager {
    constructor() {
        this.isGPUAvailable = false;
        this.gpuBackend = null;
        this.cpuBackend = 'cpu';
        this.currentBackend = null;
        this.initializationAttempted = false;
        this.gpuMemoryInfo = null;
        this.performanceMetrics = {
            gpuTrainingTime: [],
            cpuTrainingTime: [],
            lastBackendUsed: null
        };
        
        // GPU detection preferences (in order of preference)
        this.gpuBackends = [
            'webgl',  // Most compatible
            'nodejs-gpu'  // CUDA support if available
        ];
        
        // Initialize logger after avoiding circular dependency
        this.Logger = null;
        this.initializeLogger();
    }
    
    initializeLogger() {
        try {
            // Lazy load Logger to avoid circular dependency
            const Logger = require('./Logger');
            this.Logger = Logger;
        } catch (error) {
            // Fallback to console if Logger not available
            this.Logger = {
                info: console.log,
                warn: console.warn,
                error: console.error,
                debug: console.log
            };
        }
    }
    
    log(level, message, meta = {}) {
        if (this.Logger && this.Logger[level]) {
            this.Logger[level](message, meta);
        } else {
            const prefix = `[${level.toUpperCase()}]`;
            console.log(prefix, message, meta);
        }
    }
    
    async initialize() {
        if (this.initializationAttempted) {
            return this.isGPUAvailable;
        }
        
        this.initializationAttempted = true;
        
        this.log('info', '🔍 Detecting GPU capabilities...');
        
        try {
            // First, ensure TensorFlow is ready
            await tf.ready();
            
            // Try to initialize GPU backends
            for (const backend of this.gpuBackends) {
                try {
                    await this.tryInitializeBackend(backend);
                    if (this.isGPUAvailable) {
                        break;
                    }
                } catch (error) {
                    this.log('debug', `Failed to initialize ${backend} backend`, { 
                        error: error.message 
                    });
                }
            }
            
            // Fallback to CPU if no GPU available
            if (!this.isGPUAvailable) {
                await this.initializeCPUBackend();
            }
            
            // Log final status
            this.logGPUStatus();
            
            return this.isGPUAvailable;
            
        } catch (error) {
            this.log('error', 'GPU detection failed completely', { error: error.message });
            await this.initializeCPUBackend();
            return false;
        }
    }
    
    async tryInitializeBackend(backendName) {
        this.log('debug', `Trying to initialize ${backendName} backend...`);
        
        try {
            // Import the specific backend
            if (backendName === 'webgl') {
                require('@tensorflow/tfjs-backend-webgl');
            } else if (backendName === 'nodejs-gpu') {
                require('@tensorflow/tfjs-node-gpu');
            }
            
            // Set the backend
            await tf.setBackend(backendName);
            await tf.ready();
            
            // Test GPU functionality with a simple operation
            const testResult = await this.testGPUOperation();
            
            if (testResult.success) {
                this.isGPUAvailable = true;
                this.gpuBackend = backendName;
                this.currentBackend = backendName;
                this.gpuMemoryInfo = testResult.memoryInfo;
                
                this.log('info', `✅ GPU backend initialized: ${backendName}`, {
                    backend: backendName,
                    gpuMemory: testResult.memoryInfo,
                    testTime: testResult.executionTime + 'ms'
                });
                
                return true;
            } else {
                throw new Error(`GPU test operation failed: ${testResult.error}`);
            }
            
        } catch (error) {
            this.log('debug', `Backend ${backendName} initialization failed`, { 
                error: error.message 
            });
            
            // Try to fall back to CPU for this attempt
            try {
                await tf.setBackend('cpu');
                await tf.ready();
            } catch (fallbackError) {
                this.log('warn', 'Failed to fallback to CPU during GPU test', { 
                    error: fallbackError.message 
                });
            }
            
            throw error;
        }
    }
    
    async initializeCPUBackend() {
        try {
            require('@tensorflow/tfjs-backend-cpu');
            await tf.setBackend('cpu');
            await tf.ready();
            
            this.currentBackend = 'cpu';
            this.isGPUAvailable = false;
            
            this.log('info', '✅ CPU backend initialized (fallback)');
            
        } catch (error) {
            this.log('error', 'Failed to initialize CPU backend', { error: error.message });
            throw error;
        }
    }
    
    async testGPUOperation() {
        const startTime = Date.now();
        
        try {
            // Create test tensors
            const a = tf.randomNormal([100, 100]);
            const b = tf.randomNormal([100, 100]);
            
            // Perform matrix multiplication (GPU-intensive operation)
            const result = tf.matMul(a, b);
            
            // Force execution by reading data
            const data = await result.data();
            
            // Clean up tensors
            a.dispose();
            b.dispose();
            result.dispose();
            
            const executionTime = Date.now() - startTime;
            
            // Get memory info
            const memoryInfo = tf.memory();
            
            return {
                success: true,
                executionTime: executionTime,
                memoryInfo: {
                    numTensors: memoryInfo.numTensors,
                    numBytes: memoryInfo.numBytes,
                    backend: tf.getBackend()
                },
                dataLength: data.length
            };
            
        } catch (error) {
            return {
                success: false,
                error: error.message,
                executionTime: Date.now() - startTime
            };
        }
    }
    
    async switchToGPU() {
        if (!this.isGPUAvailable || this.currentBackend === this.gpuBackend) {
            return false;
        }
        
        try {
            await tf.setBackend(this.gpuBackend);
            await tf.ready();
            this.currentBackend = this.gpuBackend;
            
            this.log('info', `🚀 Switched to GPU backend: ${this.gpuBackend}`);
            return true;
            
        } catch (error) {
            this.log('error', 'Failed to switch to GPU backend', { error: error.message });
            await this.switchToCPU();
            return false;
        }
    }
    
    async switchToCPU() {
        if (this.currentBackend === 'cpu') {
            return true;
        }
        
        try {
            await tf.setBackend('cpu');
            await tf.ready();
            this.currentBackend = 'cpu';
            
            this.log('info', '🔄 Switched to CPU backend');
            return true;
            
        } catch (error) {
            this.log('error', 'Failed to switch to CPU backend', { error: error.message });
            throw error;
        }
    }
    
    async performWithGPUFallback(operation, operationName = 'operation') {
        let gpuAttempted = false;
        let startTime;
        
        try {
            // Try GPU first if available
            if (this.isGPUAvailable && this.currentBackend !== this.gpuBackend) {
                const switched = await this.switchToGPU();
                if (switched) {
                    gpuAttempted = true;
                    startTime = Date.now();
                    
                    this.log('debug', `🚀 Attempting ${operationName} on GPU`);
                    
                    try {
                        const result = await operation();
                        const duration = Date.now() - startTime;
                        
                        this.performanceMetrics.gpuTrainingTime.push(duration);
                        this.performanceMetrics.lastBackendUsed = 'gpu';
                        
                        this.log('info', `✅ GPU ${operationName} completed`, {
                            duration: duration + 'ms',
                            backend: this.currentBackend
                        });
                        
                        return result;
                        
                    } catch (gpuError) {
                        const duration = Date.now() - startTime;
                        
                        this.log('warn', `❌ GPU ${operationName} failed, falling back to CPU`, {
                            error: gpuError.message,
                            duration: duration + 'ms',
                            backend: this.currentBackend
                        });
                        
                        // Fall through to CPU attempt
                    }
                }
            }
            
            // CPU fallback (or primary if no GPU)
            await this.switchToCPU();
            startTime = Date.now();
            
            this.log('debug', `🔄 Performing ${operationName} on CPU`);
            
            const result = await operation();
            const duration = Date.now() - startTime;
            
            this.performanceMetrics.cpuTrainingTime.push(duration);
            this.performanceMetrics.lastBackendUsed = 'cpu';
            
            this.log('info', `✅ CPU ${operationName} completed`, {
                duration: duration + 'ms',
                backend: this.currentBackend,
                gpuAttempted: gpuAttempted
            });
            
            return result;
            
        } catch (error) {
            this.log('error', `❌ ${operationName} failed on both GPU and CPU`, {
                error: error.message,
                gpuAttempted: gpuAttempted,
                currentBackend: this.currentBackend
            });
            
            throw error;
        }
    }
    
    logGPUStatus() {
        const status = {
            gpuAvailable: this.isGPUAvailable,
            currentBackend: this.currentBackend,
            gpuBackend: this.gpuBackend,
            gpuMemoryInfo: this.gpuMemoryInfo
        };
        
        if (this.isGPUAvailable) {
            this.log('info', '🎮 GPU ACCELERATION ENABLED', status);
            console.log('🎮 GPU Training: ENABLED');
            console.log(`   • GPU Backend: ${this.gpuBackend}`);
            console.log(`   • Current Backend: ${this.currentBackend}`);
            console.log(`   • GPU Memory: ${this.gpuMemoryInfo?.numBytes || 'Unknown'} bytes`);
        } else {
            this.log('info', '💻 CPU-only mode active', status);
            console.log('💻 GPU Training: NOT AVAILABLE');
            console.log(`   • Current Backend: ${this.currentBackend}`);
            console.log('   • Reason: No compatible GPU found or GPU backend failed');
        }
    }
    
    getStatus() {
        return {
            gpuAvailable: this.isGPUAvailable,
            currentBackend: this.currentBackend,
            gpuBackend: this.gpuBackend,
            cpuBackend: this.cpuBackend,
            memoryInfo: tf.memory(),
            performanceMetrics: {
                ...this.performanceMetrics,
                avgGPUTime: this.performanceMetrics.gpuTrainingTime.length > 0 
                    ? this.performanceMetrics.gpuTrainingTime.reduce((a, b) => a + b, 0) / this.performanceMetrics.gpuTrainingTime.length 
                    : null,
                avgCPUTime: this.performanceMetrics.cpuTrainingTime.length > 0 
                    ? this.performanceMetrics.cpuTrainingTime.reduce((a, b) => a + b, 0) / this.performanceMetrics.cpuTrainingTime.length 
                    : null
            },
            initializationAttempted: this.initializationAttempted,
            timestamp: Date.now()
        };
    }
    
    getPerformanceComparison() {
        const metrics = this.performanceMetrics;
        
        if (metrics.gpuTrainingTime.length === 0 || metrics.cpuTrainingTime.length === 0) {
            return null;
        }
        
        const avgGPU = metrics.gpuTrainingTime.reduce((a, b) => a + b, 0) / metrics.gpuTrainingTime.length;
        const avgCPU = metrics.cpuTrainingTime.reduce((a, b) => a + b, 0) / metrics.cpuTrainingTime.length;
        
        return {
            avgGPUTime: Math.round(avgGPU),
            avgCPUTime: Math.round(avgCPU),
            speedupFactor: avgCPU / avgGPU,
            gpuSamples: metrics.gpuTrainingTime.length,
            cpuSamples: metrics.cpuTrainingTime.length,
            recommendation: avgGPU < avgCPU * 0.8 ? 'GPU preferred' : 'CPU adequate'
        };
    }
    
    // Clean up performance metrics to prevent memory leaks
    cleanupMetrics() {
        const maxSamples = 100;
        
        if (this.performanceMetrics.gpuTrainingTime.length > maxSamples) {
            this.performanceMetrics.gpuTrainingTime = this.performanceMetrics.gpuTrainingTime.slice(-maxSamples);
        }
        
        if (this.performanceMetrics.cpuTrainingTime.length > maxSamples) {
            this.performanceMetrics.cpuTrainingTime = this.performanceMetrics.cpuTrainingTime.slice(-maxSamples);
        }
    }
    
    dispose() {
        try {
            // Clean up any remaining tensors
            const memInfo = tf.memory();
            if (memInfo.numTensors > 0) {
                this.log('warn', `Disposing GPUManager with ${memInfo.numTensors} tensors still in memory`);
            }
            
            // Clear metrics
            this.performanceMetrics.gpuTrainingTime = [];
            this.performanceMetrics.cpuTrainingTime = [];
            
            this.log('info', 'GPUManager disposed');
            
        } catch (error) {
            console.error('Error disposing GPUManager:', error.message);
        }
    }
}

module.exports = GPUManager;