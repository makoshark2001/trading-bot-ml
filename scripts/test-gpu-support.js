const tf = require('@tensorflow/tfjs');
const { GPUManager, Logger } = require('../src/utils');
const LSTMModel = require('../src/models/LSTMModel');

async function testGPUSupport() {
    console.log('🎮 GPU Support Detection and Performance Test');
    console.log('=' * 60);
    
    const gpuManager = new GPUManager();
    
    try {
        // Test 1: Basic GPU Detection
        console.log('\n📊 Test 1: GPU Detection');
        console.log('-' * 30);
        
        const gpuAvailable = await gpuManager.initialize();
        const status = gpuManager.getStatus();
        
        console.log('GPU Available:', gpuAvailable ? '✅ YES' : '❌ NO');
        console.log('Current Backend:', status.currentBackend);
        console.log('GPU Backend:', status.gpuBackend || 'None');
        console.log('Memory Info:', JSON.stringify(status.memoryInfo, null, 2));
        
        // Test 2: Basic Tensor Operations Performance
        console.log('\n🚀 Test 2: Tensor Operations Performance');
        console.log('-' * 40);
        
        await testTensorOperations(gpuManager);
        
        // Test 3: LSTM Model Training Performance
        console.log('\n🧠 Test 3: LSTM Model Training Performance');
        console.log('-' * 45);
        
        await testLSTMTraining(gpuManager);
        
        // Test 4: Performance Comparison
        console.log('\n📈 Test 4: Performance Comparison');
        console.log('-' * 35);
        
        const perfComparison = gpuManager.getPerformanceComparison();
        if (perfComparison) {
            console.log('Average GPU Time:', perfComparison.avgGPUTime + 'ms');
            console.log('Average CPU Time:', perfComparison.avgCPUTime + 'ms');
            console.log('GPU Speedup Factor:', perfComparison.speedupFactor.toFixed(2) + 'x');
            console.log('Recommendation:', perfComparison.recommendation);
        } else {
            console.log('❌ No performance data available for comparison');
        }
        
        console.log('\n✅ GPU testing completed successfully');
        
    } catch (error) {
        console.error('\n❌ GPU testing failed:', error.message);
        console.error('Stack trace:', error.stack);
    } finally {
        gpuManager.dispose();
    }
}

async function testTensorOperations(gpuManager) {
    const operations = [
        {
            name: 'Matrix Multiplication (Small)',
            operation: () => performMatrixMultiplication(100, 100)
        },
        {
            name: 'Matrix Multiplication (Medium)',
            operation: () => performMatrixMultiplication(500, 500)
        },
        {
            name: 'Matrix Multiplication (Large)',
            operation: () => performMatrixMultiplication(1000, 1000)
        },
        {
            name: 'Element-wise Operations',
            operation: () => performElementWiseOps(1000, 1000)
        }
    ];
    
    for (const test of operations) {
        console.log(`\nTesting: ${test.name}`);
        
        try {
            const result = await gpuManager.performWithGPUFallback(
                test.operation,
                test.name
            );
            
            console.log(`  ✅ ${test.name}: ${result.duration}ms (${result.backend})`);
            
        } catch (error) {
            console.log(`  ❌ ${test.name}: Failed - ${error.message}`);
        }
    }
}

async function performMatrixMultiplication(rows, cols) {
    const startTime = Date.now();
    
    const a = tf.randomNormal([rows, cols]);
    const b = tf.randomNormal([cols, rows]);
    
    const result = tf.matMul(a, b);
    
    // Force execution by reading a small portion of the data
    const data = await result.slice([0, 0], [1, 1]).data();
    
    const duration = Date.now() - startTime;
    
    // Clean up tensors
    a.dispose();
    b.dispose();
    result.dispose();
    
    return {
        duration: duration,
        backend: tf.getBackend(),
        dataLength: data.length,
        operation: 'matMul'
    };
}

async function performElementWiseOps(rows, cols) {
    const startTime = Date.now();
    
    const a = tf.randomNormal([rows, cols]);
    const b = tf.randomNormal([rows, cols]);
    
    // Chain of element-wise operations
    const result = a.add(b).mul(tf.scalar(2)).relu().tanh();
    
    // Force execution
    const data = await result.slice([0, 0], [1, 1]).data();
    
    const duration = Date.now() - startTime;
    
    // Clean up tensors
    a.dispose();
    b.dispose();
    result.dispose();
    
    return {
        duration: duration,
        backend: tf.getBackend(),
        dataLength: data.length,
        operation: 'elementWise'
    };
}

async function testLSTMTraining(gpuManager) {
    console.log('Creating small LSTM model for performance testing...');
    
    try {
        // Create a small LSTM model for testing
        const modelConfig = {
            sequenceLength: 20,
            features: 10,
            units: 16,
            layers: 1,
            epochs: 5,
            learningRate: 0.01
        };
        
        const model = new LSTMModel(modelConfig);
        
        // Generate small training data
        console.log('Generating synthetic training data...');
        const samples = 100;
        const sequences = [];
        const targets = [];
        
        for (let i = 0; i < samples; i++) {
            const sequence = Array(modelConfig.sequenceLength).fill().map(() => 
                Array(modelConfig.features).fill().map(() => Math.random())
            );
            sequences.push(sequence);
            targets.push(Math.random() > 0.5 ? 1 : 0);
        }
        
        // Convert to tensors
        const trainX = tf.tensor3d(sequences);
        const trainY = tf.tensor1d(targets);
        
        console.log('Training LSTM model with GPU acceleration...');
        
        // Test training with GPU fallback
        const trainingResult = await gpuManager.performWithGPUFallback(async () => {
            model.buildModel();
            model.compileModel();
            
            const startTime = Date.now();
            
            const history = await model.train(trainX, trainY, null, null, {
                epochs: modelConfig.epochs,
                batchSize: 16,
                verbose: 0
            });
            
            const duration = Date.now() - startTime;
            
            return {
                duration: duration,
                backend: tf.getBackend(),
                finalLoss: history.history.loss[history.history.loss.length - 1],
                finalAccuracy: history.history.acc[history.history.acc.length - 1]
            };
        }, 'LSTM model training');
        
        console.log(`  ✅ LSTM Training: ${trainingResult.duration}ms (${trainingResult.backend})`);
        console.log(`      Final Loss: ${trainingResult.finalLoss.toFixed(4)}`);
        console.log(`      Final Accuracy: ${trainingResult.finalAccuracy.toFixed(4)}`);
        
        // Test prediction
        console.log('Testing LSTM prediction...');
        const testInput = tf.tensor3d([sequences[0]]);
        
        const predictionResult = await gpuManager.performWithGPUFallback(async () => {
            const startTime = Date.now();
            
            const prediction = await model.predict(testInput);
            
            const duration = Date.now() - startTime;
            
            return {
                duration: duration,
                backend: tf.getBackend(),
                prediction: prediction[0]
            };
        }, 'LSTM prediction');
        
        console.log(`  ✅ LSTM Prediction: ${predictionResult.duration}ms (${predictionResult.backend})`);
        console.log(`      Prediction: ${predictionResult.prediction.toFixed(4)}`);
        
        // Clean up
        trainX.dispose();
        trainY.dispose();
        testInput.dispose();
        model.dispose();
        
    } catch (error) {
        console.error('  ❌ LSTM testing failed:', error.message);
    }
}

// Memory monitoring function
function logMemoryUsage(stage) {
    const memory = tf.memory();
    console.log(`[${stage}] Memory Usage:`, {
        numTensors: memory.numTensors,
        numBytes: Math.round(memory.numBytes / 1024 / 1024) + 'MB'
    });
}

// Run the test
async function main() {
    console.log('🎮 Starting GPU Support Testing...\n');
    
    logMemoryUsage('Initial');
    
    try {
        await testGPUSupport();
    } catch (error) {
        console.error('Test failed:', error);
        process.exit(1);
    }
    
    logMemoryUsage('Final');
    
    console.log('\n🎉 GPU testing completed!');
    console.log('\nIf GPU is available and faster, it will be used automatically during training.');
    console.log('If GPU fails or is slower, CPU fallback will be used seamlessly.');
}

// Handle process exit
process.on('SIGINT', () => {
    console.log('\n🛑 GPU test interrupted');
    process.exit(0);
});

process.on('uncaughtException', (error) => {
    console.error('\n💥 Uncaught Exception:', error.message);
    console.error(error.stack);
    process.exit(1);
});

// Run the test
if (require.main === module) {
    main();
}

module.exports = { testGPUSupport };