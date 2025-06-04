const tf = require('@tensorflow/tfjs');
require('@tensorflow/tfjs-node'); // Load CPU-optimized backend
const { GPUManager, Logger } = require('../src/utils');
const LSTMModel = require('../src/models/LSTMModel');

async function testCPUOptimizedPerformance() {
    console.log('🚀 CPU-Optimized Performance Test');
    console.log('=' * 50);
    
    try {
        // Ensure we're using the optimized Node.js backend
        await tf.setBackend('tensorflow');
        await tf.ready();
        
        console.log('\n📊 Backend Information:');
        console.log('-' * 25);
        console.log('Current Backend:', tf.getBackend());
        console.log('TensorFlow.js Version:', tf.version.tfjs);
        console.log('Node.js Version:', process.version);
        console.log('Platform:', process.platform);
        console.log('Architecture:', process.arch);
        
        // Test 1: Basic Tensor Operations
        console.log('\n🧮 Test 1: Basic Tensor Operations');
        console.log('-' * 40);
        await testTensorOperations();
        
        // Test 2: LSTM Training Performance
        console.log('\n🧠 Test 2: LSTM Training Performance');
        console.log('-' * 42);
        await testLSTMPerformance();
        
        // Test 3: Prediction Speed
        console.log('\n⚡ Test 3: Prediction Speed');
        console.log('-' * 30);
        await testPredictionSpeed();
        
        // Test 4: Memory Usage
        console.log('\n💾 Test 4: Memory Usage');
        console.log('-' * 25);
        testMemoryUsage();
        
        console.log('\n✅ Performance testing completed successfully!');
        console.log('\n🎯 Summary: CPU-optimized TensorFlow.js should show significant');
        console.log('   performance improvement over browser-based CPU backend.');
        
    } catch (error) {
        console.error('\n❌ Performance testing failed:', error.message);
        console.error('Stack trace:', error.stack);
    }
}

async function testTensorOperations() {
    const tests = [
        { name: 'Small Matrix Mult (100x100)', size: 100 },
        { name: 'Medium Matrix Mult (500x500)', size: 500 },
        { name: 'Large Matrix Mult (1000x1000)', size: 1000 },
        { name: 'Very Large Matrix Mult (2000x2000)', size: 2000 }
    ];
    
    for (const test of tests) {
        const startTime = Date.now();
        
        try {
            // Create random matrices
            const a = tf.randomNormal([test.size, test.size]);
            const b = tf.randomNormal([test.size, test.size]);
            
            // Perform matrix multiplication
            const result = tf.matMul(a, b);
            
            // Force execution by reading a small portion
            const sample = await result.slice([0, 0], [1, 1]).data();
            
            const duration = Date.now() - startTime;
            
            // Clean up
            a.dispose();
            b.dispose();
            result.dispose();
            
            console.log(`  ✅ ${test.name}: ${duration}ms`);
            
        } catch (error) {
            console.log(`  ❌ ${test.name}: Failed - ${error.message}`);
        }
    }
}

async function testLSTMPerformance() {
    const configs = [
        {
            name: 'Tiny LSTM (10 features, 5 timesteps)',
            sequenceLength: 5,
            features: 10,
            units: 8,
            layers: 1,
            samples: 50,
            epochs: 3
        },
        {
            name: 'Small LSTM (20 features, 10 timesteps)',
            sequenceLength: 10,
            features: 20,
            units: 16,
            layers: 1,
            samples: 100,
            epochs: 5
        },
        {
            name: 'Medium LSTM (50 features, 30 timesteps)',
            sequenceLength: 30,
            features: 50,
            units: 32,
            layers: 2,
            samples: 200,
            epochs: 10
        },
        {
            name: 'Large LSTM (84 features, 60 timesteps)',
            sequenceLength: 60,
            features: 84,
            units: 50,
            layers: 2,
            samples: 300,
            epochs: 15
        }
    ];
    
    for (const config of configs) {
        console.log(`\nTesting: ${config.name}`);
        
        try {
            const model = new LSTMModel({
                sequenceLength: config.sequenceLength,
                features: config.features,
                units: config.units,
                layers: config.layers,
                learningRate: 0.01
            });
            
            // Generate synthetic data
            const sequences = [];
            const targets = [];
            
            for (let i = 0; i < config.samples; i++) {
                const sequence = Array(config.sequenceLength).fill().map(() => 
                    Array(config.features).fill().map(() => Math.random())
                );
                sequences.push(sequence);
                targets.push(Math.random() > 0.5 ? 1 : 0);
            }
            
            const trainX = tf.tensor3d(sequences);
            const trainY = tf.tensor1d(targets);
            
            // Build and compile model
            model.buildModel();
            model.compileModel();
            
            // Train with timing
            const startTime = Date.now();
            
            const history = await model.train(trainX, trainY, null, null, {
                epochs: config.epochs,
                batchSize: Math.min(32, Math.floor(config.samples / 4)),
                verbose: 0
            });
            
            const trainingTime = Date.now() - startTime;
            
            // Test prediction speed
            const predStartTime = Date.now();
            const testInput = tf.tensor3d([sequences[0]]);
            await model.predict(testInput);
            const predictionTime = Date.now() - predStartTime;
            
            // Clean up
            trainX.dispose();
            trainY.dispose();
            testInput.dispose();
            model.dispose();
            
            const finalLoss = history.history.loss[history.history.loss.length - 1];
            const finalAcc = history.history.acc[history.history.acc.length - 1];
            
            console.log(`  ✅ Training: ${trainingTime}ms`);
            console.log(`     • Epochs: ${config.epochs}`);
            console.log(`     • Final Loss: ${finalLoss.toFixed(4)}`);
            console.log(`     • Final Accuracy: ${finalAcc.toFixed(4)}`);
            console.log(`     • Prediction: ${predictionTime}ms`);
            console.log(`     • Params: ${model.model?.countParams?.() || 'Unknown'}`);
            
        } catch (error) {
            console.log(`  ❌ ${config.name}: Failed - ${error.message}`);
        }
    }
}

async function testPredictionSpeed() {
    try {
        // Create a medium-sized LSTM for prediction testing
        const model = new LSTMModel({
            sequenceLength: 30,
            features: 50,
            units: 32,
            layers: 1
        });
        
        model.buildModel();
        model.compileModel();
        
        // Generate test data
        const sequence = Array(30).fill().map(() => 
            Array(50).fill().map(() => Math.random())
        );
        
        const batchSizes = [1, 5, 10, 20, 50];
        
        for (const batchSize of batchSizes) {
            const batch = Array(batchSize).fill(sequence);
            const inputTensor = tf.tensor3d(batch);
            
            // Warm up
            await model.predict(inputTensor);
            
            // Time multiple predictions
            const iterations = 10;
            const startTime = Date.now();
            
            for (let i = 0; i < iterations; i++) {
                await model.predict(inputTensor);
            }
            
            const totalTime = Date.now() - startTime;
            const avgTime = totalTime / iterations;
            const throughput = (batchSize * iterations * 1000) / totalTime;
            
            console.log(`  Batch Size ${batchSize}: ${avgTime.toFixed(1)}ms avg, ${throughput.toFixed(1)} pred/sec`);
            
            inputTensor.dispose();
        }
        
        model.dispose();
        
    } catch (error) {
        console.log(`  ❌ Prediction speed test failed: ${error.message}`);
    }
}

function testMemoryUsage() {
    const memory = tf.memory();
    const processMemory = process.memoryUsage();
    
    console.log('TensorFlow.js Memory:');
    console.log(`  • Tensors: ${memory.numTensors}`);
    console.log(`  • Data Buffers: ${memory.numDataBuffers || 0}`);
    console.log(`  • Bytes: ${Math.round(memory.numBytes / 1024 / 1024)} MB`);
    console.log(`  • Unreliable: ${memory.unreliable || false}`);
    
    console.log('\nProcess Memory:');
    console.log(`  • Heap Used: ${Math.round(processMemory.heapUsed / 1024 / 1024)} MB`);
    console.log(`  • Heap Total: ${Math.round(processMemory.heapTotal / 1024 / 1024)} MB`);
    console.log(`  • RSS: ${Math.round(processMemory.rss / 1024 / 1024)} MB`);
    console.log(`  • External: ${Math.round(processMemory.external / 1024 / 1024)} MB`);
}

function logMemoryUsage(stage) {
    const memory = tf.memory();
    const processMemory = process.memoryUsage();
    console.log(`[${stage}] Memory - TF: ${memory.numTensors} tensors, ${Math.round(memory.numBytes / 1024 / 1024)}MB | Process: ${Math.round(processMemory.heapUsed / 1024 / 1024)}MB`);
}

// Comparison with previous performance
function showPerformanceComparison() {
    console.log('\n📈 Expected Performance Improvement:');
    console.log('-' * 40);
    console.log('Previous (Browser CPU Backend):');
    console.log('  • Small LSTM Training: ~2900ms');
    console.log('  • Matrix Mult (1000x1000): ~4800ms');
    console.log('  • Prediction: ~13ms');
    
    console.log('\nCPU-Optimized (tfjs-node) Expected:');
    console.log('  • Small LSTM Training: ~800-1500ms (2-3x faster)');
    console.log('  • Matrix Mult (1000x1000): ~1200-2400ms (2-4x faster)');
    console.log('  • Prediction: ~3-8ms (1.5-4x faster)');
    
    console.log('\nNote: Actual performance depends on your specific CPU');
    console.log('and system configuration. Results may vary.');
}

async function main() {
    console.log('🚀 Starting CPU-Optimized Performance Testing...\n');
    
    logMemoryUsage('Initial');
    
    try {
        await testCPUOptimizedPerformance();
        showPerformanceComparison();
    } catch (error) {
        console.error('Performance testing failed:', error);
        process.exit(1);
    }
    
    logMemoryUsage('Final');
    
    console.log('\n🎉 CPU-optimized performance testing completed!');
    console.log('\nYour ML service should now run significantly faster with tfjs-node.');
}

// Handle process exit
process.on('SIGINT', () => {
    console.log('\n🛑 Performance test interrupted');
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

module.exports = { testCPUOptimizedPerformance };