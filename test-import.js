// Quick test to debug MLServer import issue
console.log('🔍 Testing MLServer import...');

try {
    console.log('Step 1: Testing require...');
    const MLServer = require('./src/api/MLServer');
    console.log('✅ MLServer required successfully');
    console.log('Type of MLServer:', typeof MLServer);
    console.log('MLServer constructor:', typeof MLServer.prototype?.constructor);
    
    console.log('Step 2: Testing instantiation...');
    const server = new MLServer();
    console.log('✅ MLServer instantiated successfully');
    
    console.log('Step 3: Checking methods...');
    console.log('Available methods:', Object.getOwnPropertyNames(Object.getPrototypeOf(server)));
    console.log('Has start method:', typeof server.start);
    console.log('start method source:', server.start?.toString().substring(0, 100) + '...');
    
    if (typeof server.start === 'function') {
        console.log('✅ start method exists and is a function');
    } else {
        console.log('❌ start method is not a function');
        console.log('server.start value:', server.start);
    }
    
} catch (error) {
    console.error('❌ Error during test:', error.message);
    console.error('Stack:', error.stack);
}