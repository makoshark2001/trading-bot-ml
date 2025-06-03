// File: src/data/DataClient.js
// Replace the entire file with this content to fix connection reset issues

const axios = require('axios');
const config = require('config');
const { Logger } = require('../utils');

class DataClient {
    constructor() {
        this.baseUrl = config.get('core.baseUrl');
        this.endpoints = config.get('core.endpoints');
        
        // Create axios instance with better error handling and timeouts
        this.client = axios.create({
            baseURL: this.baseUrl,
            timeout: 30000, // Increased timeout to 30 seconds
            headers: {
                'Content-Type': 'application/json',
                'Connection': 'keep-alive'
            },
            // Retry configuration
            maxRedirects: 3,
            validateStatus: function (status) {
                return status >= 200 && status < 300;
            }
        });
        
        // Add request interceptor for logging
        this.client.interceptors.request.use(
            (config) => {
                Logger.debug('Making request to core service', {
                    url: config.url,
                    method: config.method,
                    baseURL: config.baseURL
                });
                return config;
            },
            (error) => {
                Logger.error('Request interceptor error', { error: error.message });
                return Promise.reject(error);
            }
        );
        
        // Add response interceptor for error handling
        this.client.interceptors.response.use(
            (response) => {
                Logger.debug('Received response from core service', {
                    status: response.status,
                    url: response.config.url,
                    dataSize: JSON.stringify(response.data).length
                });
                return response;
            },
            (error) => {
                Logger.error('Response interceptor error', {
                    message: error.message,
                    code: error.code,
                    status: error.response?.status,
                    url: error.config?.url
                });
                return Promise.reject(error);
            }
        );
        
        Logger.info('DataClient initialized with enhanced error handling', {
            baseUrl: this.baseUrl,
            timeout: 30000
        });
    }
    
    async getAllData() {
        try {
            Logger.debug('Fetching all data from core service');
            
            const response = await this.retryRequest(() => 
                this.client.get(this.endpoints.data)
            );
            
            Logger.debug('Successfully fetched all data', {
                pairs: response.data.pairs?.length || 0,
                status: response.data.status
            });
            
            return response.data;
        } catch (error) {
            Logger.error('Failed to fetch all data from core', {
                error: error.message,
                code: error.code,
                url: this.baseUrl + this.endpoints.data
            });
            throw new Error(`Core service unavailable: ${error.message}`);
        }
    }
    
    async getPairData(pair) {
        try {
            Logger.debug(`Fetching data for pair: ${pair}`);
            
            const response = await this.retryRequest(() => 
                this.client.get(`${this.endpoints.pair}/${pair.toUpperCase()}`)
            );
            
            Logger.debug(`Successfully fetched data for ${pair}`, {
                hasHistory: !!response.data.history,
                hasStrategies: !!response.data.strategies,
                dataPoints: response.data.history?.closes?.length || 0
            });
            
            // Validate response data
            if (!response.data.history || !response.data.history.closes) {
                throw new Error(`Invalid data structure for pair ${pair}`);
            }
            
            if (response.data.history.closes.length < 60) {
                throw new Error(`Insufficient data for pair ${pair}: ${response.data.history.closes.length} points`);
            }
            
            return response.data;
        } catch (error) {
            Logger.error(`Failed to fetch data for pair ${pair}`, {
                error: error.message,
                code: error.code,
                url: this.baseUrl + this.endpoints.pair + '/' + pair
            });
            throw new Error(`Failed to get data for ${pair}: ${error.message}`);
        }
    }
    
    async checkCoreHealth() {
        try {
            Logger.debug('Checking core service health');
            
            const response = await this.retryRequest(() => 
                this.client.get(this.endpoints.health)
            );
            
            const isHealthy = response.data.status === 'healthy';
            Logger.info('Core service health check', {
                status: response.data.status,
                healthy: isHealthy,
                dataCollection: response.data.dataCollection
            });
            
            return response.data;
        } catch (error) {
            Logger.error('Core service health check failed', {
                error: error.message,
                code: error.code,
                url: this.baseUrl + this.endpoints.health
            });
            throw new Error(`Core service health check failed: ${error.message}`);
        }
    }
    
    async waitForCoreService(maxRetries = 10, retryDelay = 5000) {
        for (let i = 0; i < maxRetries; i++) {
            try {
                await this.checkCoreHealth();
                Logger.info('Core service is ready');
                return true;
            } catch (error) {
                Logger.warn(`Core service not ready, attempt ${i + 1}/${maxRetries}`, {
                    error: error.message,
                    retryIn: retryDelay / 1000 + 's'
                });
                
                if (i < maxRetries - 1) {
                    await this.sleep(retryDelay);
                }
            }
        }
        
        throw new Error('Core service failed to become ready within timeout');
    }
    
    // Retry mechanism for failed requests
    async retryRequest(requestFunction, maxRetries = 3, baseDelay = 1000) {
        let lastError;
        
        for (let attempt = 1; attempt <= maxRetries; attempt++) {
            try {
                const result = await requestFunction();
                
                // If we get here, the request succeeded
                if (attempt > 1) {
                    Logger.info(`Request succeeded on attempt ${attempt}`);
                }
                
                return result;
                
            } catch (error) {
                lastError = error;
                
                Logger.warn(`Request attempt ${attempt}/${maxRetries} failed`, {
                    error: error.message,
                    code: error.code,
                    status: error.response?.status
                });
                
                // Don't retry on certain error types
                if (this.shouldNotRetry(error)) {
                    Logger.info('Not retrying due to error type', { 
                        code: error.code,
                        status: error.response?.status 
                    });
                    throw error;
                }
                
                // Don't wait after the last attempt
                if (attempt < maxRetries) {
                    const delay = baseDelay * Math.pow(2, attempt - 1); // Exponential backoff
                    Logger.info(`Retrying in ${delay}ms...`);
                    await this.sleep(delay);
                }
            }
        }
        
        // All retries failed
        Logger.error(`All ${maxRetries} attempts failed`, { 
            lastError: lastError.message 
        });
        throw lastError;
    }
    
    // Determine if we should retry based on error type
    shouldNotRetry(error) {
        // Don't retry on client errors (4xx) except for specific cases
        if (error.response?.status >= 400 && error.response?.status < 500) {
            // Retry on 408 (timeout), 429 (rate limit), but not on 404, 401, etc.
            return ![408, 429].includes(error.response.status);
        }
        
        // Don't retry on invalid JSON responses
        if (error.message.includes('JSON')) {
            return true;
        }
        
        // Retry on network errors, server errors, timeouts
        return false;
    }
    
    // Helper function for delays
    sleep(ms) {
        return new Promise(resolve => setTimeout(resolve, ms));
    }
    
    // Test connection to core service
    async testConnection() {
        try {
            Logger.info('Testing connection to core service...');
            
            const startTime = Date.now();
            const health = await this.checkCoreHealth();
            const responseTime = Date.now() - startTime;
            
            Logger.info('Connection test successful', {
                responseTime: `${responseTime}ms`,
                coreStatus: health.status,
                dataPoints: health.dataCollection?.totalDataPoints || 0
            });
            
            return {
                success: true,
                responseTime,
                status: health.status,
                timestamp: Date.now()
            };
            
        } catch (error) {
            Logger.error('Connection test failed', { error: error.message });
            return {
                success: false,
                error: error.message,
                timestamp: Date.now()
            };
        }
    }
    
    // Get available pairs from core service
    async getAvailablePairs() {
        try {
            Logger.debug('Fetching available pairs from core service');
            
            const response = await this.retryRequest(() => 
                this.client.get('/api/pairs')
            );
            
            Logger.debug('Successfully fetched available pairs', {
                count: response.data.pairs?.length || 0
            });
            
            return response.data;
        } catch (error) {
            Logger.error('Failed to fetch available pairs', {
                error: error.message,
                code: error.code
            });
            throw new Error(`Failed to get available pairs: ${error.message}`);
        }
    }
    
    // Get specific indicator data for a pair
    async getIndicatorData(pair, indicator) {
        try {
            Logger.debug(`Fetching ${indicator} data for pair: ${pair}`);
            
            const response = await this.retryRequest(() => 
                this.client.get(`${this.endpoints.pair}/${pair.toUpperCase()}/indicator/${indicator}`)
            );
            
            Logger.debug(`Successfully fetched ${indicator} data for ${pair}`);
            
            return response.data;
        } catch (error) {
            Logger.error(`Failed to fetch ${indicator} data for pair ${pair}`, {
                error: error.message,
                code: error.code
            });
            throw new Error(`Failed to get ${indicator} data for ${pair}: ${error.message}`);
        }
    }
    
    // Get core service configuration
    async getCoreConfig() {
        try {
            Logger.debug('Fetching core service configuration');
            
            const response = await this.retryRequest(() => 
                this.client.get('/api/config')
            );
            
            Logger.debug('Successfully fetched core configuration');
            
            return response.data;
        } catch (error) {
            Logger.error('Failed to fetch core configuration', {
                error: error.message,
                code: error.code
            });
            throw new Error(`Failed to get core configuration: ${error.message}`);
        }
    }
    
    // Check if pair data is ready for ML processing
    async isPairReadyForML(pair) {
        try {
            const pairData = await this.getPairData(pair);
            
            const isReady = 
                pairData.history &&
                pairData.history.closes &&
                pairData.history.closes.length >= 60 &&
                pairData.strategies &&
                Object.keys(pairData.strategies).length > 0;
            
            Logger.debug(`ML readiness check for ${pair}`, {
                isReady,
                dataPoints: pairData.history?.closes?.length || 0,
                strategies: Object.keys(pairData.strategies || {}).length
            });
            
            return {
                ready: isReady,
                dataPoints: pairData.history?.closes?.length || 0,
                strategies: Object.keys(pairData.strategies || {}).length,
                pair: pair
            };
            
        } catch (error) {
            Logger.error(`ML readiness check failed for ${pair}`, { error: error.message });
            return {
                ready: false,
                error: error.message,
                pair: pair
            };
        }
    }
    
    // Get connection status and diagnostics
    getConnectionStatus() {
        return {
            baseUrl: this.baseUrl,
            endpoints: this.endpoints,
            timeout: 30000,
            timestamp: Date.now()
        };
    }
}

module.exports = DataClient;