const axios = require('axios');
const config = require('config');
const { Logger } = require('../utils'); // Changed from '../utils/Logger'

class DataClient {
    constructor() {
        this.baseUrl = config.get('core.baseUrl');
        this.endpoints = config.get('core.endpoints');
        this.client = axios.create({
            baseURL: this.baseUrl,
            timeout: 10000,
            headers: {
                'Content-Type': 'application/json'
            }
        });
        
        Logger.info('DataClient initialized', {
            baseUrl: this.baseUrl
        });
    }
    
    async getAllData() {
        try {
            Logger.debug('Fetching all data from core service');
            const response = await this.client.get(this.endpoints.data);
            
            Logger.debug('Successfully fetched all data', {
                pairs: response.data.pairs?.length || 0,
                status: response.data.status
            });
            
            return response.data;
        } catch (error) {
            Logger.error('Failed to fetch all data from core', {
                error: error.message,
                url: this.baseUrl + this.endpoints.data
            });
            throw error;
        }
    }
    
    async getPairData(pair) {
        try {
            Logger.debug(`Fetching data for pair: ${pair}`);
            const response = await this.client.get(`${this.endpoints.pair}/${pair.toUpperCase()}`);
            
            Logger.debug(`Successfully fetched data for ${pair}`, {
                hasHistory: !!response.data.history,
                hasStrategies: !!response.data.strategies,
                dataPoints: response.data.history?.closes?.length || 0
            });
            
            return response.data;
        } catch (error) {
            Logger.error(`Failed to fetch data for pair ${pair}`, {
                error: error.message,
                url: this.baseUrl + this.endpoints.pair + '/' + pair
            });
            throw error;
        }
    }
    
    async checkCoreHealth() {
        try {
            Logger.debug('Checking core service health');
            const response = await this.client.get(this.endpoints.health);
            
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
                url: this.baseUrl + this.endpoints.health
            });
            throw error;
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
                    await new Promise(resolve => setTimeout(resolve, retryDelay));
                }
            }
        }
        
        throw new Error('Core service failed to become ready within timeout');
    }
}

module.exports = DataClient;