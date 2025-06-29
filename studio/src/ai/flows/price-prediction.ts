
// src/ai/flows/price-prediction.ts
'use server';

/**
 * @fileOverview This file defines a Genkit flow for predicting house prices in London.
 *
 * - predictPrice - A function that takes property details as input and returns a price prediction.
 * - PredictionInput - The input type for the predictPrice function.
 * - PredictionOutput - The return type for the predictPrice function.
 */

import {ai} from '@/ai/genkit';
import {z} from 'genkit';
// Import general types if needed by Zod schema, e.g. for enums
import type { PropertyType, EnergyRating, Tenure } from '@/types';
import { PROPERTY_TYPE_OPTIONS, ENERGY_RATING_OPTIONS, TENURE_OPTIONS, REGION_OPTIONS, PREDICTION_MONTH_OF_SALE_FORMAT_DESC } from '@/lib/constants';


// Zod schema (NOT EXPORTED, but used internally by the flow)
const PredictionInputSchema = z.object({
  fullAddress: z.string().min(5, { message: 'Full address is required.' }).describe('The full address of the property.'),
  outcode: z.enum(REGION_OPTIONS as [string, ...string[]]).describe('The outcode part of the postcode (e.g., SW1, E1).'),
  longitude: z.number().optional().describe('The longitude of the property (derived from address).'),
  latitude: z.number().optional().describe('The latitude of the property (derived from address).'),
  bedrooms: z.number().int().min(0, {message: 'Bedrooms cannot be negative.'}).max(10, { message: 'Cannot exceed 10 bedrooms.' }).describe('The number of bedrooms.'),
  bathrooms: z.number().int().min(0, {message: 'Bathrooms cannot be negative.'}).max(10, { message: 'Cannot exceed 10 bathrooms.' }).describe('The number of bathrooms.'),
  livingRooms: z.number().int().min(0, {message: 'Living rooms cannot be negative.'}).max(10, { message: 'Cannot exceed 10 living rooms.' }).describe('The number of living rooms.'),
  floorAreaSqM: z.number().positive({ message: 'Floor area must be a positive number.' }).describe('The floor area of the property in square meters.'),
  tenure: z.enum(TENURE_OPTIONS as [Tenure, ...Tenure[]]).describe('The tenure type of the property (e.g., Freehold, Leasehold).'),
  propertyType: z.enum(PROPERTY_TYPE_OPTIONS as [PropertyType, ...PropertyType[]]).describe('The type of property.'),
  currentEnergyRating: z.enum(ENERGY_RATING_OPTIONS as [EnergyRating, ...EnergyRating[]]).describe('The current energy efficiency rating of the property.'),
  sale_month: z.number().int().min(1).max(12).describe('The month of sale (1-12).'),
  sale_year: z.number().int().min(1990).max(new Date().getFullYear() + 10).describe('The year of sale (YYYY).'),
});
// Type inferred from Zod schema, exported for use in components/services
export type PredictionInput = z.infer<typeof PredictionInputSchema>;

// Zod schema (NOT EXPORTED, but used internally by the flow)
const PredictionOutputSchema = z.object({
  price: z.number().describe('The predicted price of the property.'),
  isStable: z.boolean().describe('Whether the predicted price trend for the next 12 months is stable.'),
  priceHistoryChartData: z.array(z.object({
    month: z.string(), // e.g., "Jan 2025"
    price: z.number(),
  })).describe('The predicted price for the next 12 months.'),
});
// Type inferred from Zod schema, exported for use in components/services
export type PredictionOutput = z.infer<typeof PredictionOutputSchema>;

export async function predictPrice(input: PredictionInput): Promise<PredictionOutput> {
  return predictPriceFlow(input);
}

const prompt = ai.definePrompt({
  name: 'pricePredictionPrompt',
  input: {schema: PredictionInputSchema},
  output: {schema: PredictionOutputSchema},
  prompt: `You are an AI real estate expert specializing in London property prices.

You will receive property details and must predict its price for the specified month and year of sale. You also need to determine if the price trend for the next 12 months is stable.

Property Details:
Full Address: {{{fullAddress}}}
Outcode: {{{outcode}}}
Longitude: {{{longitude}}}
Latitude: {{{latitude}}}
Property Type: {{{propertyType}}}
Bedrooms: {{{bedrooms}}}
Bathrooms: {{{bathrooms}}}
Living Rooms: {{{livingRooms}}}
Floor Area (sqm): {{{floorAreaSqM}}}
Tenure: {{{tenure}}}
Current Energy Rating: {{{currentEnergyRating}}}
Sale Month: {{{sale_month}}}
Sale Year: {{{sale_year}}}

Consider factors such as historical price data (1991-2023), property type, location, size, condition (implied by energy rating), tenure and current market trends for the specified sale period.

Output the predicted price, whether the trend is stable (isStable: true/false), and the predicted price for the next 12 months.

Example for priceHistoryChartData:
[
  { "month": "Jan 2025", "price": 520000 },
  { "month": "Feb 2025", "price": 525000 },
  ...
]
Ensure priceHistoryChartData covers 12 months from the sale_month and sale_year. Prices must be realistic for London.`,
});

const predictPriceFlow = ai.defineFlow(
  {
    name: 'predictPriceFlow',
    inputSchema: PredictionInputSchema,
    outputSchema: PredictionOutputSchema,
  },
  async (input: PredictionInput): Promise<PredictionOutput> => {
    let basePrice = 200000;
    basePrice += input.bedrooms * 70000;
    basePrice += input.bathrooms * 40000;
    basePrice += input.livingRooms * 35000; // Adjusted for livingRooms
    basePrice += input.floorAreaSqM * 1600; // Adjusted for floorAreaSqM

    if (input.tenure === 'Freehold') basePrice *= 1.1;

    const energyRatingModifiers: Record<EnergyRating, number> = { 'A': 1.1, 'B': 1.05, 'C': 1.0, 'D': 0.95, 'E': 0.9, 'F': 0.85, 'G': 0.8 };
    basePrice *= energyRatingModifiers[input.currentEnergyRating];

    const highValueOutcodes = ['SW1A', 'SW1E', 'SW1H', 'SW1P', 'SW1V', 'SW1W', 'SW1X', 'SW1Y', 'W1K', 'W1J', 'WC2', 'EC2'];
    const midValueOutcodes = ['N1', 'E1', 'SE1', 'EC1', 'NW1', 'W1G', 'W1U', 'W1B', 'W1C', 'W1D', 'W1F', 'W1T', 'W1W', 'W2', 'W8', 'W9', 'W11', 'WC1'];
    
    if (highValueOutcodes.some(oc => input.outcode.startsWith(oc))) {
      basePrice *= 1.35; // Increased modifier for very high value
    } else if (midValueOutcodes.some(oc => input.outcode.startsWith(oc))) {
      basePrice *= 1.15;
    } else { 
        basePrice *= 1.0;
    }

    // Simulate some effect from sale_year and sale_month
    const yearFactor = 1 + (input.sale_year - 2023) * 0.02; // 2% increase per year from 2023
    const monthFactor = 1 + (input.sale_month - 6) * 0.002; // Slight seasonal adjustment around mid-year
    basePrice *= yearFactor * monthFactor;

    const predictedSalePrice = Math.round(basePrice / 1000) * 1000;

    const priceHistoryChartData = [];
    let lastPrice = predictedSalePrice * 0.98; 
    const monthNames = ["Jan", "Feb", "Mar", "Apr", "May", "Jun", "Jul", "Aug", "Sep", "Oct", "Nov", "Dec"];

    for (let i = 0; i < 12; i++) {
      // Start from the input sale_month and sale_year
      const date = new Date(input.sale_year, input.sale_month - 1 + i, 1);
      const month = monthNames[date.getMonth()];
      const year = date.getFullYear();
      lastPrice = lastPrice * (1 + (Math.random() * 0.015 - 0.005)); // Random fluctuation
      priceHistoryChartData.push({
        month: `${month} ${year}`,
        price: Math.round(lastPrice / 1000) * 1000,
      });
    }
    
    // Ensure the first month in chart data is very close to the predicted sale price
     if (priceHistoryChartData.length > 0) {
        const initialChartPrice = priceHistoryChartData[0].price;
        const adjustmentFactor = predictedSalePrice / initialChartPrice;
        priceHistoryChartData.forEach(item => {
            item.price = Math.round((item.price * adjustmentFactor) / 1000) * 1000;
        });
        // Ensure the first element's price is very close to the main predicted price
        priceHistoryChartData[0].price = Math.round((predictedSalePrice * (0.995 + Math.random() * 0.01)) /1000) * 1000;
    }


    const trendOptions: Array<'increasing' | 'decreasing' | 'stable'> = ['increasing', 'decreasing', 'stable'];
    const randomTrend = trendOptions[Math.floor(Math.random() * trendOptions.length)];

    const fakeOutput: PredictionOutput = {
      price: predictedSalePrice,
      isStable: randomTrend === 'stable',
      priceHistoryChartData: priceHistoryChartData,
    };

    return fakeOutput;
  }
);

