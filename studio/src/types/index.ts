
// Basic Property Types
export type PropertyType = 
  | 'Semi-Detached House' | 'Terrace Property' | 'Purpose Built Flat'
  | 'Mid Terrace House' | 'Converted Flat' | 'End Terrace House'
  | 'Flat/Maisonette' | 'Detached House' | 'Terraced'
  | 'Bungalow Property' | 'Mid Terrace Property'
  | 'Detached Bungalow' | 'Semi-Detached Bungalow'
  | 'Mid Terrace Bungalow' | 'Semi-Detached Property'
  | 'Detached Property' | 'End Terrace Property' | 'Terraced Bungalow'
  | 'End Terrace Bungalow' | 'Flat' | 'Detached' | 'Semi-detached' | 'Bungalow' | 'Maisonette';

export type EnergyRating = 'A' | 'B' | 'C' | 'D' | 'E' | 'F' | 'G';
export type Tenure = 'Freehold' | 'Leasehold' | 'Feudal' | 'Shared';

// Main Data Structures
export interface Property {
  id: string;
  name: string;
  fullAddress: string;
  price: number;
  propertyType: PropertyType;
  bedrooms: number;
  bathrooms: number;
  livingRooms: number;
  floorAreaSqM?: number;
  currentEnergyRating: EnergyRating;
  tenure: Tenure;
  outcode: string;
  image: string;
  description: string;
  dataAiHint?: string;
  longitude?: number;
  latitude?: number;
  sale_month: number; 
  sale_year: number; 
  uploaderName: string;
  uploaderEmail: string;
  uploaderPhone?: string;
}

export interface OutcodeData {
  id: string; // This is the outcode
  name: string;
  avgPrice: number;
  priceCategory: 'low' | 'medium' | 'high';
  description: string;
}

export interface QuarterlyPricePoint {
  quarter: string; // e.g., "Q1 2023"
  price: number;
}

export interface RegionMarketData {
  regionId: string; // This is the outcode
  regionName: string;
  quarterlyPriceHistory: QuarterlyPricePoint[];
  currentAveragePrice: number;
  priceRank: string; // e.g., "Rank: 5 of 50"
}

// Note: AI Flow specific Input/Output types (PredictionInput, PredictionOutput, etc.)
// will continue to be defined and exported from their respective flow files (e.g., src/ai/flows/price-prediction.ts)
// as they are directly inferred from Zod schemas. Components should import them from there.
