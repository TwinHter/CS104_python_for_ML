
'use server';

import * as fs from 'node:fs';
import * as path from 'node:path';
import type { Property, RegionMarketData, QuarterlyPricePoint, PropertyType, Tenure, EnergyRating } from '@/types';
import initialPropertiesDataFromFile from '@/lib/data/properties.json';
import { londonOutcodes } from '@/lib/data/london_outcodes_data';

// Path to the properties.json file
const propertiesFilePath = path.join(process.cwd(), 'src', 'lib', 'data', 'properties.json');

// Initialize liveProperties from the file. This copy will be mutated for in-session updates.
let liveProperties: Property[] = [...initialPropertiesDataFromFile] as Property[];

interface CsvRow {
  ID?: string;
  fullAddress?: string;
  postcode?: string;
  country?: string;
  outcode?: string;
  latitude?: number;
  longitude?: number;
  bathrooms?: number;
  bedrooms?: number;
  floorAreaSqM?: number;
  livingRooms?: number;
  tenure?: Tenure;
  propertyType?: PropertyType;
  currentEnergyRating?: EnergyRating;
  sale_month?: number;
  sale_year?: number;
  price?: number;
  street?: string;
  incode?: string;
  livingRooms1?: number; // This was in your new CSV, keeping for robust parsing if needed elsewhere
  area_bin?: string;    // This was in your new CSV, keeping for robust parsing if needed elsewhere
}

export const getPropertyDetails = async (propertyId: string): Promise<Property | null> => {
  const property = liveProperties.find(p => p.id === propertyId);
  return property || null;
};


export const getProperties = async (): Promise<Property[]> => {
  return [...liveProperties];
};

export const addProperty = async (
  propertyData: Omit<Property, 'id'> & {
    image: string; // data URL
    longitude?: number;
    latitude?: number;
    sale_month: number;
    sale_year: number;
    uploaderName: string;
    uploaderEmail: string;
    uploaderPhone?: string;
  }
): Promise<Property> => {
  const newProperty: Property = {
    ...propertyData,
    id: Date.now().toString(),
  };

  try {
    const fileContent = fs.readFileSync(propertiesFilePath, 'utf-8');
    const currentPropertiesFromFile: Property[] = JSON.parse(fileContent);
    currentPropertiesFromFile.unshift(newProperty);
    fs.writeFileSync(propertiesFilePath, JSON.stringify(currentPropertiesFromFile, null, 2), 'utf-8');
    liveProperties = [...currentPropertiesFromFile];
  } catch (error) {
    console.error("Failed to write property to JSON file:", error);
    liveProperties.unshift(newProperty);
  }
  return newProperty;
};

function getQuarter(month: number): number {
  if (month <= 3) return 1;
  if (month <= 6) return 2;
  if (month <= 9) return 3;
  return 4;
}

export const getRegionMarketData = async (regionId: string): Promise<RegionMarketData> => {
  const regionDetails = londonOutcodes.find(r => r.id === regionId);
  if (!regionDetails) {
    throw new Error(`Region ${regionId} not found.`);
  }

  let parsedCsvData: CsvRow[] = [];
  try {
    const csvFilePath = path.join(process.cwd(), 'public', 'data.csv');
    const fileContent = fs.readFileSync(csvFilePath, 'utf-8');
    const lines = fileContent.trim().split(/\r?\n/);

    const headerLine = lines[0];
    const headers = headerLine.split(',').map(h => h.trim().toLowerCase()); // Normalize headers to lowercase
    
    const headerMap = headers.reduce((acc, h, i) => {
        acc[h] = i;
        return acc;
    }, {} as Record<string, number>);


    parsedCsvData = lines.slice(1).map((line, lineIndex) => {
      const values = line.split(',');
      
      const getVal = (colName: string, isNumeric = false, isFloat = false): string | number | undefined => {
        const lowerColName = colName.toLowerCase();
        const valIndex = headerMap[lowerColName];
        if (valIndex === undefined || valIndex >= values.length) return undefined;
        
        const rawVal = values[valIndex];
        if (rawVal === undefined || rawVal.trim() === '' || rawVal.trim().toLowerCase() === 'null') {
          return undefined;
        }
        
        if (isNumeric) {
          const numVal = isFloat ? parseFloat(rawVal) : parseInt(rawVal, 10);
          return isNaN(numVal) ? undefined : numVal;
        }
        return rawVal.trim();
      };

      // This CsvRow mapping needs to align with the "older" format of your CSV.
      // I am assuming the "older" format had at least these fields relevant for map statistics:
      // outcode, sale_year, sale_month, price, latitude, longitude
      // And possibly these for property details: bedrooms, bathrooms, propertyType, tenure
      const rowData: CsvRow = {
        ID: getVal('ID') as string | undefined,
        fullAddress: getVal('fullAddress') as string | undefined,
        outcode: getVal('outcode') as string | undefined,
        latitude: getVal('latitude', true, true) as number | undefined,
        longitude: getVal('longitude', true, true) as number | undefined,
        bedrooms: getVal('bedrooms', true) as number | undefined,
        bathrooms: getVal('bathrooms', true) as number | undefined,
        propertyType: getVal('propertyType') as PropertyType | undefined,
        tenure: getVal('tenure') as Tenure | undefined,
        sale_month: getVal('sale_month', true) as number | undefined,
        sale_year: getVal('sale_year', true) as number | undefined,
        price: getVal('price', true, true) as number | undefined,
      };
      return rowData;
    }).filter(row => 
        row.outcode && 
        typeof row.price === 'number' && row.price > 0 &&
        typeof row.sale_year === 'number' &&
        typeof row.sale_month === 'number'
    );
  } catch (error) {
    console.error("Failed to read or parse data.csv:", error);
  }

  const regionSales = parsedCsvData.filter(sale => sale.outcode === regionId);

  const quarterlyPrices: Record<string, { total: number, count: number }> = {};
  const currentFullYear = new Date().getFullYear();
  const fiveYearsAgo = currentFullYear - 5;

  regionSales.forEach(sale => {
    if (sale.sale_year && sale.sale_month && sale.price && sale.sale_year >= fiveYearsAgo && sale.sale_year <= currentFullYear) {
      const quarter = getQuarter(sale.sale_month);
      const key = `Q${quarter} ${sale.sale_year}`;
      if (!quarterlyPrices[key]) {
        quarterlyPrices[key] = { total: 0, count: 0 };
      }
      quarterlyPrices[key].total += sale.price;
      quarterlyPrices[key].count += 1;
    }
  });

  const quarterlyPriceHistory: QuarterlyPricePoint[] = [];
  for (let year = fiveYearsAgo; year <= currentFullYear; year++) {
    for (let q = 1; q <= 4; q++) {
      if (year === currentFullYear && q > getQuarter(new Date().getMonth() + 1)) {
        break; 
      }
      const key = `Q${q} ${year}`;
      if (quarterlyPrices[key] && quarterlyPrices[key].count > 0) {
        quarterlyPriceHistory.push({
          quarter: key,
          price: Math.round(quarterlyPrices[key].total / quarterlyPrices[key].count / 1000) * 1000,
        });
      } else {
         if (year < currentFullYear || (year === currentFullYear && q < getQuarter(new Date().getMonth() +1))) {
            const previousQuarterData = quarterlyPriceHistory.length > 0 ? quarterlyPriceHistory[quarterlyPriceHistory.length -1] : null;
            let estimatedPriceForMissingQuarter = regionDetails.avgPrice * (0.95 + Math.random() * 0.1); 
            if(previousQuarterData) { 
                estimatedPriceForMissingQuarter = previousQuarterData.price * (0.995 + Math.random() * 0.01);
            }
            quarterlyPriceHistory.push({
                quarter: key,
                price: Math.round(estimatedPriceForMissingQuarter / 1000) * 1000
            });
         }
      }
    }
  }

  quarterlyPriceHistory.sort((a, b) => {
    const [aQStr, aYStr] = a.quarter.split(" ");
    const [bQStr, bYStr] = b.quarter.split(" ");
    const aY = parseInt(aYStr);
    const bY = parseInt(bYStr);
    const aQ = parseInt(aQStr.substring(1));
    const bQ = parseInt(bQStr.substring(1));
    if (aY !== bY) return aY - bY;
    return aQ - bQ;
  });
  const limitedQuarterlyPriceHistory = quarterlyPriceHistory.slice(-12);

  let currentAveragePrice = regionDetails.avgPrice;
  const allRegionSalesPrices = regionSales.map(s => s.price).filter(p => typeof p === 'number') as number[];
  if (allRegionSalesPrices.length > 0) {
    currentAveragePrice = Math.round(allRegionSalesPrices.reduce((sum, p) => sum + p, 0) / allRegionSalesPrices.length / 1000) * 1000;
  }

  const outcodesWithCsvAvgPrice = londonOutcodes.map(oc => {
    const salesInOutcode = parsedCsvData.filter(s => s.outcode === oc.id && typeof s.price === 'number');
    if (salesInOutcode.length > 0) {
      const avg = salesInOutcode.reduce((sum, s) => sum + (s.price as number), 0) / salesInOutcode.length;
      return { ...oc, avgPrice: Math.round(avg / 1000) * 1000 };
    }
    return oc;
  }).filter(oc => oc.avgPrice > 0); 

  const sortedRegions = [...outcodesWithCsvAvgPrice].sort((a, b) => b.avgPrice - a.avgPrice);
  const rank = sortedRegions.findIndex(r => r.id === regionId) + 1;
  const totalRegions = sortedRegions.length;
  const priceRank = totalRegions > 0 && rank > 0 ? `Rank: ${rank} of ${totalRegions}` : "Rank: N/A";

  return {
    regionId: regionDetails.id,
    regionName: regionDetails.name,
    currentAveragePrice,
    quarterlyPriceHistory: limitedQuarterlyPriceHistory.length > 0 ? limitedQuarterlyPriceHistory : [{ quarter: `Avg. ${regionDetails.id}`, price: currentAveragePrice }],
    priceRank,
  };
};
