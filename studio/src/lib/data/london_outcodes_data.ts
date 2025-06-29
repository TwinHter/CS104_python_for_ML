
import type { OutcodeData } from '@/types';
import londonOutcodesExtendedData from './london_outcodes_extended_data.json';

// Use the extended data as the primary source for outcode information
export const londonOutcodes: OutcodeData[] = londonOutcodesExtendedData.map(data => ({
  id: data.id,
  name: data.name,
  avgPrice: data.avgPrice,
  priceCategory: data.priceCategory as 'low' | 'medium' | 'high', // Ensure type assertion
  description: data.description,
}));
