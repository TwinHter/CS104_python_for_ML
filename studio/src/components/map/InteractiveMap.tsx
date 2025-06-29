
"use client";

import React from 'react';
import Image from 'next/image';
import type { OutcodeData } from '@/types'; // Not directly used but good for context
import { PLACEHOLDER_HINTS } from '@/lib/constants';

interface InteractiveMapProps {
  regionsData: OutcodeData[]; // Not used for rendering map, but might be useful if functionality expands
  onRegionSelect: (regionId: string) => void; // Not used by this static version
  selectedRegionId?: string | null; // Not used by this static version
}

const InteractiveMap: React.FC<InteractiveMapProps> = ({ regionsData, onRegionSelect, selectedRegionId }) => {
  return (
    <div className="bg-muted p-2 rounded-lg shadow-md aspect-auto max-h-[600px] overflow-hidden flex flex-col justify-center items-center">
      <Image
        src="https://www.doogal.co.uk/images/london_postcode_map.gif" 
        alt="Map of London outcodes"
        width={800} 
        height={600} 
        className="w-full h-auto object-contain rounded-md"
        data-ai-hint={PLACEHOLDER_HINTS.londonMap}
        unoptimized // Required for GIFs if not using a custom loader
      />
      <p className="text-xs text-muted-foreground mt-2 text-center">
        Illustrative map of London regions. Please use the list or filters to select an outcode for details.
      </p>
    </div>
  );
};

export default InteractiveMap;

    