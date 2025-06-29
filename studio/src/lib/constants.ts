
import type { PropertyType as AppPropertyType, EnergyRating, Tenure as AppTenure } from '@/types'; // Renamed to avoid conflict
import { londonOutcodes } from '@/lib/data/london_outcodes_data'; 
import { TrendingUp, MapIcon as PageMapIcon, ListChecksIcon, Database, ShieldCheck, Users } from 'lucide-react'; 

// App Info
export const APP_NAME = "London Housing";
export const APP_TAGLINE = "AI-powered London house price prediction and property insights."; 
export const APP_TITLE_TEMPLATE = (pageTitle?: string) => 
  pageTitle ? `${pageTitle} | ${APP_NAME}` : `${APP_NAME} - ${APP_TAGLINE}`;


// Navbar
export const NAV_ITEMS = [
  { href: '/', label: 'Introduction' },
  { href: '/prediction', label: 'Prediction' },
  { href: '/map', label: 'Map' },
  { href: '/recommendations', label: 'Recommendations' },
];

// Introduction Page
export const INTRO_HERO_TITLE = APP_NAME;
export const INTRO_HERO_DESCRIPTION = "Explore the future of London's property market with our intelligent house price prediction tool, utilizing advanced AI and detailed historical data.";
export const INTRO_HERO_CTA_TEXT = "Get Started with Prediction";
export const INTRO_HERO_CTA_LINK = "/prediction";

export const INTRO_FEATURES_SECTION_TITLE = "Application Features";
export const INTRO_FEATURE_CARDS = [
  { 
    title: "AI Price Prediction & Forecast", 
    description: "Input property details (address, type, size, rooms, etc.) to get an AI-driven price estimate for a specific sale month and a 12-month price trend forecast.", 
    IconComponent: TrendingUp 
  },
  { 
    title: "London Outcode Explorer", 
    description: "Explore average prices and market trends for London outcodes. Select a region to see its current average price, historical quarterly prices, and price rank.", 
    IconComponent: PageMapIcon
  },
  { 
    title: "Property Recommendations", 
    description: "Find properties matching your criteria: price range, type, area, rooms, and more. Upload your own listings to share.", 
    IconComponent: ListChecksIcon
  },
];

export const INTRO_DATA_AI_SECTION_TITLE = "Data & AI Model";
export const INTRO_DATA_AI_CARD_DESCRIPTION = "The foundation of our analysis";
export const INTRO_DATA_AI_TEXT_P1 = "Our project utilizes comprehensive London housing data, including historical transactions from the <strong>UK Land Registry Price Paid Data</strong> (1991-2023), enriched with:";
export const INTRO_DATA_AI_CHARACTERISTICS_LIST = [
  "<strong>Location:</strong> Full Address, Geolocation (Latitude, Longitude), Outcode.",
  "<strong>Characteristics:</strong> Property Type, Tenure.",
  "<strong>Size & Layout:</strong> Bedrooms, Bathrooms, Living Rooms, Floor Area (sqm).",
  "<strong>Condition:</strong> Current Energy Efficiency Rating (A-G).",
  "<strong>Timing:</strong> Month and Year of Sale for accurate predictions.",
];
export const INTRO_DATA_AI_MODELS_TEXT = "We employ and fine-tune advanced machine learning models including <strong>XGBoost</strong>, <strong>Neural Networks</strong>, and <strong>Recurrent Neural Networks (RNNs)</strong>, trained on meticulously processed data to identify complex price-influencing factors.";
export const INTRO_DATA_AI_ICON = Database;

export const INTRO_ACCURACY_SECTION_TITLE = "Model Accuracy";
export const INTRO_ACCURACY_CARD_DESCRIPTION = "Evaluating our model's performance";
export const INTRO_ACCURACY_TEXT = "Model accuracy is paramount. We use a comprehensive suite of metrics including <strong>MAE (Mean Absolute Error)</strong>, <strong>MSE (Mean Squared Error)</strong>, <strong>R² (R-squared)</strong>, and <strong>RMSE (Root Mean Squared Error)</strong> to continuously evaluate and refine our predictions, aiming for maximum reliability.";
export const INTRO_ACCURACY_CHART_TEXT = "Example chart showing model performance (e.g., MAE, R²).";
export const INTRO_ACCURACY_ICON = ShieldCheck;

export const INTRO_TEAM_SECTION_TITLE = "Project Team";
export const INTRO_TEAM_ICON = Users;
export const INTRO_TEAM_MEMBERS = [
  { name: "Nguyễn Hữu Đặng Nguyên", role: "ID: 23521045" },
  { name: "Trần Vạn Tấn", role: "ID: 23521813" },
  { name: "Trần Vinh Khánh", role: "ID: 23520729" },
];


// Page Hero Texts
export const PREDICTION_PAGE_HERO_TITLE = "Property Price Prediction";
export const PREDICTION_PAGE_HERO_DESCRIPTION = "Enter the property details below to receive an AI-powered price prediction and market insights. Longitude and Latitude are auto-filled from the address.";

export const MAP_PAGE_HERO_TITLE = "London Regional Statistics Explorer";
export const MAP_PAGE_HERO_DESCRIPTION = "Explore London's outcodes and their market statistics. The map image is illustrative. Use filters and the list below to select a region and view its current average price, historical quarterly price trends, and market rank.";

export const RECOMMENDATIONS_PAGE_HERO_TITLE = "Suitable Property Recommendations";
export const RECOMMENDATIONS_PAGE_HERO_DESCRIPTION = "Find your ideal London property based on your budget and requirements. Explore our curated list or upload your own.";

export const CONTACT_PAGE_TITLE_TEMPLATE = (propertyName: string) => `Enquire About: ${propertyName}`;

// Form Select Options
export const PROPERTY_TYPE_OPTIONS: AppPropertyType[] = [
    'Semi-Detached House', 'Terrace Property', 'Purpose Built Flat',
    'Mid Terrace House', 'Converted Flat', 'End Terrace House',
    'Flat/Maisonette', 'Detached House', 'Terraced',
    'Bungalow Property', 'Mid Terrace Property',
    'Detached Bungalow', 'Semi-Detached Bungalow',
    'Mid Terrace Bungalow', 'Semi-Detached Property',
    'Detached Property', 'End Terrace Property', 'Terraced Bungalow',
    'End Terrace Bungalow', 'Flat', 'Detached', 'Semi-detached', 'Bungalow', 'Maisonette' 
];
export const ENERGY_RATING_OPTIONS: EnergyRating[] = ['A', 'B', 'C', 'D', 'E', 'F', 'G'];
export const TENURE_OPTIONS: AppTenure[] = ['Freehold', 'Leasehold', 'Feudal', 'Shared'];
export const BEDROOM_OPTIONS: number[] = [0, 1, 2, 3, 4, 5, 6]; 
export const BATHROOM_OPTIONS: number[] = [0, 1, 2, 3, 4, 5];
export const RECEPTION_OPTIONS: number[] = [0, 1, 2, 3, 4]; 
export const LIVING_ROOM_OPTIONS: number[] = [0, 1, 2, 3, 4];


export const REGION_OPTIONS: string[] = [
 'E1', 'E10', 'E11', 'E13', 'E14', 'E15', 'E17', 'E18', 'E3', 'E4',
 'E5', 'E7', 'E8', 'EC1Y', 'EC2Y', 'EC4A', 'N10', 'N11', 'N12',
 'N13', 'N14', 'N15', 'N16', 'N17', 'N18', 'N19', 'N2', 'N20',
 'N21', 'N22', 'N4', 'N5', 'N6', 'N7', 'N8', 'N9', 'NW1', 'NW10',
 'NW11', 'NW2', 'NW3', 'NW4', 'NW5', 'NW6', 'NW7', 'NW8', 'SE1',
 'SE10', 'SE11', 'SE12', 'SE13', 'SE14', 'SE15', 'SE16', 'SE17',
 'SE18', 'SE19', 'SE2', 'SE20', 'SE21', 'SE23', 'SE24', 'SE25',
 'SE26', 'SE27', 'SE28', 'SE3', 'SE4', 'SE5', 'SE6', 'SE7', 'SE9',
 'SW10', 'SW11', 'SW12', 'SW13', 'SW14', 'SW15', 'SW16', 'SW17',
 'SW18', 'SW19', 'SW1E', 'SW1P', 'SW1V', 'SW1X', 'SW1Y', 'SW2',
 'SW3', 'SW4', 'SW5', 'SW6', 'SW7', 'SW8', 'W10', 'W11', 'W12',
 'W13', 'W14', 'W1B', 'W1D', 'W1F', 'W1H', 'W1J', 'W1K', 'W1W',
 'W2', 'W4', 'W5', 'W6', 'W7', 'W8', 'W9', 'WC1B', 'WC1H', 'WC1R',
 'WC1X', 'WC2B', 'WC2H', 'WC2N', 'E12', 'E6', 'N1', 'N3', 'NW9',
 'SE22', 'SE8', 'SW1H', 'SW20', 'SW9', 'W1G', 'W1T', 'W3', 'WC1A',
 'WC1E', 'WC2E', 'E2', 'E9', 'EC1R', 'EC2A', 'SW1W', 'W1U', 'WC1N',
 'EC4R', 'E16', 'EC1N', 'SW1A', 'W1S', 'EC1A', 'EC1M', 'EC4V',
 'E1W', 'EC3V', 'EC4Y', 'EC1V', 'EC4M', 'EC3A', 'EC3N', 'WC2R',
 'EC3M', 'EC2M', 'EC3R', 'WC2A', 'WC1V', 'EC2V', 'EC2R', 'W1C'
].sort();

// Placeholder Image Hints
export const PLACEHOLDER_HINTS = {
  londonMap: "london postcode map", 
  accuracyChart: "graph accuracy",
  priceChart: "graph price trend",
  defaultHouse: "house exterior",
  salesmanPortrait: "professional portrait",
  uploadedProperty: "uploaded property",
  propertyGeneric: "property image"
};

// File Upload Constants
export const MAX_FILE_SIZE_MB = 5;
export const MAX_FILE_SIZE_BYTES = MAX_FILE_SIZE_MB * 1024 * 1024;
export const ACCEPTED_IMAGE_TYPES = ["image/jpeg", "image/jpg", "image/png", "image/webp"];
export const ACCEPTED_IMAGE_TYPES_STRING = ".jpg, .jpeg, .png, .webp";

// Calculate min/max price defaults after londonOutcodes is populated
export const MIN_PRICE_FILTER_DEFAULT = londonOutcodes.length > 0 ? Math.max(0, Math.min(...londonOutcodes.map(o => o.avgPrice), 100000)) : 100000;
export const MAX_PRICE_FILTER_DEFAULT = londonOutcodes.length > 0 ? Math.max(...londonOutcodes.map(o => o.avgPrice), 3000000) : 3000000;


// Prediction Form Constants
export const PREDICTION_FORM_DEFAULT_BEDROOMS = 1;
export const PREDICTION_FORM_DEFAULT_BATHROOMS = 1;
export const PREDICTION_FORM_DEFAULT_LIVING_ROOMS = 1;
export const PREDICTION_MONTH_OF_SALE_FORMAT_DESC = "Month of sale must be in YYYY-MM format (e.g., 2024-07).";

// Salesman default info
export const DEFAULT_SALESMAN_INFO = {
    name: "London Dwellings AI Team",
    email: "contact@londondwellings.ai",
    phone: "+44 123 456 7890",
    bio: "Our dedicated team is here to assist you with your property inquiries.",
    imageUrl: `https://placehold.co/150x150.png`,
    dataAiHint: "team avatar"
};

    
