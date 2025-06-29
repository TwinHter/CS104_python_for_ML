
"use client";

import { useState, useMemo, useEffect, type ChangeEvent, useRef } from 'react';
import Image from 'next/image';
import Link from 'next/link';
import { useSearchParams } from 'next/navigation';
import { useForm, type SubmitHandler } from 'react-hook-form';
import { zodResolver } from '@hookform/resolvers/zod';
import { z } from 'zod';
import PageHero from '@/components/shared/PageHero';
import { Button } from '@/components/ui/button';
import { Card, CardContent, CardDescription, CardFooter, CardHeader, CardTitle } from '@/components/ui/card';
import { Input } from '@/components/ui/input';
import { Textarea } from '@/components/ui/textarea';
import { Select, SelectContent, SelectItem, SelectTrigger, SelectValue } from '@/components/ui/select';
import { Slider } from '@/components/ui/slider';
import { Form, FormControl, FormDescription, FormField, FormItem, FormLabel, FormMessage } from '@/components/ui/form';
import { useRecommend, type NewPropertyData } from '@/hooks/useRecommend';
import type { Property, PropertyType, EnergyRating, Tenure } from '@/types';
import { 
  DollarSign, Home, BedDouble, Search, MapPin, ListFilter, Bath, Sofa, Zap, FileText, Tv2, Contact, UploadCloud, User, MailIcon, Building2, Coins, X, Loader2, Phone
} from 'lucide-react';
import {
  RECOMMENDATIONS_PAGE_HERO_TITLE,
  RECOMMENDATIONS_PAGE_HERO_DESCRIPTION,
  MAX_PRICE_FILTER_DEFAULT,
  MIN_PRICE_FILTER_DEFAULT,
  PROPERTY_TYPE_OPTIONS,
  BEDROOM_OPTIONS,
  REGION_OPTIONS, // This is actually outcode options
  BATHROOM_OPTIONS,
  RECEPTION_OPTIONS, // Kept for general room counts if needed, but livingRooms is primary
  TENURE_OPTIONS,
  ENERGY_RATING_OPTIONS,
  MAX_FILE_SIZE_MB,
  MAX_FILE_SIZE_BYTES,
  ACCEPTED_IMAGE_TYPES,
  ACCEPTED_IMAGE_TYPES_STRING,
  PLACEHOLDER_HINTS
} from '@/lib/constants';

type RecommendationFilters = {
  maxPrice?: number;
  propertyType?: Property['propertyType'];
  outcode?: string; 
  bedrooms?: number;
  bathrooms?: number;
  livingRooms?: number; 
  tenure?: Property['tenure'];
  currentEnergyRating?: Property['currentEnergyRating']; 
};


const uploadPropertyFormSchema = z.object({
  name: z.string().min(5, "Property name must be at least 5 characters."),
  fullAddress: z.string().min(10, "Full address must be at least 10 characters."), 
  price: z.coerce.number().positive("Price must be a positive number."),
  propertyType: z.enum(PROPERTY_TYPE_OPTIONS as [PropertyType, ...PropertyType[]], { required_error: "Property type is required." }), 
  bedrooms: z.coerce.number().int().min(0, "Bedrooms must be 0 or more."),
  bathrooms: z.coerce.number().int().min(0, "Bathrooms must be 0 or more."),
  livingRooms: z.coerce.number().int().min(0, "Living rooms must be 0 or more."), 
  floorAreaSqM: z.coerce.number().positive("Area must be a positive number.").optional(), 
  currentEnergyRating: z.enum(ENERGY_RATING_OPTIONS as [EnergyRating, ...EnergyRating[]], { required_error: "Energy rating is required." }), 
  tenure: z.enum(TENURE_OPTIONS as [Tenure, ...Tenure[]], { required_error: "Tenure is required." }),
  outcode: z.enum(REGION_OPTIONS as [string, ...string[]], { required_error: "Outcode is required." }), 
  description: z.string().min(20, "Description must be at least 20 characters.").max(500, "Description cannot exceed 500 characters."),
  imageFile: z
    .custom<FileList>()
    .refine((files) => files && files.length === 1, "Property image is required.")
    .refine((files) => files && files?.[0]?.size <= MAX_FILE_SIZE_BYTES, `Max image size is ${MAX_FILE_SIZE_MB}MB.`)
    .refine(
      (files) => files && ACCEPTED_IMAGE_TYPES.includes(files?.[0]?.type),
      `Only ${ACCEPTED_IMAGE_TYPES_STRING} formats are supported.`
    ),
  uploaderName: z.string().min(2, "Your name must be at least 2 characters."),
  uploaderEmail: z.string().email("Please enter a valid email address."),
  uploaderPhone: z.string().optional(),
});

type UploadPropertyFormValues = z.infer<typeof uploadPropertyFormSchema>;

export default function RecommendationsPage() {
  const searchParams = useSearchParams();
  const initialOutcode = searchParams.get('region'); 

  const { 
    properties: displayedProperties, 
    isLoadingProperties, 
    fetchPropertiesError,
    addProperty,
    isAddingProperty
  } = useRecommend();

  const [showUploadForm, setShowUploadForm] = useState(false);
  const uploadSectionRef = useRef<HTMLDivElement>(null);

  const currentMaxPrice = useMemo(() => Math.max(...(displayedProperties || []).map(p => p.price), MAX_PRICE_FILTER_DEFAULT), [displayedProperties]);
  const currentMinPrice = useMemo(() => Math.min(...(displayedProperties || []).map(p => p.price), MIN_PRICE_FILTER_DEFAULT), [displayedProperties]);

  const [filters, setFilters] = useState<RecommendationFilters>({
    maxPrice: currentMaxPrice,
    outcode: initialOutcode || undefined,
  });
  const [searchTerm, setSearchTerm] = useState('');

  const uploadForm = useForm<UploadPropertyFormValues>({
    resolver: zodResolver(uploadPropertyFormSchema),
    defaultValues: {
      name: "", fullAddress: "", price: undefined, propertyType: undefined, 
      bedrooms: BEDROOM_OPTIONS[1], bathrooms: BATHROOM_OPTIONS[1], livingRooms: RECEPTION_OPTIONS[1], 
      floorAreaSqM: undefined, currentEnergyRating: undefined, tenure: undefined, outcode: undefined, 
      description: "", imageFile: undefined, uploaderName: "", uploaderEmail: "", uploaderPhone: ""
    },
  });

  useEffect(() => {
    if (initialOutcode) {
      setFilters(prev => ({ ...prev, outcode: initialOutcode, maxPrice: currentMaxPrice }));
    } else {
      setFilters(prev => ({ ...prev, maxPrice: currentMaxPrice }));
    }
  }, [initialOutcode, currentMaxPrice]);

  useEffect(() => {
    setFilters(prev => ({ ...prev, maxPrice: currentMaxPrice }));
  }, [currentMaxPrice]);

  useEffect(() => {
    if (showUploadForm && uploadSectionRef.current) {
      uploadSectionRef.current.scrollIntoView({ behavior: 'smooth', block: 'start' });
    }
  }, [showUploadForm]);

  const filteredProperties = useMemo(() => {
    if (!displayedProperties) return [];
    return displayedProperties.filter(property => {
      if (filters.maxPrice && property.price > filters.maxPrice) return false;
      if (filters.propertyType && property.propertyType !== filters.propertyType) return false;
      if (filters.outcode && property.outcode !== filters.outcode) return false;
      if (filters.bedrooms !== undefined && property.bedrooms < filters.bedrooms) return false;
      if (filters.bathrooms !== undefined && property.bathrooms < filters.bathrooms) return false;
      if (filters.livingRooms !== undefined && property.livingRooms < filters.livingRooms) return false;
      if (filters.tenure && property.tenure !== filters.tenure) return false;
      if (filters.currentEnergyRating && property.currentEnergyRating !== filters.currentEnergyRating) return false;
      if (searchTerm && !`${property.name} ${property.fullAddress} ${property.description} ${property.uploaderName}`.toLowerCase().includes(searchTerm.toLowerCase())) return false;
      return true;
    });
  }, [filters, searchTerm, displayedProperties]);

  const handleFilterChange = (key: keyof RecommendationFilters, value: any) => {
    let processedValue = value;
    if (value === 'all' || value === '') {
      processedValue = undefined;
    } else if ((key === 'bedrooms' || key === 'bathrooms' || key === 'livingRooms') && typeof value === 'string') {
      processedValue = parseInt(value);
      if (isNaN(processedValue)) processedValue = undefined;
    }
    setFilters(prev => ({ ...prev, [key]: processedValue }));
  };
  
  const handleSearchChange = (event: ChangeEvent<HTMLInputElement>) => {
    setSearchTerm(event.target.value);
  };

  const handleFileUploadSubmit: SubmitHandler<UploadPropertyFormValues> = async (data) => {
    const propertyDataForHook: NewPropertyData = {
      name: data.name,
      fullAddress: data.fullAddress,
      price: data.price,
      propertyType: data.propertyType,
      bedrooms: data.bedrooms,
      bathrooms: data.bathrooms,
      livingRooms: data.livingRooms,
      floorAreaSqM: data.floorAreaSqM,
      currentEnergyRating: data.currentEnergyRating,
      tenure: data.tenure,
      outcode: data.outcode,
      description: data.description,
      imageFile: data.imageFile,
      uploaderName: data.uploaderName,
      uploaderEmail: data.uploaderEmail,
      uploaderPhone: data.uploaderPhone,
    };
    try {
      await addProperty(propertyDataForHook);
      uploadForm.reset();
      setShowUploadForm(false);
    } catch (e) {
      console.error("Upload submit error", e);
    }
  };

  return (
    <div className="space-y-12">
      <PageHero
        title={RECOMMENDATIONS_PAGE_HERO_TITLE}
        description={RECOMMENDATIONS_PAGE_HERO_DESCRIPTION}
      />

      <Card className="shadow-xl animate-fadeIn" style={{animationDelay: '0.2s'}}>
        <CardHeader>
          <CardTitle className="font-headline text-xl flex items-center"><ListFilter className="mr-2 h-5 w-5 text-primary"/>Search Filters</CardTitle>
          <CardDescription>Refine your criteria to find your dream home.</CardDescription>
        </CardHeader>
        <CardContent className="space-y-6">
          <div className="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-4 gap-6 items-end">
            <div>
              <label htmlFor="maxPriceFilterRec" className="block text-sm font-medium text-foreground mb-1">Max Price (£)</label>
              <Slider
                id="maxPriceFilterRec"
                value={[filters.maxPrice || currentMaxPrice]}
                max={currentMaxPrice} 
                min={currentMinPrice}
                step={50000}
                onValueChange={(value) => handleFilterChange('maxPrice', value[0])}
                disabled={isLoadingProperties || !displayedProperties || displayedProperties.length === 0}
              />
              <div className="text-xs text-muted-foreground mt-1 text-right">
                £{(filters.maxPrice || currentMaxPrice).toLocaleString()}
              </div>
            </div>
            <div>
              <label htmlFor="propertyTypeFilterRec" className="block text-sm font-medium text-foreground mb-1">Property Type</label>
              <Select onValueChange={(value) => handleFilterChange('propertyType', value)} value={filters.propertyType || 'all'}>
                <SelectTrigger id="propertyTypeFilterRec"><SelectValue placeholder="All types" /></SelectTrigger>
                <SelectContent>
                  <SelectItem value="all">All types</SelectItem>
                  {PROPERTY_TYPE_OPTIONS.map(pt => <SelectItem key={pt} value={pt}>{pt}</SelectItem>)}
                </SelectContent>
              </Select>
            </div>
            <div>
              <label htmlFor="outcodeFilterRec" className="block text-sm font-medium text-foreground mb-1">Outcode</label>
              <Select onValueChange={(value) => handleFilterChange('outcode', value)} value={filters.outcode || 'all'}>
                <SelectTrigger id="outcodeFilterRec"><SelectValue placeholder="All outcodes" /></SelectTrigger>
                <SelectContent>
                  <SelectItem value="all">All outcodes</SelectItem>
                  {REGION_OPTIONS.map(oc => <SelectItem key={oc} value={oc}>{oc}</SelectItem>)}
                </SelectContent>
              </Select>
            </div>
            <div>
              <label htmlFor="bedroomsFilterRec" className="block text-sm font-medium text-foreground mb-1">Min. Bedrooms</label>
              <Select onValueChange={(value) => handleFilterChange('bedrooms', value)} value={filters.bedrooms?.toString() || 'all'}>
                <SelectTrigger id="bedroomsFilterRec"><SelectValue placeholder="Any" /></SelectTrigger>
                <SelectContent>
                  <SelectItem value="all">Any</SelectItem>
                  {BEDROOM_OPTIONS.map(num => <SelectItem key={num} value={String(num)}>{num === 0 ? 'Studio / 0 beds' : `${num}+ beds`}</SelectItem>)}
                </SelectContent>
              </Select>
            </div>
            <div>
              <label htmlFor="bathroomsFilterRec" className="block text-sm font-medium text-foreground mb-1">Min. Bathrooms</label>
              <Select onValueChange={(value) => handleFilterChange('bathrooms', value)} value={filters.bathrooms?.toString() || 'all'}>
                <SelectTrigger id="bathroomsFilterRec"><SelectValue placeholder="Any" /></SelectTrigger>
                <SelectContent>
                  <SelectItem value="all">Any</SelectItem>
                  {BATHROOM_OPTIONS.map(num => <SelectItem key={num} value={String(num)}>{num}+ baths</SelectItem>)}
                </SelectContent>
              </Select>
            </div>
             <div>
              <label htmlFor="livingRoomsFilterRec" className="block text-sm font-medium text-foreground mb-1">Min. Living Rooms</label>
              <Select onValueChange={(value) => handleFilterChange('livingRooms', value)} value={filters.livingRooms?.toString() || 'all'}>
                <SelectTrigger id="livingRoomsFilterRec"><SelectValue placeholder="Any" /></SelectTrigger>
                <SelectContent>
                  <SelectItem value="all">Any</SelectItem>
                  {RECEPTION_OPTIONS.map(num => <SelectItem key={num} value={String(num)}>{num}+ rooms</SelectItem>)}
                </SelectContent>
              </Select>
            </div>
             <div>
              <label htmlFor="tenureFilterRec" className="block text-sm font-medium text-foreground mb-1">Tenure</label>
              <Select onValueChange={(value) => handleFilterChange('tenure', value)} value={filters.tenure || 'all'}>
                <SelectTrigger id="tenureFilterRec"><SelectValue placeholder="Any tenure" /></SelectTrigger>
                <SelectContent>
                  <SelectItem value="all">Any tenure</SelectItem>
                  {TENURE_OPTIONS.map(t => <SelectItem key={t} value={t}>{t}</SelectItem>)}
                </SelectContent>
              </Select>
            </div>
            <div>
              <label htmlFor="currentEnergyRatingFilterRec" className="block text-sm font-medium text-foreground mb-1">Energy Rating</label>
              <Select onValueChange={(value) => handleFilterChange('currentEnergyRating', value)} value={filters.currentEnergyRating || 'all'}>
                <SelectTrigger id="currentEnergyRatingFilterRec"><SelectValue placeholder="Any rating" /></SelectTrigger>
                <SelectContent>
                  <SelectItem value="all">Any rating</SelectItem>
                  {ENERGY_RATING_OPTIONS.map(rating => <SelectItem key={rating} value={rating}>Rating {rating}</SelectItem>)}
                </SelectContent>
              </Select>
            </div>
          </div>
          <div>
            <label htmlFor="searchTermRec" className="block text-sm font-medium text-foreground mb-1">Keyword Search</label>
            <Input 
              id="searchTermRec" 
              placeholder="e.g., near park, balcony, victorian, uploader name..." 
              value={searchTerm}
              onChange={handleSearchChange}
              className="bg-card"
            />
          </div>
        </CardContent>
      </Card>

      <div className="my-8 text-center animate-fadeIn" style={{animationDelay: '0.2s'}}>
        {!showUploadForm && (
          <Button onClick={() => setShowUploadForm(true)} size="lg" variant="outline" className="border-primary text-primary hover:bg-primary/10 hover:text-primary">
            <UploadCloud className="mr-2 h-5 w-5" /> List Your Property
          </Button>
        )}
      </div>

      {showUploadForm && (
        <div ref={uploadSectionRef}>
          <Card className="shadow-xl animate-fadeIn mb-12" style={{ animationDelay: '0.1s' }}>
            <CardHeader>
              <div className="flex justify-between items-center">
                <CardTitle className="font-headline text-xl flex items-center"><UploadCloud className="mr-2 h-5 w-5 text-primary" />Upload New Property Listing</CardTitle>
                <Button variant="ghost" size="icon" onClick={() => setShowUploadForm(false)} aria-label="Close upload form">
                  <X className="h-5 w-5" />
                </Button>
              </div>
              <CardDescription>Contribute to our listings.</CardDescription>
            </CardHeader>
            <Form {...uploadForm}>
              <form onSubmit={uploadForm.handleSubmit(handleFileUploadSubmit)}>
                <CardContent className="space-y-6">
                  <h3 className="text-lg font-semibold text-foreground border-b pb-2">Property Details</h3>
                  <div className="grid grid-cols-1 md:grid-cols-2 gap-6">
                    <FormField control={uploadForm.control} name="name" render={({ field }) => (
                      <FormItem><FormLabel className="flex items-center"><Home className="mr-2 h-4 w-4 text-muted-foreground" />Property Name</FormLabel><FormControl><Input placeholder="e.g., Modern City Apartment" {...field} /></FormControl><FormMessage /></FormItem>)} />
                    <FormField control={uploadForm.control} name="fullAddress" render={({ field }) => (
                      <FormItem><FormLabel className="flex items-center"><MapPin className="mr-2 h-4 w-4 text-muted-foreground" />Full Address</FormLabel><FormControl><Input placeholder="e.g., 123 Main St, London, E1 1AB" {...field} /></FormControl><FormMessage /></FormItem>)} />
                  </div>
                  <div className="grid grid-cols-1 md:grid-cols-3 gap-6">
                    <FormField control={uploadForm.control} name="price" render={({ field }) => (
                      <FormItem><FormLabel className="flex items-center"><Coins className="mr-2 h-4 w-4 text-muted-foreground" />Price (£)</FormLabel><FormControl><Input type="number" placeholder="e.g., 500000" {...field} onChange={e => field.onChange(e.target.value === '' ? undefined : parseFloat(e.target.value))} value={field.value ?? ""} /></FormControl><FormMessage /></FormItem>)} />
                    <FormField control={uploadForm.control} name="propertyType" render={({ field }) => (
                      <FormItem><FormLabel className="flex items-center"><Building2 className="mr-2 h-4 w-4 text-muted-foreground" />Property Type</FormLabel><Select onValueChange={field.onChange} defaultValue={field.value}><FormControl><SelectTrigger><SelectValue placeholder="Select type" /></SelectTrigger></FormControl><SelectContent>{PROPERTY_TYPE_OPTIONS.map(pt => <SelectItem key={pt} value={pt}>{pt}</SelectItem>)}</SelectContent></Select><FormMessage /></FormItem>)} />
                    <FormField control={uploadForm.control} name="outcode" render={({ field }) => (
                      <FormItem><FormLabel className="flex items-center"><MapPin className="mr-2 h-4 w-4 text-muted-foreground" />Outcode</FormLabel><Select onValueChange={field.onChange} defaultValue={field.value}><FormControl><SelectTrigger><SelectValue placeholder="Select outcode" /></SelectTrigger></FormControl><SelectContent>{REGION_OPTIONS.map(oc => <SelectItem key={oc} value={oc}>{oc}</SelectItem>)}</SelectContent></Select><FormMessage /></FormItem>)} />
                  </div>
                  <div className="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-4 gap-6">
                    <FormField control={uploadForm.control} name="bedrooms" render={({ field }) => (
                      <FormItem><FormLabel className="flex items-center"><BedDouble className="mr-2 h-4 w-4 text-muted-foreground" />Bedrooms</FormLabel><FormControl><Input type="number" placeholder="e.g., 2" {...field} /></FormControl><FormMessage /></FormItem>)} />
                    <FormField control={uploadForm.control} name="bathrooms" render={({ field }) => (
                      <FormItem><FormLabel className="flex items-center"><Bath className="mr-2 h-4 w-4 text-muted-foreground" />Bathrooms</FormLabel><FormControl><Input type="number" placeholder="e.g., 1" {...field} /></FormControl><FormMessage /></FormItem>)} />
                    <FormField control={uploadForm.control} name="livingRooms" render={({ field }) => (
                      <FormItem><FormLabel className="flex items-center"><Tv2 className="mr-2 h-4 w-4 text-muted-foreground" />Living Rooms</FormLabel><FormControl><Input type="number" placeholder="e.g., 1" {...field} /></FormControl><FormMessage /></FormItem>)} />
                    <FormField control={uploadForm.control} name="floorAreaSqM" render={({ field }) => (
                      <FormItem><FormLabel className="flex items-center"><Home className="mr-2 h-4 w-4 text-muted-foreground"/>Floor Area (sqm)</FormLabel><FormControl><Input type="number" placeholder="e.g., 70 (optional)" {...field} onChange={e => field.onChange(e.target.value === '' ? undefined : parseFloat(e.target.value))} value={field.value ?? ""} /></FormControl><FormMessage /></FormItem>)} />
                  </div>
                  <div className="grid grid-cols-1 md:grid-cols-2 gap-6">
                    <FormField control={uploadForm.control} name="tenure" render={({ field }) => (
                      <FormItem><FormLabel className="flex items-center"><FileText className="mr-2 h-4 w-4 text-muted-foreground" />Tenure</FormLabel><Select onValueChange={field.onChange} defaultValue={field.value}><FormControl><SelectTrigger><SelectValue placeholder="Select tenure" /></SelectTrigger></FormControl><SelectContent>{TENURE_OPTIONS.map(t => <SelectItem key={t} value={t}>{t}</SelectItem>)}</SelectContent></Select><FormMessage /></FormItem>)} />
                    <FormField control={uploadForm.control} name="currentEnergyRating" render={({ field }) => (
                      <FormItem><FormLabel className="flex items-center"><Zap className="mr-2 h-4 w-4 text-muted-foreground" />Energy Rating</FormLabel><Select onValueChange={field.onChange} defaultValue={field.value}><FormControl><SelectTrigger><SelectValue placeholder="Select rating" /></SelectTrigger></FormControl><SelectContent>{ENERGY_RATING_OPTIONS.map(rating => <SelectItem key={rating} value={rating}>Rating {rating}</SelectItem>)}</SelectContent></Select><FormMessage /></FormItem>)} />
                  </div>
                  <FormField control={uploadForm.control} name="description" render={({ field }) => (
                    <FormItem><FormLabel>Description</FormLabel><FormControl><Textarea placeholder="Detailed description of the property..." {...field} rows={4} /></FormControl><FormMessage /></FormItem>)} />
                  <FormField control={uploadForm.control} name="imageFile" render={({ field: { onChange, value, ...rest } }) => (
                    <FormItem><FormLabel>Property Image</FormLabel><FormControl><Input type="file" accept={ACCEPTED_IMAGE_TYPES_STRING} onChange={(e) => onChange(e.target.files)} {...rest} className="file:text-primary file:font-medium"/></FormControl><FormDescription>Max file size: {MAX_FILE_SIZE_MB}MB. Accepted: {ACCEPTED_IMAGE_TYPES_STRING}.</FormDescription><FormMessage /></FormItem>)} />
                  
                  <h3 className="text-lg font-semibold text-foreground border-b pt-4 pb-2">Your Information</h3>
                  <div className="grid grid-cols-1 md:grid-cols-3 gap-6">
                    <FormField control={uploadForm.control} name="uploaderName" render={({ field }) => (
                      <FormItem><FormLabel className="flex items-center"><User className="mr-2 h-4 w-4 text-muted-foreground" />Your Name</FormLabel><FormControl><Input placeholder="e.g., John Doe" {...field} /></FormControl><FormMessage /></FormItem>)} />
                    <FormField control={uploadForm.control} name="uploaderEmail" render={({ field }) => (
                      <FormItem><FormLabel className="flex items-center"><MailIcon className="mr-2 h-4 w-4 text-muted-foreground" />Your Email</FormLabel><FormControl><Input type="email" placeholder="e.g., john.doe@example.com" {...field} /></FormControl><FormMessage /></FormItem>)} />
                    <FormField control={uploadForm.control} name="uploaderPhone" render={({ field }) => (
                      <FormItem><FormLabel className="flex items-center"><Phone className="mr-2 h-4 w-4 text-muted-foreground" />Your Phone (Optional)</FormLabel><FormControl><Input type="tel" placeholder="e.g., 020 1234 5678" {...field} value={field.value ?? ""} /></FormControl><FormMessage /></FormItem>)} />
                  </div>
                </CardContent>
                <CardFooter>
                  <Button type="submit" disabled={isAddingProperty || uploadForm.formState.isSubmitting} className="w-full">
                    { (isAddingProperty || uploadForm.formState.isSubmitting) ? <Loader2 className="mr-2 h-4 w-4 animate-spin" /> : <UploadCloud className="mr-2 h-4 w-4" />}
                    {(isAddingProperty || uploadForm.formState.isSubmitting) ? 'Uploading...' : 'Upload Property'}
                  </Button>
                </CardFooter>
              </form>
            </Form>
          </Card>
        </div>
      )}
    
      {isLoadingProperties && (
        <div className="flex justify-center items-center min-h-[300px]">
          <Loader2 className="h-12 w-12 animate-spin text-primary" />
          <p className="ml-4 text-lg">Loading properties...</p>
        </div>
      )}

      {fetchPropertiesError && (
         <Card className="shadow-xl animate-fadeIn bg-destructive/10 border-destructive/30">
            <CardHeader>
                <CardTitle className="font-headline text-xl text-destructive">Error Loading Properties</CardTitle>
            </CardHeader>
            <CardContent>
                <p>Could not load property recommendations: {fetchPropertiesError.message}</p>
            </CardContent>
         </Card>
      )}

      {!isLoadingProperties && !fetchPropertiesError && (
        <div className="animate-fadeIn" style={{animationDelay: '0.4s'}}>
          <h2 className="text-2xl font-headline font-semibold mb-6 text-foreground">
            {filteredProperties.length} propert{filteredProperties.length === 1 ? 'y' : 'ies'} found
          </h2>
          {filteredProperties.length > 0 ? (
            <div className="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-3 gap-8">
              {filteredProperties.map((property) => (
                <Card key={property.id} className="overflow-hidden shadow-lg hover:shadow-2xl transition-shadow duration-300 flex flex-col bg-card">
                  <CardHeader className="p-0 relative">
                    <Image 
                      src={property.image} 
                      alt={property.name} 
                      width={600} 
                      height={400} 
                      className="w-full h-56 object-cover"
                      data-ai-hint={property.dataAiHint || PLACEHOLDER_HINTS.defaultHouse}
                    />
                    <div className="absolute top-2 right-2 bg-primary/80 text-primary-foreground px-2 py-1 rounded-md text-xs font-semibold backdrop-blur-sm">
                      {property.outcode}
                    </div>
                    <div className="absolute bottom-2 left-2 bg-black/50 text-white px-2 py-1 rounded-md text-xs font-semibold backdrop-blur-sm flex items-center">
                      <Zap size={12} className="mr-1 text-yellow-400" /> Energy Rating: {property.currentEnergyRating}
                    </div>
                  </CardHeader>
                  <CardContent className="p-6 flex-grow">
                    <CardTitle className="font-headline text-xl mb-2 text-primary hover:underline">
                      <Link href={`/contact/${property.id}`}>{property.name}</Link>
                    </CardTitle>
                    <CardDescription className="text-sm text-muted-foreground mb-2 flex items-center">
                      <MapPin size={14} className="mr-1.5 flex-shrink-0" /> {property.fullAddress}
                    </CardDescription>
                    <div className="text-sm text-muted-foreground mb-3 space-y-1">
                      <div className="flex items-center"><Home size={14} className="mr-1.5" /> {property.propertyType} - {property.tenure}</div>
                      <div className="flex items-center"><BedDouble size={14} className="mr-1.5" /> {property.bedrooms} bed{property.bedrooms === 1 ? '' : 's'}</div>
                      <div className="flex items-center"><Bath size={14} className="mr-1.5" /> {property.bathrooms} bath{property.bathrooms === 1 ? '' : 's'}</div>
                      <div className="flex items-center"><Tv2 size={14} className="mr-1.5" /> {property.livingRooms} living room{property.livingRooms === 1 ? '' : 's'}</div>
                      {property.floorAreaSqM && <div className="flex items-center"><Home size={14} className="mr-1.5" /> {property.floorAreaSqM} m²</div>}
                    </div>
                    <p className="text-foreground/90 text-sm mb-4 line-clamp-3">{property.description}</p>
                  </CardContent>
                  <CardFooter className="p-6 bg-muted/20 border-t">
                    <div className="flex justify-between items-center w-full">
                      <p className="text-2xl font-bold text-primary flex items-center">
                        £{property.price.toLocaleString()}
                      </p>
                      <Button asChild>
                        <Link href={`/contact/${property.id}`}>
                          <Contact className="mr-2 h-4 w-4" /> Enquire
                        </Link>
                      </Button>
                    </div>
                  </CardFooter>
                </Card>
              ))}
            </div>
          ) : (
            <div className="text-center py-12">
              <Search size={48} className="mx-auto text-muted-foreground mb-4" />
              <p className="text-xl text-muted-foreground">No properties found matching your criteria.</p>
              <p className="text-sm text-muted-foreground mt-2">Try adjusting your filters or broadening your search.</p>
            </div>
          )}
        </div>
      )}
    </div>
  );
}

    