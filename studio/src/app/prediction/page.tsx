
"use client";

import { useForm, type SubmitHandler } from 'react-hook-form';
import { zodResolver } from '@hookform/resolvers/zod';
import { z } from 'zod';
import { usePredict } from '@/hooks/usePredict';
import type { PredictionInput, PredictionOutput } from '@/ai/flows/price-prediction';
import PageHero from '@/components/shared/PageHero';
import { Button } from '@/components/ui/button';
import { Card, CardContent, CardDescription, CardFooter, CardHeader, CardTitle } from '@/components/ui/card';
import { Form, FormControl, FormDescription, FormField, FormItem, FormLabel, FormMessage } from '@/components/ui/form';
import { Input } from '@/components/ui/input';
import { Select, SelectContent, SelectItem, SelectTrigger, SelectValue } from '@/components/ui/select';
import { Loader2, TrendingUp, Home, Coins, LineChart as LineChartIcon, MapPin, Building2, Bath, Sofa, Zap, FileText, CalendarDays, Tv2, HelpCircle } from 'lucide-react';
import { Line, LineChart, ResponsiveContainer, XAxis, YAxis, Tooltip as RechartsTooltip, CartesianGrid } from 'recharts';
import { ChartConfig, ChartContainer, ChartTooltipContent } from "@/components/ui/chart";
import { 
  PREDICTION_PAGE_HERO_TITLE, 
  PREDICTION_PAGE_HERO_DESCRIPTION,
  PROPERTY_TYPE_OPTIONS,
  ENERGY_RATING_OPTIONS,
  TENURE_OPTIONS,
  REGION_OPTIONS, // These are outcode options
  PREDICTION_FORM_DEFAULT_BEDROOMS,
  PREDICTION_FORM_DEFAULT_BATHROOMS,
  PREDICTION_FORM_DEFAULT_LIVING_ROOMS,
  PREDICTION_MONTH_OF_SALE_FORMAT_DESC
} from '@/lib/constants';
import type { PropertyType, EnergyRating, Tenure } from '@/types';
import { useEffect, useState, useCallback } from 'react';
import axios from 'axios';

// Schema for the form itself, using renamed fields where appropriate
const predictionFormClientSchema = z.object({
  fullAddress: z.string().min(5, { message: 'Full address must be at least 5 characters.' }),
  outcode: z.enum(REGION_OPTIONS as [string, ...string[]], { required_error: 'Outcode is required.' }),
  longitude: z.coerce.number().optional(),
  latitude: z.coerce.number().optional(),
  bedrooms: z.coerce.number().int().min(0, { message: 'Must be 0 or more bedrooms.' }).max(10, { message: 'Cannot exceed 10 bedrooms.' }),
  bathrooms: z.coerce.number().int().min(0, { message: 'Must be 0 or more bathrooms.' }).max(10, { message: 'Cannot exceed 10 bathrooms.' }),
  livingRooms: z.coerce.number().int().min(0, { message: 'Must be 0 or more living rooms.' }).max(10, { message: 'Cannot exceed 10 living rooms.' }),
  floorAreaSqM: z.coerce.number().positive({ message: 'Floor area must be a positive number.' }),
  tenure: z.enum(TENURE_OPTIONS as [Tenure, ...Tenure[]], { required_error: 'Tenure is required.' }),
  propertyType: z.enum(PROPERTY_TYPE_OPTIONS as [PropertyType, ...PropertyType[]], { required_error: 'Property type is required.' }),
  currentEnergyRating: z.enum(ENERGY_RATING_OPTIONS as [EnergyRating, ...EnergyRating[]], { required_error: 'Energy rating is required.' }),
  monthOfSale: z.string().regex(/^\d{4}-(0[1-9]|1[0-2])$/, { message: PREDICTION_MONTH_OF_SALE_FORMAT_DESC }),
});

type PredictionFormClientValues = z.infer<typeof predictionFormClientSchema>;

export default function PredictionPage() {
  const { predict, isPredicting, predictionData, predictionError, resetPrediction } = usePredict();
  const currentYear = new Date().getFullYear();
  const currentMonth = (new Date().getMonth() + 1).toString().padStart(2, '0');
  const [isGeocoding, setIsGeocoding] = useState(false);

  const form = useForm<PredictionFormClientValues>({
    resolver: zodResolver(predictionFormClientSchema),
    defaultValues: {
      fullAddress: '',
      outcode: undefined,
      longitude: undefined,
      latitude: undefined,
      bedrooms: PREDICTION_FORM_DEFAULT_BEDROOMS,
      bathrooms: PREDICTION_FORM_DEFAULT_BATHROOMS,
      livingRooms: PREDICTION_FORM_DEFAULT_LIVING_ROOMS,
      floorAreaSqM: undefined,
      tenure: undefined,
      propertyType: undefined,
      currentEnergyRating: undefined,
      monthOfSale: `${currentYear}-${currentMonth}`,
    },
  });

  const geocodeAddress = useCallback(async (addressToGeocode: string) => {
    const trimmedAddress = addressToGeocode.trim();
    if (trimmedAddress.length < 5) { 
      form.setValue('longitude', undefined);
      form.setValue('latitude', undefined);
      if (form.formState.errors.fullAddress?.type === 'manual') {
        form.clearErrors('fullAddress');
      }
      return;
    }
    setIsGeocoding(true);
    form.clearErrors('fullAddress');

    let processedAddress = trimmedAddress;
    if (!processedAddress.toLowerCase().includes('london')) {
      processedAddress += ', London';
    }

    try {
      const response = await axios.get(`https://nominatim.openstreetmap.org/search?format=json&q=${encodeURIComponent(processedAddress)}&countrycodes=gb&limit=1`);
      if (response.data && response.data.length > 0) {
        const { lat, lon } = response.data[0];
        form.setValue('longitude', parseFloat(lon), { shouldValidate: true });
        form.setValue('latitude', parseFloat(lat), { shouldValidate: true });
      } else {
        form.setError('fullAddress', { type: 'manual', message: 'Address not found. Please check or enter coordinates manually.' });
        form.setValue('longitude', undefined);
        form.setValue('latitude', undefined);
      }
    } catch (error) {
      console.error("Geocoding error:", error);
      form.setError('fullAddress', { type: 'manual', message: 'Geocoding failed. Please enter coordinates manually.' });
      form.setValue('longitude', undefined);
      form.setValue('latitude', undefined);
    } finally {
      setIsGeocoding(false);
    }
  }, [form]);

  const watchedAddress = form.watch('fullAddress');

  useEffect(() => {
    const handler = setTimeout(() => {
      if (watchedAddress !== undefined) { 
          const currentFormAddress = form.getValues('fullAddress');
          if (watchedAddress === currentFormAddress && watchedAddress.trim().length >= 5) { 
            geocodeAddress(watchedAddress);
          } else if (watchedAddress.trim().length < 5 && (form.getValues('latitude') !== undefined || form.getValues('longitude') !== undefined)) {
            form.setValue('longitude', undefined);
            form.setValue('latitude', undefined);
          }
      }
    }, 1000); 

    return () => {
      clearTimeout(handler);
    };
  }, [watchedAddress, geocodeAddress, form]);


  const onSubmit: SubmitHandler<PredictionFormClientValues> = async (data) => {
    resetPrediction();
    
    const [yearStr, monthStr] = data.monthOfSale.split('-');
    const sale_year_val = parseInt(yearStr, 10);
    const sale_month_val = parseInt(monthStr, 10);

    const inputDataForApi: PredictionInput = {
      fullAddress: data.fullAddress,
      latitude: data.latitude,
      longitude: data.longitude,
      bedrooms: data.bedrooms,
      bathrooms: data.bathrooms,
      livingRooms: data.livingRooms,
      sale_month: sale_month_val,
      sale_year: sale_year_val,
      tenure: data.tenure,
      currentEnergyRating: data.currentEnergyRating,
      floorAreaSqM: data.floorAreaSqM,
      outcode: data.outcode,
      propertyType: data.propertyType,
    };

    try {
      await predict(inputDataForApi);
    } catch (e) {
      console.error("Submit error", e)
    }
  };

  const chartConfig: ChartConfig = {
    price: {
      label: "Price (£)",
      color: "hsl(var(--primary))",
    },
  } satisfies ChartConfig;

  return (
    <div className="space-y-12">
      <PageHero
        title={PREDICTION_PAGE_HERO_TITLE}
        description={PREDICTION_PAGE_HERO_DESCRIPTION}
      />

      <Card className="max-w-3xl mx-auto shadow-xl animate-fadeIn" style={{animationDelay: '0.2s'}}>
        <CardHeader>
          <CardTitle className="font-headline text-2xl flex items-center"><Home className="mr-2 h-6 w-6 text-primary" />Enter Property Details</CardTitle>
          <CardDescription>Provide the following details for our AI to predict the price. Longitude and latitude can be auto-filled.</CardDescription>
        </CardHeader>
        <Form {...form}>
          <form onSubmit={form.handleSubmit(onSubmit)}>
            <CardContent className="space-y-6">
              <FormField
                control={form.control}
                name="fullAddress"
                render={({ field }) => (
                  <FormItem>
                    <FormLabel className="flex items-center">
                      <MapPin className="mr-2 h-4 w-4" />
                      Full Address
                      {isGeocoding && <Loader2 className="ml-2 h-4 w-4 animate-spin text-primary" />}
                    </FormLabel>
                    <FormControl>
                      <Input placeholder="e.g., 10 Downing Street, London" {...field} />
                    </FormControl>
                    <FormDescription>Coordinates will be auto-filled after you stop typing. If not in London, it will be appended.</FormDescription>
                    <FormMessage />
                  </FormItem>
                )}
              />
              <div className="grid grid-cols-1 md:grid-cols-3 gap-6">
                <FormField
                  control={form.control}
                  name="outcode"
                  render={({ field }) => (
                    <FormItem>
                      <div className="flex items-center justify-between mb-1">
                        <FormLabel className="flex items-center mb-0">
                            <MapPin className="mr-2 h-4 w-4" />Outcode
                        </FormLabel>
                        <Button
                            type="button"
                            variant="link"
                            size="sm"
                            className="p-0 h-auto text-xs text-muted-foreground hover:text-primary"
                            asChild
                        >
                            <a href="https://www.royalmail.com/find-a-postcode" target="_blank" rel="noopener noreferrer" aria-label="Find outcode on Royal Mail (opens in new tab)">
                                <HelpCircle className="mr-1 h-3 w-3" />
                                <span className="hidden sm:inline">Find Outcode</span>
                                <span className="sm:hidden">Help</span>
                            </a>
                        </Button>
                      </div>
                      <Select onValueChange={field.onChange} defaultValue={field.value}>
                        <FormControl>
                          <SelectTrigger>
                            <SelectValue placeholder="Select outcode" />
                          </SelectTrigger>
                        </FormControl>
                        <SelectContent>
                          {REGION_OPTIONS.map((oc) => (
                            <SelectItem key={oc} value={oc}>{oc}</SelectItem>
                          ))}
                        </SelectContent>
                      </Select>
                      <FormMessage />
                    </FormItem>
                  )}
                />
                 <FormField
                  control={form.control}
                  name="longitude"
                  render={({ field }) => (
                    <FormItem>
                      <FormLabel>Longitude</FormLabel>
                      <FormControl>
                        <Input type="number" step="any" placeholder="Auto-filled or e.g., -0.1278" {...field} onChange={e => field.onChange(e.target.value === '' ? undefined : parseFloat(e.target.value))} value={field.value ?? ""} />
                      </FormControl>
                      <FormMessage />
                    </FormItem>
                  )}
                />
                <FormField
                  control={form.control}
                  name="latitude"
                  render={({ field }) => (
                    <FormItem>
                      <FormLabel>Latitude</FormLabel>
                      <FormControl>
                        <Input type="number" step="any" placeholder="Auto-filled or e.g., 51.5074" {...field} onChange={e => field.onChange(e.target.value === '' ? undefined : parseFloat(e.target.value))} value={field.value ?? ""} />
                      </FormControl>
                      <FormMessage />
                    </FormItem>
                  )}
                />
              </div>

              <div className="grid grid-cols-1 md:grid-cols-3 gap-6">
                <FormField
                  control={form.control}
                  name="bedrooms"
                  render={({ field }) => (
                    <FormItem>
                      <FormLabel className="flex items-center"><Sofa className="mr-2 h-4 w-4" />Bedrooms</FormLabel>
                      <FormControl>
                        <Input type="number" placeholder="e.g., 3" {...field} />
                      </FormControl>
                      <FormMessage />
                    </FormItem>
                  )}
                />
                <FormField
                  control={form.control}
                  name="bathrooms"
                  render={({ field }) => (
                    <FormItem>
                      <FormLabel className="flex items-center"><Bath className="mr-2 h-4 w-4" />Bathrooms</FormLabel>
                      <FormControl>
                        <Input type="number" placeholder="e.g., 2" {...field} />
                      </FormControl>
                      <FormMessage />
                    </FormItem>
                  )}
                />
                <FormField
                  control={form.control}
                  name="livingRooms" 
                  render={({ field }) => (
                    <FormItem>
                      <FormLabel className="flex items-center"><Tv2 className="mr-2 h-4 w-4" />Living Rooms</FormLabel> 
                      <FormControl>
                        <Input type="number" placeholder="e.g., 1" {...field} />
                      </FormControl>
                      <FormMessage />
                    </FormItem>
                  )}
                />
              </div>
              <div className="grid grid-cols-1 md:grid-cols-2 gap-6">
                <FormField
                    control={form.control}
                    name="floorAreaSqM" 
                    render={({ field }) => (
                        <FormItem>
                        <FormLabel className="flex items-center"><Home className="mr-2 h-4 w-4" />Floor Area (sqm)</FormLabel> 
                        <FormControl>
                            <Input type="number" placeholder="e.g., 70" {...field} onChange={e => field.onChange(e.target.value === '' ? undefined : parseFloat(e.target.value))} value={field.value ?? ""} />
                        </FormControl>
                        <FormMessage />
                        </FormItem>
                    )}
                 />
                <FormField
                  control={form.control}
                  name="propertyType"
                  render={({ field }) => (
                    <FormItem>
                      <FormLabel className="flex items-center"><Building2 className="mr-2 h-4 w-4" />Property Type</FormLabel>
                      <Select onValueChange={field.onChange} defaultValue={field.value}>
                        <FormControl>
                          <SelectTrigger>
                            <SelectValue placeholder="Select property type" />
                          </SelectTrigger>
                        </FormControl>
                        <SelectContent>
                          {PROPERTY_TYPE_OPTIONS.map((pt) => (
                            <SelectItem key={pt} value={pt}>{pt}</SelectItem>
                          ))}
                        </SelectContent>
                      </Select>
                      <FormMessage />
                    </FormItem>
                  )}
                />
              </div>
              <div className="grid grid-cols-1 md:grid-cols-3 gap-6">
                 <FormField
                  control={form.control}
                  name="tenure"
                  render={({ field }) => (
                    <FormItem>
                      <FormLabel className="flex items-center"><FileText className="mr-2 h-4 w-4" />Tenure</FormLabel>
                      <Select onValueChange={field.onChange} defaultValue={field.value}>
                        <FormControl>
                          <SelectTrigger>
                            <SelectValue placeholder="Select tenure" />
                          </SelectTrigger>
                        </FormControl>
                        <SelectContent>
                          {TENURE_OPTIONS.map((t) => (
                            <SelectItem key={t} value={t}>{t}</SelectItem>
                          ))}
                        </SelectContent>
                      </Select>
                      <FormMessage />
                    </FormItem>
                  )}
                />
                <FormField
                  control={form.control}
                  name="currentEnergyRating"
                  render={({ field }) => (
                    <FormItem>
                      <FormLabel className="flex items-center"><Zap className="mr-2 h-4 w-4" />Current Energy Rating</FormLabel>
                      <Select onValueChange={field.onChange} defaultValue={field.value}>
                        <FormControl>
                          <SelectTrigger>
                            <SelectValue placeholder="Select energy rating" />
                          </SelectTrigger>
                        </FormControl>
                        <SelectContent>
                          {ENERGY_RATING_OPTIONS.map((rating) => (
                            <SelectItem key={rating} value={rating}>Rating {rating}</SelectItem>
                          ))}
                        </SelectContent>
                      </Select>
                      <FormMessage />
                    </FormItem>
                  )}
                />
                 <FormField
                  control={form.control}
                  name="monthOfSale"
                  render={({ field }) => (
                    <FormItem>
                      <FormLabel className="flex items-center"><CalendarDays className="mr-2 h-4 w-4" />Month of Sale (YYYY-MM)</FormLabel>
                      <FormControl>
                        <Input placeholder="e.g., 2024-07" {...field} />
                      </FormControl>
                      <FormDescription>{PREDICTION_MONTH_OF_SALE_FORMAT_DESC}</FormDescription>
                      <FormMessage />
                    </FormItem>
                  )}
                />
              </div>
            </CardContent>
            <CardFooter>
              <Button type="submit" disabled={isPredicting || isGeocoding} className="w-full">
                {isPredicting ? <Loader2 className="mr-2 h-4 w-4 animate-spin" /> : <TrendingUp className="mr-2 h-4 w-4" />}
                {isPredicting ? 'Predicting...' : 'Predict Price'}
              </Button>
            </CardFooter>
          </form>
        </Form>
      </Card>

      {isPredicting && (
        <div className="flex justify-center items-center mt-12">
          <Loader2 className="h-10 w-10 animate-spin text-primary" />
          <p className="ml-3 text-lg">Fetching prediction...</p>
        </div>
      )}

      {predictionError && (
         <Card className="mt-12 shadow-xl animate-fadeIn bg-destructive/10 border-destructive/30" style={{animationDelay: '0.4s'}}>
            <CardHeader>
                <CardTitle className="font-headline text-xl text-destructive">Prediction Error</CardTitle>
            </CardHeader>
            <CardContent>
                <p>Could not load prediction results: {predictionError.message}</p>
            </CardContent>
         </Card>
      )}

      {predictionData && !isPredicting && (
        <Card className="mt-12 shadow-xl animate-fadeIn" style={{animationDelay: '0.4s'}}>
          <CardHeader>
            <CardTitle className="font-headline text-2xl flex items-center"><Coins className="mr-2 h-6 w-6 text-primary" />Prediction Results</CardTitle>
          </CardHeader>
          <CardContent className="space-y-6">
            <div className="grid grid-cols-1 md:grid-cols-2 gap-6 text-center">
              <div className="bg-muted p-6 rounded-lg">
                <p className="text-sm text-muted-foreground font-medium">Predicted Price</p>
                <p className="text-3xl font-bold text-primary">
                  £{predictionData.price.toLocaleString()}
                </p>
              </div>
              <div className="bg-muted p-6 rounded-lg">
                <p className="text-sm text-muted-foreground font-medium">Trend (12 months)</p>
                <p className="text-3xl font-bold capitalize" style={{ color: predictionData.isStable ? 'hsl(var(--chart-5))' : 'hsl(var(--chart-1))' }}>
                  {predictionData.isStable ? 'Stable' : 'Dynamic'}
                </p>
              </div>
            </div>
            
            <div>
              <h3 className="text-xl font-headline mb-4 flex items-center"><LineChartIcon className="mr-2 h-5 w-5 text-primary" />12-Month Price Forecast</h3>
              {predictionData.priceHistoryChartData && predictionData.priceHistoryChartData.length > 0 ? (
                <ChartContainer config={chartConfig} className="h-[300px] w-full">
                  <ResponsiveContainer width="100%" height="100%">
                    <LineChart data={predictionData.priceHistoryChartData} margin={{ top: 5, right: 30, left: 20, bottom: 5 }}>
                      <CartesianGrid strokeDasharray="3 3" stroke="hsl(var(--border))" />
                      <XAxis dataKey="month" stroke="hsl(var(--muted-foreground))" tick={{ fontSize: 12 }} />
                      <YAxis 
                        stroke="hsl(var(--muted-foreground))" 
                        tickFormatter={(value) => `£${(value / 1000).toFixed(0)}k`} 
                        tick={{ fontSize: 12 }} 
                        domain={['dataMin - 10000', 'dataMax + 10000']}
                      />
                      <RechartsTooltip
                        content={<ChartTooltipContent indicator="line" />}
                        cursor={{ stroke: "hsl(var(--accent))", strokeWidth: 2, strokeDasharray: "3 3" }}
                      />
                      <Line type="monotone" dataKey="price" stroke="hsl(var(--primary))" strokeWidth={2} dot={{ r: 4, fill: "hsl(var(--primary))" }} activeDot={{ r: 6, stroke: "hsl(var(--background))", strokeWidth: 2 }} />
                    </LineChart>
                  </ResponsiveContainer>
                </ChartContainer>
              ) : (
                <p className="text-muted-foreground">No chart data available.</p>
              )}
            </div>
          </CardContent>
        </Card>
      )}
    </div>
  );
}
