
"use client";

import { useEffect, useState } from 'react';
import Image from 'next/image';
import Link from 'next/link';
import { useParams, notFound } from 'next/navigation';
import { useQuery } from '@tanstack/react-query';
import { getPropertyDetails } from '@/services/api'; 
import PageHero from '@/components/shared/PageHero';
import { Card, CardContent, CardHeader, CardTitle } from '@/components/ui/card';
import { Button } from '@/components/ui/button';
import { Loader2, Mail, Phone, MapPin, Home, ArrowLeft, UserCircle, Briefcase, TrendingUp, DollarSign, AlertTriangle } from 'lucide-react';
import { Avatar, AvatarFallback, AvatarImage } from "@/components/ui/avatar";
import { Separator } from "@/components/ui/separator";
import { useToast } from '@/hooks/use-toast';
import { usePredict } from '@/hooks/usePredict';
import type { PredictionInput, PredictionOutput } from '@/ai/flows/price-prediction';
import { CONTACT_PAGE_TITLE_TEMPLATE, PLACEHOLDER_HINTS, DEFAULT_SALESMAN_INFO, REGION_OPTIONS } from '@/lib/constants';
import type { Property } from '@/types';

export default function ContactPropertyPage() {
  const params = useParams();
  const propertyId = params.propertyId as string;
  const { toast } = useToast();

  const { data: property, isLoading: isLoadingProperty, error: propertyError } = useQuery<Property | null>({
    queryKey: ['propertyDetails', propertyId],
    queryFn: () => getPropertyDetails(propertyId), 
    enabled: !!propertyId,
  });

  const contactPerson = {
    name: property?.uploaderName || DEFAULT_SALESMAN_INFO.name,
    email: property?.uploaderEmail || DEFAULT_SALESMAN_INFO.email,
    phone: property?.uploaderPhone || DEFAULT_SALESMAN_INFO.phone,
    imageUrl: DEFAULT_SALESMAN_INFO.imageUrl, 
    dataAiHint: property?.uploaderName ? PLACEHOLDER_HINTS.salesmanPortrait : DEFAULT_SALESMAN_INFO.dataAiHint,
  };

  const { predict } = usePredict(); 
  const [predictionResult, setPredictionResult] = useState<PredictionOutput | null>(null);
  const [isPredictingPropertyPrice, setIsPredictingPropertyPrice] = useState(false);
  const [predictionErrorText, setPredictionErrorText] = useState<string | null>(null);

  useEffect(() => {
    if (propertyError) {
      toast({ title: "Error", description: "Could not load property details.", variant: "destructive" });
    }
  }, [propertyError, toast]);

  const handlePredictPropertyPrice = async () => {
    if (!property) return;

    setIsPredictingPropertyPrice(true);
    setPredictionResult(null);
    setPredictionErrorText(null);

    const inputData: PredictionInput = {
      fullAddress: property.fullAddress,
      latitude: property.latitude,
      longitude: property.longitude,
      bedrooms: property.bedrooms,
      bathrooms: property.bathrooms,
      livingRooms: property.livingRooms,
      sale_month: property.sale_month || (new Date().getMonth() + 1),
      sale_year: property.sale_year || new Date().getFullYear(),
      tenure: property.tenure,
      currentEnergyRating: property.currentEnergyRating,
      floorAreaSqM: property.floorAreaSqM || 70,
      outcode: property.outcode,
      propertyType: property.propertyType,
    };

    try {
      const result = await predict(inputData); 
      setPredictionResult(result);
    } catch (e: any) {
      console.error("Prediction error on contact page:", e);
      setPredictionErrorText(e.message || "Failed to get price prediction.");
      toast({
        title: "Prediction Failed",
        description: e.message || "Could not retrieve prediction for this property.",
        variant: "destructive",
      });
    } finally {
      setIsPredictingPropertyPrice(false);
    }
  };

  const renderPriceComparison = () => {
    if (!predictionResult || !property) return null;

    const listedPrice = property.price;
    const predictedPriceVal = predictionResult.price;
    const diff = predictedPriceVal - listedPrice;
    const percentageDiff = (diff / listedPrice) * 100;

    let comparisonText;
    if (Math.abs(percentageDiff) < 5) {
      comparisonText = "This is closely aligned with the listed price.";
    } else if (percentageDiff > 0) {
      comparisonText = `This is ${percentageDiff.toFixed(1)}% higher than the listed price.`;
    } else {
      comparisonText = `This is ${Math.abs(percentageDiff).toFixed(1)}% lower than the listed price.`;
    }

    return (
      <div className="mt-3 space-y-1">
        <p className="text-sm font-semibold">AI Predicted Price: <span className="text-primary font-bold text-base">£{predictedPriceVal.toLocaleString()}</span></p>
        <p className="text-xs text-muted-foreground">{comparisonText}</p>
      </div>
    );
  };

  if (isLoadingProperty) {
    return (
      <div className="flex justify-center items-center min-h-[calc(100vh-200px)]">
        <Loader2 className="h-16 w-16 animate-spin text-primary" />
      </div>
    );
  }

  if (!property) {
    notFound();
    return null;
  }

  return (
    <div className="space-y-12">
      <PageHero
        title={CONTACT_PAGE_TITLE_TEMPLATE(property.name)}
        description={<span className="flex items-center justify-center"><MapPin size={16} className="mr-1.5" /> {property.fullAddress}</span>}
      />

      <div className="max-w-4xl mx-auto">
        <Button variant="outline" asChild className="mb-8 animate-fadeIn" style={{animationDelay: '0.1s'}}>
          <Link href="/recommendations">
            <ArrowLeft className="mr-2 h-4 w-4" /> Back to Recommendations
          </Link>
        </Button>

        <div className="grid md:grid-cols-2 gap-8">
          <Card className="shadow-xl animate-fadeIn" style={{animationDelay: '0.2s'}}>
            <CardHeader>
              <CardTitle className="font-headline text-2xl flex items-center"><Home className="mr-2 h-6 w-6 text-primary"/>Property Overview</CardTitle>
            </CardHeader>
            <CardContent className="space-y-4">
              <Image
                src={property.image}
                alt={property.name}
                width={600}
                height={400}
                className="rounded-lg object-cover w-full aspect-video shadow-md"
                data-ai-hint={property.dataAiHint || PLACEHOLDER_HINTS.propertyGeneric}
              />
              <div>
                <h3 className="font-semibold text-lg">{property.name}</h3>
                <p className="text-muted-foreground text-sm">{property.fullAddress}</p>
              </div>
              <p className="text-2xl font-bold text-primary flex items-center">
                <DollarSign size={24} className="mr-1" />
                {property.price.toLocaleString()}
                <span className="text-sm font-normal text-muted-foreground ml-1">(Listed Price)</span>
              </p>
              <ul className="text-sm space-y-1 text-foreground/80">
                <li><strong>Type:</strong> {property.propertyType}</li>
                <li><strong>Bedrooms:</strong> {property.bedrooms}</li>
                <li><strong>Bathrooms:</strong> {property.bathrooms}</li>
                <li><strong>Living Rooms:</strong> {property.livingRooms}</li>
                {property.floorAreaSqM && <li><strong>Area:</strong> {property.floorAreaSqM} m²</li>}
                <li><strong>Tenure:</strong> {property.tenure}</li>
                <li><strong>Energy Rating:</strong> {property.currentEnergyRating}</li>
                <li><strong>Outcode:</strong> {property.outcode}</li>
                {property.longitude && property.latitude && (
                  <li><strong>Coordinates:</strong> {property.latitude.toFixed(4)}, {property.longitude.toFixed(4)}</li>
                )}
                <li><strong>Listed:</strong> {new Date(property.sale_year, property.sale_month -1).toLocaleDateString('en-GB', { month: 'long', year: 'numeric' })}</li>
              </ul>

              <Separator className="my-4" />

              <div className="space-y-2">
                <h4 className="font-semibold text-md">Price Analysis</h4>
                {!predictionResult && !isPredictingPropertyPrice && !predictionErrorText && (
                  <Button onClick={handlePredictPropertyPrice} variant="secondary" className="w-full">
                    <TrendingUp className="mr-2 h-4 w-4" /> Predict Current Market Price
                  </Button>
                )}
                {isPredictingPropertyPrice && (
                  <div className="flex items-center justify-center p-4">
                    <Loader2 className="mr-2 h-5 w-5 animate-spin" />
                    <span>Calculating AI prediction...</span>
                  </div>
                )}
                {predictionErrorText && !isPredictingPropertyPrice && (
                  <div className="text-destructive text-sm p-3 bg-destructive/10 rounded-md flex items-center">
                    <AlertTriangle size={16} className="mr-2"/> {predictionErrorText}
                  </div>
                )}
                {predictionResult && !isPredictingPropertyPrice && (
                   <div className="p-3 bg-muted/50 rounded-md border">
                    {renderPriceComparison()}
                    <Button onClick={handlePredictPropertyPrice} variant="outline" size="sm" className="w-full mt-3">
                      <TrendingUp className="mr-2 h-4 w-4" /> Re-predict
                    </Button>
                  </div>
                )}
              </div>
            </CardContent>
          </Card>

          <Card className="shadow-xl animate-fadeIn" style={{animationDelay: '0.4s'}}>
            <CardHeader>
              <CardTitle className="font-headline text-2xl flex items-center"><UserCircle className="mr-2 h-6 w-6 text-primary"/>Contact Person</CardTitle>
            </CardHeader>
            <CardContent className="space-y-4">
              <div className="flex items-center space-x-4">
                <Avatar className="h-20 w-20">
                  <AvatarImage src={contactPerson.imageUrl} alt={contactPerson.name} data-ai-hint={contactPerson.dataAiHint} />
                  <AvatarFallback>{contactPerson.name.substring(0,2).toUpperCase()}</AvatarFallback>
                </Avatar>
                <div>
                  <h3 className="text-xl font-semibold">{contactPerson.name}</h3>
                </div>
              </div>
              <div className="space-y-2 pt-2 border-t">
                <p className="flex items-center">
                  <Mail size={16} className="mr-2 text-primary" />
                  <a href={`mailto:${contactPerson.email}`} className="hover:underline text-accent-foreground hover:text-primary">
                    {contactPerson.email}
                  </a>
                </p>
                {contactPerson.phone && (
                  <p className="flex items-center">
                    <Phone size={16} className="mr-2 text-primary" />
                    <a href={`tel:${contactPerson.phone.replace(/\s/g, '')}`} className="hover:underline text-accent-foreground hover:text-primary">
                      {contactPerson.phone}
                    </a>
                  </p>
                )}
              </div>
              <Button className="w-full mt-4" asChild>
                 <a href={`mailto:${contactPerson.email}?subject=Enquiry%20about%20Property%20ID:%20${property.id}%20-%20${encodeURIComponent(property.name)}&body=Dear%20${contactPerson.name.split(' ')[0]},%0D%0A%0D%0AI%20am%20interested%20in%20the%20property%20'${encodeURIComponent(property.name)}'%20(ID:%20${property.id})%20located%20at%20${encodeURIComponent(property.fullAddress)}.%0D%0A%0D%0APlease%20provide%20me%20with%20more%20information.%0D%0A%0D%0AThank%20you.`}>
                  <Mail className="mr-2 h-4 w-4" /> Email {contactPerson.name.split(' ')[0]}
                </a>
              </Button>
            </CardContent>
          </Card>
        </div>
      </div>
    </div>
  );
}
