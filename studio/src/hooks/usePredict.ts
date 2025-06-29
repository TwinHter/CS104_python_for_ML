
"use client";

import { useMutation } from '@tanstack/react-query';
import axios from 'axios';
import type { PredictionInput, PredictionOutput } from '@/ai/flows/price-prediction';
import { useToast } from './use-toast';

const fastapiBaseUrl = process.env.NEXT_PUBLIC_FASTAPI_URL || 'http://localhost:8000';

export function usePredict() {
  const { toast } = useToast();

  const mutation = useMutation<PredictionOutput, Error, PredictionInput>({
    mutationFn: async (inputData: PredictionInput) => {
      // Use axios to POST to the FastAPI backend endpoint
      const response = await axios.post<PredictionOutput>(`${fastapiBaseUrl}/model`, inputData);
      return response.data;
    },
    onSuccess: (data, variables) => {
      toast({
        title: "Prediction Generated",
        description: `The price prediction for the property has been calculated.`,
      });
    },
    onError: (error: any) => { 
      console.error('Prediction failed:', error);
      let errorMessage = "Could not retrieve the price prediction. Please check your input and try again.";
      if (error.response) {
        // The request was made and the server responded with a status code
        // that falls out of the range of 2xx
        errorMessage = error.response.data?.detail || error.response.data?.message || error.message;
      } else if (error.request) {
        // The request was made but no response was received
        errorMessage = "No response from prediction server. Is the FastAPI backend running and accessible?";
      } else {
        // Something happened in setting up the request that triggered an Error
        errorMessage = error.message;
      }
      toast({
        title: "Prediction Failed",
        description: errorMessage,
        variant: "destructive",
      });
    },
  });

  return {
    predict: mutation.mutateAsync,
    isPredicting: mutation.isPending,
    predictionData: mutation.data,
    predictionError: mutation.error,
    resetPrediction: mutation.reset,
  };
}

