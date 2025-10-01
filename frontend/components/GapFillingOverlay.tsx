'use client';

import React, { useState, useCallback, useEffect } from 'react';

interface GapFillingOverlayProps {
  isActive: boolean;
  onGapFilled: (document: any) => void;
  onClose: () => void;
  gapFillingCoordinates?: { x: number; y: number } | null;
}

interface GapAnalysisResult {
  gap_coordinates: { x: number; y: number };
  analysis_radius: number;
  nearby_documents: Array<{
    document_id: string;
    coordinates: number[];
    title: string;
    categories: string[];
    distance: number;
  }>;
  context_analysis: {
    primary_topics: string[];
    secondary_topics: string[];
    semantic_cluster: string;
    topic_diversity: number;
    document_count: number;
  };
  gap_score: number;
  is_valid_gap: boolean;
  analyzed_at: string;
}

interface GeneratedDocument {
  document_id: string;
  title: string;
  content: string;
  coordinates: number[];
  source: string;
  user_id: string;
  created_at: string;
  semantic_cluster: string;
}

const GapFillingOverlay: React.FC<GapFillingOverlayProps> = ({
  isActive,
  onGapFilled,
  onClose,
  gapFillingCoordinates
}) => {
  const [isAnalyzing, setIsAnalyzing] = useState(false);
  const [isGenerating, setIsGenerating] = useState(false);
  const [analysisResult, setAnalysisResult] = useState<GapAnalysisResult | null>(null);
  const [generatedDocument, setGeneratedDocument] = useState<GeneratedDocument | null>(null);
  const [error, setError] = useState<string | null>(null);
  const [targetCoordinates, setTargetCoordinates] = useState<{ x: number; y: number } | null>(null);
  const [showModal, setShowModal] = useState(false);
  const [forceGapFilling, setForceGapFilling] = useState(false);

  // Reset state when gap filling mode is activated
  useEffect(() => {
    if (isActive) {
      setShowModal(false);
      setAnalysisResult(null);
      setGeneratedDocument(null);
      setError(null);
      setTargetCoordinates(null);
    }
  }, [isActive]);

  // Watch for gap filling coordinates changes
  useEffect(() => {
    if (isActive && gapFillingCoordinates) {
      handleMapClick(gapFillingCoordinates);
    }
  }, [isActive, gapFillingCoordinates]);

  const handleMapClick = useCallback(async (coordinates: { x: number; y: number }) => {
    if (!isActive) return;

    setTargetCoordinates(coordinates);
    setError(null);
    setAnalysisResult(null);
    setGeneratedDocument(null);
    setShowModal(true);
    setIsGenerating(true);

    try {
      // Use the true Vec2Text API with timeout
      const controller = new AbortController();
      const timeoutId = setTimeout(() => controller.abort(), 15000); // 15 second timeout for gap filling
      
      const response = await fetch(
        `http://localhost:8001/api/true-vec2text/generate`,
        {
          method: 'POST',
          headers: {
            'Content-Type': 'application/json',
          },
          body: JSON.stringify({
            x: coordinates.x,
            y: coordinates.y
          }),
          signal: controller.signal
        }
      );
      
      clearTimeout(timeoutId);
      
      if (!response.ok) {
        throw new Error(`Advanced gap filling failed: ${response.status}`);
      }
      
      const result = await response.json();
      
      // Convert the true Vec2Text API response to our document format
      const document: GeneratedDocument = {
        document_id: `true_vec2text_${Date.now()}`,
        title: result.title,
        content: result.generated_text,
        coordinates: result.predicted_coordinates,
        source: 'true_vec2text',
        user_id: 'user_123',
        created_at: new Date().toISOString(),
        semantic_cluster: 'true_vec2text_generated'
      };
      
      setGeneratedDocument(document);
      setIsGenerating(false);

    } catch (err) {
      setError(err instanceof Error ? err.message : 'Failed to generate text using advanced pipeline');
      setIsGenerating(false);
    }
  }, [isActive]);

  const handleConfirmGeneration = () => {
    if (generatedDocument) {
      onGapFilled(generatedDocument);
      setAnalysisResult(null);
      setGeneratedDocument(null);
      setTargetCoordinates(null);
      setShowModal(false);
    }
  };

  const handleCancel = () => {
    setAnalysisResult(null);
    setGeneratedDocument(null);
    setTargetCoordinates(null);
    setError(null);
    setIsAnalyzing(false);
    setIsGenerating(false);
    setShowModal(false);
  };

  const handleCloseModal = () => {
    setShowModal(false);
  };

  if (!isActive) return null;

  return (
    <>
      {/* Status Panel - Always visible when gap filling mode is active */}
      <div className="fixed top-4 left-1/2 transform -translate-x-1/2 z-40">
        <div className="bg-orange-100 border border-orange-300 rounded-lg p-4 shadow-lg">
          <div className="flex items-center space-x-3">
            <div className="w-3 h-3 bg-orange-500 rounded-full animate-pulse"></div>
            <span className="text-orange-800 font-medium">Gap Filling Mode Active</span>
            <span className="text-orange-600 text-sm">Click on empty areas to generate text</span>
            <button
              onClick={onClose}
              className="ml-2 text-orange-600 hover:text-orange-800 font-bold"
            >
              Exit
            </button>
          </div>
        </div>
      </div>

      {/* Modal - Only shown when analyzing/generating */}
      {showModal && (
        <div className="fixed inset-0 bg-black bg-opacity-50 flex items-center justify-center z-50">
          <div className="bg-white rounded-lg p-6 max-w-2xl max-h-[80vh] overflow-y-auto">
            <div className="flex justify-between items-center mb-4">
              <h2 className="text-xl font-bold text-gray-800">Gap Filling Analysis</h2>
              <button
                onClick={handleCloseModal}
                className="text-gray-500 hover:text-gray-700 text-2xl"
              >
                ×
              </button>
            </div>

        <div className="mb-4">
          <p className="text-gray-600 mb-2">
            Click anywhere on the map to analyze and fill empty areas with contextual text.
          </p>
          <p className="text-sm text-gray-500">
            The system will generate text that fits semantically and spatially in the selected area.
          </p>
          
          <div className="mt-3 p-3 bg-green-50 rounded-lg">
            <p className="text-sm text-green-800">
              <strong>True Vec2Text:</strong> Using custom neural network for direct embedding-to-text generation. Works anywhere on the map with coordinate-specific text generation.
            </p>
          </div>
        </div>

        {targetCoordinates && (
          <div className="mb-4 p-3 bg-blue-50 rounded-lg">
            <p className="text-sm text-blue-800">
              <strong>Target Coordinates:</strong> ({targetCoordinates.x.toFixed(2)}, {targetCoordinates.y.toFixed(2)})
            </p>
          </div>
        )}

        {isAnalyzing && (
          <div className="mb-4 p-4 bg-yellow-50 rounded-lg">
            <div className="flex items-center">
              <div className="animate-spin rounded-full h-4 w-4 border-b-2 border-yellow-600 mr-2"></div>
              <p className="text-yellow-800">Analyzing gap area...</p>
            </div>
          </div>
        )}

        {isGenerating && (
          <div className="mb-4 p-4 bg-blue-50 rounded-lg">
            <div className="flex items-center">
              <div className="animate-spin rounded-full h-4 w-4 border-b-2 border-blue-600 mr-2"></div>
              <p className="text-blue-800">Generating contextual text...</p>
            </div>
          </div>
        )}

        {error && (
          <div className="mb-4 p-4 bg-red-50 rounded-lg">
            <p className="text-red-800">{error}</p>
            <button
              onClick={handleCancel}
              className="mt-2 px-3 py-1 bg-red-100 text-red-700 rounded hover:bg-red-200"
            >
              Try Different Area
            </button>
          </div>
        )}

        {generatedDocument && (
          <div className="mb-4">
            <h3 className="font-semibold text-gray-800 mb-2">Generation Details</h3>
            <div className="grid grid-cols-2 gap-4 text-sm">
              <div>
                <p><strong>Method:</strong> True Vec2Text Neural Network</p>
                <p><strong>Target Coords:</strong> ({targetCoordinates?.x.toFixed(2) || 'N/A'}, {targetCoordinates?.y.toFixed(2) || 'N/A'})</p>
                <p><strong>Predicted Coords:</strong> ({generatedDocument.coordinates[0].toFixed(2)}, {generatedDocument.coordinates[1].toFixed(2)})</p>
              </div>
              <div>
                <p><strong>Generation:</strong> Neural Network</p>
                <p><strong>Status:</strong> Ready for Map</p>
                <p><strong>Source:</strong> True Vec2Text</p>
              </div>
            </div>
          </div>
        )}

        {generatedDocument && (
          <div className="mb-4">
            <h3 className="font-semibold text-gray-800 mb-2">Generated Document</h3>
            <div className="border rounded-lg p-4">
              <h4 className="font-medium text-lg mb-2">{generatedDocument.title}</h4>
              <p className="text-gray-700 mb-3">{generatedDocument.content}</p>
              <div className="text-sm text-gray-500">
                <p><strong>Predicted Coordinates:</strong> ({generatedDocument.coordinates[0].toFixed(2)}, {generatedDocument.coordinates[1].toFixed(2)})</p>
                <p><strong>Semantic Cluster:</strong> {generatedDocument.semantic_cluster}</p>
                <p><strong>Document ID:</strong> {generatedDocument.document_id}</p>
              </div>
            </div>
          </div>
        )}

        <div className="flex justify-end space-x-3">
          {generatedDocument && (
            <>
              <button
                onClick={handleCancel}
                className="px-4 py-2 bg-gray-200 text-gray-700 rounded hover:bg-gray-300"
              >
                Cancel
              </button>
              <button
                onClick={handleConfirmGeneration}
                className="px-4 py-2 bg-green-600 text-white rounded hover:bg-green-700"
              >
                Add to Map
              </button>
            </>
          )}
          {!generatedDocument && !isAnalyzing && !isGenerating && (
            <button
              onClick={handleCancel}
              className="px-4 py-2 bg-gray-200 text-gray-700 rounded hover:bg-gray-300"
            >
              Close
            </button>
          )}
            </div>
          </div>
        </div>
      )}
    </>
  );
};

export default GapFillingOverlay;
