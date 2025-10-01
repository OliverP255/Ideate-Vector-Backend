'use client';

import React, { useState, useCallback, useEffect } from 'react';
import DeckGL from '@deck.gl/react';
import { ScatterplotLayer, TextLayer, PathLayer } from '@deck.gl/layers';
import { Map } from 'react-map-gl/mapbox';
import 'mapbox-gl/dist/mapbox-gl.css';
import useLassoSelectionMode from './LassoSelectionMode';

// Helper function to convert HSV to RGB
function hsvToRgb(h: number, s: number, v: number): [number, number, number] {
  const c = v * s;
  const x = c * (1 - Math.abs(((h / 60) % 2) - 1));
  const m = v - c;
  
  let r = 0, g = 0, b = 0;
  
  if (h >= 0 && h < 60) {
    r = c; g = x; b = 0;
  } else if (h >= 60 && h < 120) {
    r = x; g = c; b = 0;
  } else if (h >= 120 && h < 180) {
    r = 0; g = c; b = x;
  } else if (h >= 180 && h < 240) {
    r = 0; g = x; b = c;
  } else if (h >= 240 && h < 300) {
    r = x; g = 0; b = c;
  } else if (h >= 300 && h < 360) {
    r = c; g = 0; b = x;
  }
  
  return [Math.round((r + m) * 255), Math.round((g + m) * 255), Math.round((b + m) * 255)];
}

// Helper function to generate circle path
function generateCirclePath(center: [number, number], radius: number, segments: number = 64): [number, number][] {
  const [centerX, centerY] = center;
  const path: [number, number][] = [];
  
  for (let i = 0; i <= segments; i++) {
    const angle = (i / segments) * 2 * Math.PI;
    const x = centerX + radius * Math.cos(angle);
    const y = centerY + radius * Math.sin(angle);
    path.push([x, y]);
  }
  
  return path;
}

// Types
interface Document {
  document_id: string;
  coordinates: [number, number];
  title: string;
  similarity_score?: number;
  spatial_distance?: number;
  rerank_method?: string;
}

interface ClickResult {
  click_coordinates: { x: number; y: number };
  radius: number;
  spatial_candidates: number;
  documents: Document[];
  user_id?: string;
  query_text?: string;
  processed_at: string;
  spatial_area?: {
    center: [number, number];
    radius: number;
  };
}

interface LassoSelection {
  id: string;
  path: [number, number][];
  documents: Document[];
  center: [number, number];
  color: [number, number, number, number];
  distanceToCenter: { [documentId: string]: number };
}

interface KnowledgeMapProps {
  width?: number;
  height?: number;
  initialViewState?: any;
  onDocumentClick?: (document: Document) => void;
  onMapClick?: (clickResult: ClickResult) => void;
  onGapFillingMapClick?: (coordinates: { x: number; y: number }) => void;
  onLassoComplete?: (selection: LassoSelection) => void;
  userOverlay?: {
    userId: string;
    readDocuments: string[];
    convexHull?: any;
  };
  searchResult?: ClickResult | null;
  uploadedDocuments?: Document[];
  gapFilledDocuments?: Document[];
  isLassoSelectionMode?: boolean;
}

const KnowledgeMap: React.FC<KnowledgeMapProps> = ({
  width = '100%',
  height = '100vh',
  initialViewState = {
    longitude: 0,
    latitude: 0,
    zoom: 8, // Much more zoomed in to see dots better
    pitch: 0,
    bearing: 0
  },
  onDocumentClick,
  onMapClick,
  onGapFillingMapClick,
  onLassoComplete,
  userOverlay,
  searchResult,
  uploadedDocuments = [],
  gapFilledDocuments = [],
  isLassoSelectionMode = false
}) => {
  const [viewState, setViewState] = useState(initialViewState);
  const [documents, setDocuments] = useState<Document[]>([]);
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState<string | null>(null);
  const [clickResult, setClickResult] = useState<ClickResult | null>(null);
  const [lassoSelections, setLassoSelections] = useState<LassoSelection[]>([]);

  // Load initial documents
  useEffect(() => {
    loadDocuments();
  }, []);

  const loadDocuments = async () => {
    try {
      setLoading(true);
      setError(null);
      
      // Add timeout to prevent hanging
      const controller = new AbortController();
      const timeoutId = setTimeout(() => controller.abort(), 10000); // 10 second timeout
      
      const response = await fetch('http://localhost:8001/api/mapping/coordinates', {
        signal: controller.signal,
        headers: {
          'Accept': 'application/json',
          'Content-Type': 'application/json'
        }
      });
      
      clearTimeout(timeoutId);
      
      if (!response.ok) {
        throw new Error(`Failed to load documents: ${response.status} ${response.statusText}`);
      }
      
      const data = await response.json();
      
      // Check if data and coordinates exist
      if (!data || !data.coordinates || !Array.isArray(data.coordinates)) {
        throw new Error('Invalid data format received from server');
      }
      
      // Transform coordinates to document format
      const docs: Document[] = data.coordinates.map((coord: any) => ({
        document_id: coord.document_id || 'unknown',
        coordinates: coord.coordinates || [0, 0],
        title: coord.title || `Document ${coord.document_id || 'unknown'}`,
        rerank_method: coord.source || 'unknown'
      }));
      
      setDocuments(docs);
      console.log(`✅ Loaded ${docs.length} documents successfully`);
    } catch (err) {
      if (err instanceof Error && err.name === 'AbortError') {
        setError('Request timed out - please check if the backend server is running');
      } else {
        setError(err instanceof Error ? err.message : 'Failed to load documents');
      }
      console.error('Error loading documents:', err);
    } finally {
      setLoading(false);
    }
  };

  const handleMapClick = useCallback(async (info: any) => {
    if (!info.coordinate) return;

    const [x, y] = info.coordinate;
    
    // If gap filling mode is active, handle it differently
    if (onGapFillingMapClick) {
      onGapFillingMapClick({ x, y });
      return;
    }
    
    try {
      setLoading(true);
      setError(null);
      
      // Add timeout to prevent hanging
      const controller = new AbortController();
      const timeoutId = setTimeout(() => controller.abort(), 15000); // 15 second timeout for click
      
      const response = await fetch(
        `http://localhost:8001/api/click?x=${x}&y=${y}&radius=0.5&limit=5`,
        {
          signal: controller.signal,
          headers: {
            'Accept': 'application/json',
            'Content-Type': 'application/json'
          }
        }
      );
      
      clearTimeout(timeoutId);
      
      if (!response.ok) {
        throw new Error(`Click failed: ${response.status} ${response.statusText}`);
      }
      
      const result: ClickResult = await response.json();
      setClickResult(result);
      
      if (onMapClick) {
        onMapClick(result);
      }
      
      console.log(`✅ Click handled: found ${result.documents?.length || 0} documents`);
    } catch (err) {
      if (err instanceof Error && err.name === 'AbortError') {
        setError('Click request timed out - please try again');
      } else {
        setError(err instanceof Error ? err.message : 'Click failed');
      }
      console.error('Error handling click:', err);
    } finally {
      setLoading(false);
    }
  }, [onMapClick, onGapFillingMapClick]);

  const handleDocumentClick = useCallback((info: any) => {
    if (info.object) {
      const document = info.object as Document;
      if (onDocumentClick) {
        onDocumentClick(document);
      }
    }
  }, [onDocumentClick]);

  // Lasso selection handlers
  const handleLassoComplete = useCallback((selection: LassoSelection) => {
    if (onLassoComplete) {
      onLassoComplete(selection);
    }
  }, [onLassoComplete]);

  const handleLassoUpdate = useCallback((updatedSelections: LassoSelection[]) => {
    setLassoSelections(updatedSelections);
  }, []);

  // Debug logging for search result
  useEffect(() => {
    if (searchResult && searchResult.spatial_area) {
      console.log('🎯 Rendering spatial area circle:', {
        center: searchResult.spatial_area.center,
        radius: searchResult.spatial_area.radius,
        documents: searchResult.documents?.length || 0
      });
    }
  }, [searchResult]);

  // Initialize lasso selection mode
  const lassoSelectionMode = useLassoSelectionMode({
    documents,
    isActive: isLassoSelectionMode,
    onSelectionComplete: handleLassoComplete,
    onSelectionUpdate: handleLassoUpdate,
    onExit: () => {} // We'll handle exit in the parent component
  });

  // Handle mouse up for lasso selection
  useEffect(() => {
    const handleMouseUp = () => {
      if (isLassoSelectionMode && lassoSelectionMode.handlers.onMouseUp) {
        lassoSelectionMode.handlers.onMouseUp();
      }
    };

    if (isLassoSelectionMode) {
      document.addEventListener('mouseup', handleMouseUp);
      return () => document.removeEventListener('mouseup', handleMouseUp);
    }
  }, [isLassoSelectionMode, lassoSelectionMode.handlers]);

  // Create layers with safety checks
  const layers = [
    // Main document points
    new ScatterplotLayer({
      id: 'documents',
      data: documents || [],
      getPosition: (d: Document) => d.coordinates || [0, 0],
      getRadius: 300, // Smaller radius in meters
      radiusMinPixels: 6, // Minimum 6 pixels regardless of zoom
      radiusMaxPixels: 60, // Maximum 60 pixels regardless of zoom
      getFillColor: (d: Document) => {
        // UMAP-based coloring using document coordinates
        const coords = d.coordinates || [0, 0];
        const x = coords[0];
        const y = coords[1];
        
        // Normalize coordinates to 0-1 range (assuming coordinates range from -20 to 20)
        const normalizedX = Math.max(0, Math.min(1, (x + 20) / 40));
        const normalizedY = Math.max(0, Math.min(1, (y + 20) / 40));
        
        // Create vibrant UMAP-based colors
        const hue = (normalizedX * 360) % 360;
        const saturation = 0.9 + (normalizedY * 0.1); // 0.9 to 1.0 (highly saturated)
        const value = 0.9 + (normalizedY * 0.1); // 0.9 to 1.0 (very bright)
        
        // Convert HSV to RGB
        const rgb = hsvToRgb(hue, saturation, value);
        return [rgb[0], rgb[1], rgb[2], 255]; // No transparency for better visibility
      },
      getLineColor: [255, 255, 255, 150], // More visible white border
      lineWidthMinPixels: 2,
      pickable: true,
      onClick: handleDocumentClick,
      updateTriggers: {
        getFillColor: [documents]
      }
    }),

    // Spatial area circle (semantic search area) - filled black transparent circle
    ...(searchResult && searchResult.spatial_area ? [
      new ScatterplotLayer({
        id: 'spatial-area-circle',
        data: [{
          position: searchResult.spatial_area.center,
          radius: searchResult.spatial_area.radius
        }],
        getPosition: (d: any) => d.position,
        getRadius: (d: any) => d.radius,
        radiusMinPixels: 10,
        radiusMaxPixels: 1000,
        getFillColor: [0, 0, 0, 100], // Black with transparency
        getLineColor: [0, 0, 0, 0], // No border
        pickable: false
      })
    ] : []),

    // Removed document labels - showing only colored dots

    // Search result highlights (from semantic search)
    ...(searchResult && searchResult.documents ? [
      new ScatterplotLayer({
        id: 'search-highlights',
        data: searchResult.documents || [],
        getPosition: (d: Document) => d.coordinates || [0, 0],
        getRadius: 450, // Smaller highlight radius in meters
        radiusMinPixels: 9, // Minimum 9 pixels for highlights
        radiusMaxPixels: 90, // Maximum 90 pixels for highlights
        getFillColor: [255, 255, 0, 200], // Bright yellow highlight
        getLineColor: [255, 0, 0, 255], // Red border
        lineWidthMinPixels: 3,
        pickable: true,
        onClick: handleDocumentClick
      })
    ] : []),

    // Click result highlights (from map clicks)
    ...(clickResult && clickResult.documents ? [
      new ScatterplotLayer({
        id: 'click-highlights',
        data: clickResult.documents || [],
        getPosition: (d: Document) => d.coordinates || [0, 0],
        getRadius: 450, // Smaller highlight radius in meters
        radiusMinPixels: 9, // Minimum 9 pixels for highlights
        radiusMaxPixels: 90, // Maximum 90 pixels for highlights
        getFillColor: [255, 255, 0, 200], // Bright yellow highlight
        getLineColor: [255, 0, 0, 255], // Red border
        lineWidthMinPixels: 3,
        pickable: true,
        onClick: handleDocumentClick
      })
    ] : []),

    // Uploaded documents (session-only)
    ...(uploadedDocuments && uploadedDocuments.length > 0 ? [
      new ScatterplotLayer({
        id: 'uploaded-documents',
        data: uploadedDocuments || [],
        getPosition: (d: Document) => d.coordinates || [0, 0],
        getRadius: 400, // Slightly smaller than regular documents
        radiusMinPixels: 8, // Minimum 8 pixels
        radiusMaxPixels: 80, // Maximum 80 pixels
        getFillColor: [255, 0, 255, 255], // Bright magenta for uploaded docs
        getLineColor: [255, 255, 255, 255], // White border
        lineWidthMinPixels: 3,
        pickable: true,
        onClick: handleDocumentClick
      })
    ] : []),

    // Gap-filled documents (generated text)
    ...(gapFilledDocuments && gapFilledDocuments.length > 0 ? [
      new ScatterplotLayer({
        id: 'gap-filled-documents',
        data: gapFilledDocuments || [],
        getPosition: (d: Document) => d.coordinates || [0, 0],
        getRadius: 400, // Same size as uploaded documents
        radiusMinPixels: 8, // Minimum 8 pixels
        radiusMaxPixels: 80, // Maximum 80 pixels
        getFillColor: [255, 165, 0, 255], // Bright orange for gap-filled docs
        getLineColor: [255, 255, 255, 255], // White border
        lineWidthMinPixels: 3,
        pickable: true,
        onClick: handleDocumentClick
      })
    ] : []),

    // Lasso selection layers
    ...lassoSelectionMode.layers
  ];

  return (
    <div className="relative w-full h-full">
      <DeckGL
        initialViewState={viewState}
        controller={true}
        layers={layers}
        onClick={isLassoSelectionMode ? lassoSelectionMode.handlers.onMouseDown : handleMapClick}
        onHover={isLassoSelectionMode ? lassoSelectionMode.handlers.onMouseMove : undefined}
        onViewStateChange={({ viewState }) => setViewState(viewState)}
        width={width}
        height={height}
      >
        <Map
          mapStyle="mapbox://styles/mapbox/dark-v10"
          mapboxAccessToken={process.env.NEXT_PUBLIC_MAPBOX_TOKEN || 'pk.eyJ1IjoibWFwYm94IiwiYSI6ImNpejY4NXVycTA2emYycXBndHRqcmZ3N3gifQ.rJcFIG214AriISLbB6B5aw'}
        />
      </DeckGL>

      {/* Loading overlay */}
      {loading && (
        <div className="absolute top-4 left-4 bg-black bg-opacity-75 text-white px-4 py-2 rounded">
          Loading...
        </div>
      )}

      {/* Lasso selection mode indicator */}
      {isLassoSelectionMode && (
        <div className="absolute top-4 left-4 bg-purple-600 bg-opacity-90 text-white px-4 py-2 rounded">
          <div className="flex items-center gap-2">
            <div className="w-3 h-3 bg-white rounded-full animate-pulse"></div>
            <span className="font-semibold">Lasso Selection Mode</span>
          </div>
          <p className="text-sm mt-1">Click and drag to draw freeform selections around clusters</p>
          <p className="text-xs mt-1 opacity-75">Press ESC to exit</p>
        </div>
      )}

      {/* Error display */}
      {error && (
        <div className="absolute top-4 right-4 bg-red-600 text-white px-4 py-2 rounded">
          Error: {error}
        </div>
      )}

      {/* Spatial area info only - no duplicate document list */}
      {(searchResult || clickResult)?.spatial_area && (
        <div className="absolute bottom-4 left-4 bg-black bg-opacity-75 text-white p-4 rounded max-w-md">
          <h3 className="font-bold mb-2">Semantic Search Area</h3>
          <div className="p-2 bg-blue-900 rounded">
            <p className="text-sm">Center: ({(searchResult || clickResult)?.spatial_area?.center[0].toFixed(2)}, {(searchResult || clickResult)?.spatial_area?.center[1].toFixed(2)})</p>
            <p className="text-sm">Radius: {(searchResult || clickResult)?.spatial_area?.radius.toFixed(2)}</p>
            <p className="text-sm">Documents: {(searchResult || clickResult)?.documents?.length || 0}</p>
          </div>
        </div>
      )}

      {/* Lasso selection information */}
      {lassoSelections.length > 0 && (
        <div className="absolute bottom-4 right-4 bg-black bg-opacity-75 text-white p-4 rounded max-w-md">
          <h3 className="font-bold mb-2">Lasso Selections ({lassoSelections.length})</h3>
          <div className="space-y-2 max-h-60 overflow-y-auto">
            {lassoSelections.map((selection, index) => (
              <div key={selection.id} className="p-2 bg-gray-800 rounded">
                <div className="flex items-center gap-2 mb-1">
                  <div 
                    className="w-4 h-4 rounded-full border-2 border-white"
                    style={{ backgroundColor: `rgba(${selection.color[0]}, ${selection.color[1]}, ${selection.color[2]}, 0.8)` }}
                  ></div>
                  <span className="text-sm font-semibold">Selection {index + 1}</span>
                </div>
                <p className="text-xs">Center: ({selection.center[0].toFixed(2)}, {selection.center[1].toFixed(2)})</p>
                <p className="text-xs">Path points: {selection.path.length}</p>
                <p className="text-xs">Documents: {selection.documents.length}</p>
              </div>
            ))}
          </div>
        </div>
      )}

      {/* User overlay info */}
      {userOverlay && (
        <div className="absolute top-4 right-4 bg-blue-600 text-white p-4 rounded">
          <h3 className="font-bold mb-2">User Overlay</h3>
          <p>User ID: {userOverlay.userId || 'Unknown'}</p>
          <p>Read documents: {userOverlay.readDocuments?.length || 0}</p>
          {userOverlay.convexHull && <p>Convex hull: Available</p>}
        </div>
      )}
    </div>
  );
};

export default KnowledgeMap;