'use client';

import React, { useState, useCallback, useEffect } from 'react';
import { ScatterplotLayer, PathLayer } from '@deck.gl/layers';

// Types
interface Document {
  document_id: string;
  coordinates: [number, number];
  title: string;
  similarity_score?: number;
  spatial_distance?: number;
  rerank_method?: string;
}

interface LassoSelection {
  id: string;
  path: [number, number][];
  documents: Document[];
  center: [number, number];
  color: [number, number, number, number];
  distanceToCenter: { [documentId: string]: number };
}

interface LassoSelectionModeProps {
  documents: Document[];
  isActive: boolean;
  onSelectionComplete: (selection: LassoSelection) => void;
  onSelectionUpdate: (selections: LassoSelection[]) => void;
  onExit: () => void;
}

// Helper function to calculate distance between two points
function calculateDistance(point1: [number, number], point2: [number, number]): number {
  const dx = point1[0] - point2[0];
  const dy = point1[1] - point2[1];
  return Math.sqrt(dx * dx + dy * dy);
}

// Helper function to check if a point is inside a polygon using ray casting algorithm
function pointInPolygon(point: [number, number], polygon: [number, number][]): boolean {
  const [x, y] = point;
  let inside = false;
  
  for (let i = 0, j = polygon.length - 1; i < polygon.length; j = i++) {
    const [xi, yi] = polygon[i];
    const [xj, yj] = polygon[j];
    
    if (((yi > y) !== (yj > y)) && (x < (xj - xi) * (y - yi) / (yj - yi) + xi)) {
      inside = !inside;
    }
  }
  
  return inside;
}

// Helper function to calculate polygon centroid
function calculateCentroid(polygon: [number, number][]): [number, number] {
  let cx = 0;
  let cy = 0;
  
  for (const [x, y] of polygon) {
    cx += x;
    cy += y;
  }
  
  return [cx / polygon.length, cy / polygon.length];
}

// Helper function to find documents within a lasso selection
function findDocumentsInLasso(documents: Document[], lassoPath: [number, number][]): Document[] {
  return documents.filter(doc => pointInPolygon(doc.coordinates, lassoPath));
}

// Helper function to calculate distance to center for each document
function calculateDistancesToCenter(documents: Document[], center: [number, number]): { [documentId: string]: number } {
  const distances: { [documentId: string]: number } = {};
  
  for (const doc of documents) {
    distances[doc.document_id] = calculateDistance(doc.coordinates, center);
  }
  
  return distances;
}

// Helper function to sort documents by distance to center
function sortDocumentsByDistance(documents: Document[], distances: { [documentId: string]: number }): Document[] {
  return documents.sort((a, b) => distances[a.document_id] - distances[b.document_id]);
}

// Helper function to generate random color
function generateRandomColor(): [number, number, number, number] {
  const colors = [
    [255, 0, 0, 150],     // Red
    [0, 255, 0, 150],     // Green
    [0, 0, 255, 150],     // Blue
    [255, 255, 0, 150],   // Yellow
    [255, 0, 255, 150],   // Magenta
    [0, 255, 255, 150],   // Cyan
    [255, 165, 0, 150],   // Orange
    [128, 0, 128, 150],   // Purple
    [255, 192, 203, 150], // Pink
    [0, 128, 0, 150],     // Dark Green
  ];
  return colors[Math.floor(Math.random() * colors.length)] as [number, number, number, number];
}

const useLassoSelectionMode = ({
  documents,
  isActive,
  onSelectionComplete,
  onSelectionUpdate,
  onExit
}: LassoSelectionModeProps) => {
  const [selections, setSelections] = useState<LassoSelection[]>([]);
  const [isDrawing, setIsDrawing] = useState(false);
  const [currentPath, setCurrentPath] = useState<[number, number][]>([]);
  const [currentColor] = useState<[number, number, number, number]>(() => generateRandomColor());

  // Handle mouse down to start drawing
  const handleMouseDown = useCallback((info: any) => {
    if (!isActive || !info.coordinate) return;
    
    const startPoint: [number, number] = [info.coordinate[0], info.coordinate[1]];
    
    setCurrentPath([startPoint]);
    setIsDrawing(true);
  }, [isActive]);

  // Handle mouse move to update path
  const handleMouseMove = useCallback((info: any) => {
    if (!isActive || !isDrawing || !info.coordinate) return;
    
    const currentPoint: [number, number] = [info.coordinate[0], info.coordinate[1]];
    
    setCurrentPath(prev => [...prev, currentPoint]);
  }, [isActive, isDrawing]);

  // Handle mouse up to complete selection
  const handleMouseUp = useCallback(() => {
    if (!isActive || !isDrawing || currentPath.length < 3) return;
    
    // Close the polygon by connecting to the start point
    const closedPath = [...currentPath, currentPath[0]];
    
    const documentsInLasso = findDocumentsInLasso(documents, closedPath);
    
    if (documentsInLasso.length > 0) {
      const center = calculateCentroid(closedPath);
      const distances = calculateDistancesToCenter(documentsInLasso, center);
      const sortedDocuments = sortDocumentsByDistance(documentsInLasso, distances);
      
      const newSelection: LassoSelection = {
        id: `lasso_${Date.now()}`,
        path: closedPath,
        documents: sortedDocuments,
        center,
        color: currentColor,
        distanceToCenter: distances
      };
      
      const updatedSelections = [...selections, newSelection];
      setSelections(updatedSelections);
      onSelectionUpdate(updatedSelections);
      onSelectionComplete(newSelection);
      
      console.log(`✅ Lasso selection completed: ${documentsInLasso.length} documents found`);
      console.log(`📍 Selection center: (${center[0].toFixed(2)}, ${center[1].toFixed(2)})`);
    }
    
    setCurrentPath([]);
    setIsDrawing(false);
  }, [isActive, isDrawing, currentPath, documents, selections, currentColor, onSelectionComplete, onSelectionUpdate]);

  // Handle escape key to exit mode
  useEffect(() => {
    const handleKeyDown = (event: KeyboardEvent) => {
      if (event.key === 'Escape' && isActive) {
        onExit();
      }
    };
    
    if (isActive) {
      document.addEventListener('keydown', handleKeyDown);
      return () => document.removeEventListener('keydown', handleKeyDown);
    }
  }, [isActive, onExit]);

  // Clear selections when mode is deactivated
  useEffect(() => {
    if (!isActive) {
      setSelections([]);
      setCurrentPath([]);
      setIsDrawing(false);
    }
  }, [isActive]);

  // Create layers for lasso selections
  const createLassoLayers = () => {
    const layers = [];
    
    // Completed selections
    selections.forEach(selection => {
      layers.push(
        new PathLayer({
          id: `lasso-${selection.id}`,
          data: [selection.path],
          getPath: (d: any) => d,
          getColor: selection.color,
          getWidth: 3,
          pickable: false
        })
      );
      
      // Selection center marker
      layers.push(
        new ScatterplotLayer({
          id: `lasso-center-${selection.id}`,
          data: [{
            position: selection.center,
            radius: 0.5
          }],
          getPosition: (d: any) => d.position,
          getRadius: (d: any) => d.radius,
          radiusMinPixels: 6,
          radiusMaxPixels: 12,
          getFillColor: selection.color,
          getLineColor: [255, 255, 255, 255],
          lineWidthMinPixels: 2,
          pickable: false
        })
      );
      
      // Documents in selection (highlighted by distance to center)
      if (selection.documents.length > 0) {
        layers.push(
          new ScatterplotLayer({
            id: `lasso-documents-${selection.id}`,
            data: selection.documents.map(doc => ({
              ...doc,
              distance: selection.distanceToCenter[doc.document_id]
            })),
            getPosition: (d: any) => d.coordinates,
            getRadius: (d: any) => Math.max(300, 800 - (d.distance * 100)), // Closer to center = larger
            radiusMinPixels: 8,
            radiusMaxPixels: 120,
            getFillColor: (d: any) => {
              // Color intensity based on distance to center
              const maxDistance = Math.max(...Object.values(selection.distanceToCenter));
              const normalizedDistance = d.distance / maxDistance;
              const intensity = 1 - (normalizedDistance * 0.5); // 0.5 to 1.0 intensity
              
              return [
                selection.color[0],
                selection.color[1], 
                selection.color[2],
                Math.round(selection.color[3] * intensity)
              ];
            },
            getLineColor: [255, 255, 255, 255],
            lineWidthMinPixels: 2,
            pickable: true
          })
        );
      }
    });
    
    // Current path being drawn
    if (isDrawing && currentPath.length > 0) {
      layers.push(
        new PathLayer({
          id: 'current-lasso',
          data: [currentPath],
          getPath: (d: any) => d,
          getColor: [255, 255, 255, 200], // White for current path
          getWidth: 2,
          pickable: false
        })
      );
      
      // Current path points
      layers.push(
        new ScatterplotLayer({
          id: 'current-lasso-points',
          data: currentPath.map((point, index) => ({
            position: point,
            radius: 0.2,
            isStart: index === 0
          })),
          getPosition: (d: any) => d.position,
          getRadius: (d: any) => d.radius,
          radiusMinPixels: 4,
          radiusMaxPixels: 8,
          getFillColor: (d: any) => d.isStart ? [0, 255, 0, 255] : [255, 255, 255, 255], // Green start point
          getLineColor: [255, 255, 255, 255],
          lineWidthMinPixels: 1,
          pickable: false
        })
      );
    }
    
    return layers;
  };

  return {
    layers: createLassoLayers(),
    handlers: {
      onMouseDown: handleMouseDown,
      onMouseMove: handleMouseMove,
      onMouseUp: handleMouseUp
    },
    selections,
    isDrawing,
    currentPath
  };
};

export default useLassoSelectionMode;
