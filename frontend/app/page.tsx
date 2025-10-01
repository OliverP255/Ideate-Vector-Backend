'use client';

import React, { useState, useEffect } from 'react';
import KnowledgeMap from '../components/KnowledgeMap';
import LiquidGlassSearchBar from '../components/LiquidGlassSearchBar';
import ActionButtons from '../components/ActionButtons';
import LiquidGlassResultsPanel from '../components/LiquidGlassResultsPanel';
import UserAvatar from '../components/UserAvatar';
import GapFillingOverlay from '../components/GapFillingOverlay';

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
  total_count?: number;
  has_more?: boolean;
  current_offset?: number;
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

export default function Home() {
  const [selectedDocument, setSelectedDocument] = useState<Document | null>(null);
  const [clickResult, setClickResult] = useState<ClickResult | null>(null);
  const [userId, setUserId] = useState('user-123');
  const [uploadedDocuments, setUploadedDocuments] = useState<Document[]>([]);
  const [gapFilledDocuments, setGapFilledDocuments] = useState<Document[]>([]);
  const [gapFillingCoordinates, setGapFillingCoordinates] = useState<{ x: number; y: number } | null>(null);
  const [isLassoSelectionMode, setIsLassoSelectionMode] = useState(false);
  const [lassoSelections, setLassoSelections] = useState<LassoSelection[]>([]);
  const [isUploading, setIsUploading] = useState(false);
  const [uploadProgress, setUploadProgress] = useState(0);
  const [showResultsPanel, setShowResultsPanel] = useState(false);

  // Keyboard shortcuts
  useEffect(() => {
    const handleKeyPress = (e: KeyboardEvent) => {
      if (e.key === 'Escape') {
        setShowResultsPanel(false);
        setIsLassoSelectionMode(false);
      } else if (e.key === 'g' && !e.ctrlKey && !e.metaKey) {
        setIsLassoSelectionMode(!isLassoSelectionMode);
      } else if (e.key === 'u' && !e.ctrlKey && !e.metaKey) {
        // Upload shortcut - handled by ActionButtons
      }
    };

    document.addEventListener('keydown', handleKeyPress);
    return () => document.removeEventListener('keydown', handleKeyPress);
  }, [isLassoSelectionMode]);

  const handleDocumentClick = (document: Document) => {
    setSelectedDocument(document);
    console.log('Document clicked:', document);
  };

  const handleMapClick = (result: ClickResult) => {
    setClickResult(result);
    console.log('Map clicked:', result);
  };

  const handleGapFillingMapClick = (coordinates: { x: number; y: number }) => {
    setGapFillingCoordinates(coordinates);
  };

  const handleSearch = async (query: string, coordinates: { x: number; y: number }, radius: number) => {
    try {
      console.log(`🔍 Starting semantic search: "${query}"`);
      
      // Add timeout to prevent hanging
      const controller = new AbortController();
      const timeoutId = setTimeout(() => controller.abort(), 10000); // 10 second timeout for search
      
      // Use the new semantic search with spatial area endpoint
      const response = await fetch(
        `http://localhost:8001/api/search/semantic-spatial?query=${encodeURIComponent(query)}&limit=10&offset=0&spatial_expansion_factor=0.8`,
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
        throw new Error(`Search failed: ${response.status} ${response.statusText}`);
      }
      
      const searchResult = await response.json();
      
      // Convert search result to ClickResult format for compatibility
      const result: ClickResult = {
        click_coordinates: { x: coordinates.x, y: coordinates.y },
        radius: radius,
        spatial_candidates: searchResult.total_count,
        documents: searchResult.results.map((doc: any) => ({
          document_id: doc.document_id,
          coordinates: doc.coordinates,
          title: doc.title,
          similarity_score: doc.similarity_score,
          spatial_distance: doc.spatial_distance,
          rerank_method: 'semantic_search'
        })),
        query_text: query,
        processed_at: new Date().toISOString(),
        total_count: searchResult.total_count,
        has_more: searchResult.has_more,
        current_offset: searchResult.offset,
        spatial_area: searchResult.spatial_area
      };
      
      setClickResult(result);
      setShowResultsPanel(true);
      
      console.log(`✅ Search completed: found ${result.documents?.length || 0} documents (total: ${result.total_count})`);
      if (result.spatial_area) {
        console.log(`📍 Spatial area: center (${result.spatial_area.center[0].toFixed(2)}, ${result.spatial_area.center[1].toFixed(2)}), radius ${result.spatial_area.radius.toFixed(2)}`);
      }
    } catch (error) {
      if (error instanceof Error && error.name === 'AbortError') {
        console.error('Search request timed out');
      } else {
        console.error('Search error:', error);
      }
    }
  };

  const handleSearchClear = () => {
    setClickResult(null);
    setShowResultsPanel(false);
  };

  const handleLoadMore = async () => {
    if (!clickResult || !clickResult.has_more) return;
    
    try {
      console.log(`🔄 Loading more results for query: "${clickResult.query_text}"`);
      
      const nextOffset = (clickResult.current_offset || 0) + 10;
      
      // Add timeout to prevent hanging
      const controller = new AbortController();
      const timeoutId = setTimeout(() => controller.abort(), 10000); // 10 second timeout for load more
      
      const response = await fetch(
        `http://localhost:8001/api/search/semantic-spatial?query=${encodeURIComponent(clickResult.query_text || '')}&limit=10&offset=${nextOffset}&spatial_expansion_factor=0.8`,
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
        throw new Error(`Load more failed: ${response.status} ${response.statusText}`);
      }
      
      const searchResult = await response.json();
      
      // Append new results to existing ones
      const newDocuments = searchResult.results.map((doc: any) => ({
        document_id: doc.document_id,
        coordinates: doc.coordinates,
        title: doc.title,
        similarity_score: doc.similarity_score,
        spatial_distance: doc.spatial_distance,
        rerank_method: 'semantic_search'
      }));
      
      const updatedResult: ClickResult = {
        ...clickResult,
        documents: [...clickResult.documents, ...newDocuments],
        has_more: searchResult.has_more,
        current_offset: searchResult.offset
      };
      
      setClickResult(updatedResult);
      
      console.log(`✅ Loaded more results: ${newDocuments.length} additional documents`);
    } catch (error) {
      if (error instanceof Error && error.name === 'AbortError') {
        console.error('Load more request timed out');
      } else {
        console.error('Load more error:', error);
      }
    }
  };

  const handleUserChange = (newUserId: string) => {
    setUserId(newUserId);
  };

  const handleDocumentUpload = async (files: File[]) => {
    setIsUploading(true);
    setUploadProgress(0);

    try {
      for (let i = 0; i < files.length; i++) {
        const file = files[i];
        const progress = ((i + 1) / files.length) * 100;
        setUploadProgress(progress);

        // Simulate upload process
        await new Promise(resolve => setTimeout(resolve, 1000));

        // Here you would actually upload the file to your backend
        // For now, we'll create a mock document
        const uploadedDoc: Document = {
          document_id: `upload_${Date.now()}_${i}`,
          coordinates: [Math.random() * 40 - 20, Math.random() * 40 - 20], // Random coordinates
          title: file.name,
          similarity_score: 0,
          spatial_distance: 0,
          rerank_method: 'upload'
        };

        setUploadedDocuments(prev => [...prev, uploadedDoc]);
      }

      // Show success message
      console.log(`✅ Uploaded ${files.length} documents successfully`);
    } catch (error) {
      console.error('Upload error:', error);
    } finally {
      setIsUploading(false);
      setUploadProgress(0);
    }
  };

  const handleGapFilling = () => {
    // This will be handled by the GapFillingOverlay
    console.log('Gap filling initiated');
  };

  const handleGapFilled = (document: any) => {
    const gapFilledDoc: Document = {
      document_id: document.document_id,
      coordinates: document.coordinates,
      title: document.title,
      similarity_score: 0,
      spatial_distance: 0,
      rerank_method: 'gap_filling'
    };
    setGapFilledDocuments(prev => [...prev, gapFilledDoc]);
    console.log('Gap filled with document:', gapFilledDoc);
  };

  const handleLassoComplete = (selection: LassoSelection) => {
    setLassoSelections(prev => [...prev, selection]);
    console.log('Lasso selection completed:', selection);
  };

  const handleCenterMap = (coordinates: [number, number]) => {
    // This would center the map on the given coordinates
    // Implementation depends on your map component
    console.log('Centering map on:', coordinates);
  };

  return (
    <div className="min-h-screen bg-gradient-to-b from-[#05060a] to-[#0b1224] overflow-hidden">
      {/* User Avatar */}
      <UserAvatar userId={userId} onUserChange={handleUserChange} />

      {/* Main Map - Full Screen */}
      <div className="absolute inset-0">
        <KnowledgeMap
          onDocumentClick={handleDocumentClick}
          onMapClick={handleMapClick}
          onGapFillingMapClick={handleGapFillingMapClick}
          onLassoComplete={handleLassoComplete}
          userOverlay={{
            userId,
            readDocuments: [],
            convexHull: undefined
          }}
          searchResult={clickResult}
          uploadedDocuments={uploadedDocuments}
          gapFilledDocuments={gapFilledDocuments}
          isLassoSelectionMode={isLassoSelectionMode}
        />
      </div>

      {/* Bottom Search Bar and Action Buttons */}
      <div className="fixed bottom-6 left-1/2 transform -translate-x-1/2 z-30">
        <div className="flex items-center gap-4">
          {/* Search Bar */}
          <LiquidGlassSearchBar
            onSearch={handleSearch}
            onClear={handleSearchClear}
            placeholder="Search research papers..."
          />
          
          {/* Action Buttons */}
          <ActionButtons
            onUpload={handleDocumentUpload}
            onGapFilling={handleGapFilling}
            uploadProgress={uploadProgress}
            isUploading={isUploading}
          />
        </div>
      </div>

      {/* Results Panel */}
      <LiquidGlassResultsPanel
        isOpen={showResultsPanel}
        onClose={() => setShowResultsPanel(false)}
        searchResult={clickResult}
        onDocumentClick={handleDocumentClick}
        onLoadMore={handleLoadMore}
        onCenterMap={handleCenterMap}
      />

      {/* Gap Filling Overlay */}
      <GapFillingOverlay
        isActive={!!gapFillingCoordinates}
        onGapFilled={handleGapFilled}
        onClose={() => {
          setGapFillingCoordinates(null);
        }}
        gapFillingCoordinates={gapFillingCoordinates}
      />

      {/* Keyboard Shortcuts Help */}
      <div className="fixed bottom-4 right-4 z-20">
        <div className="glass rounded-lg p-3 text-xs text-secondary">
          <div className="space-y-1">
            <div><kbd className="bg-white/10 px-1 rounded">/</kbd> Focus search</div>
            <div><kbd className="bg-white/10 px-1 rounded">Esc</kbd> Close panels</div>
            <div><kbd className="bg-white/10 px-1 rounded">G</kbd> Toggle gap mode</div>
            <div><kbd className="bg-white/10 px-1 rounded">U</kbd> Upload files</div>
          </div>
        </div>
      </div>
    </div>
  );
}