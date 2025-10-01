'use client';

import React, { useState, useEffect } from 'react';

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

interface LiquidGlassResultsPanelProps {
  isOpen: boolean;
  onClose: () => void;
  searchResult: ClickResult | null;
  onDocumentClick: (document: Document) => void;
  onLoadMore: () => void;
  onCenterMap: (coordinates: [number, number]) => void;
}

const LiquidGlassResultsPanel: React.FC<LiquidGlassResultsPanelProps> = ({
  isOpen,
  onClose,
  searchResult,
  onDocumentClick,
  onLoadMore,
  onCenterMap
}) => {
  const [readDocuments, setReadDocuments] = useState<Set<string>>(new Set());
  const [pinnedDocuments, setPinnedDocuments] = useState<Set<string>>(new Set());

  // Auto-hide after idle time (configurable - default off)
  useEffect(() => {
    if (!isOpen) return;

    let idleTimer: NodeJS.Timeout;
    
    const resetIdleTimer = () => {
      clearTimeout(idleTimer);
      idleTimer = setTimeout(() => {
        // Auto-hide is disabled by default per spec
        // onClose();
      }, 12000); // 12 seconds
    };

    const handleActivity = () => resetIdleTimer();
    
    document.addEventListener('mousemove', handleActivity);
    document.addEventListener('keydown', handleActivity);
    document.addEventListener('scroll', handleActivity);
    
    resetIdleTimer();
    
    return () => {
      clearTimeout(idleTimer);
      document.removeEventListener('mousemove', handleActivity);
      document.removeEventListener('keydown', handleActivity);
      document.removeEventListener('scroll', handleActivity);
    };
  }, [isOpen]);

  // Handle ESC key
  useEffect(() => {
    const handleEsc = (e: KeyboardEvent) => {
      if (e.key === 'Escape' && isOpen) {
        onClose();
      }
    };

    document.addEventListener('keydown', handleEsc);
    return () => document.removeEventListener('keydown', handleEsc);
  }, [isOpen, onClose]);

  const handleDocumentClick = (document: Document) => {
    onDocumentClick(document);
    onCenterMap(document.coordinates);
  };

  const toggleRead = (documentId: string) => {
    setReadDocuments(prev => {
      const newSet = new Set(prev);
      if (newSet.has(documentId)) {
        newSet.delete(documentId);
      } else {
        newSet.add(documentId);
      }
      return newSet;
    });
  };

  const togglePin = (documentId: string) => {
    setPinnedDocuments(prev => {
      const newSet = new Set(prev);
      if (newSet.has(documentId)) {
        newSet.delete(documentId);
      } else {
        newSet.add(documentId);
      }
      return newSet;
    });
  };

  const getClusterColor = (document: Document): string => {
    // Generate a consistent color based on document coordinates
    const [x, y] = document.coordinates;
    const hue = ((x + 20) * 360 / 40) % 360;
    const saturation = 70 + ((y + 20) * 30 / 40);
    const lightness = 50 + ((y + 20) * 20 / 40);
    return `hsl(${hue}, ${saturation}%, ${lightness}%)`;
  };

  if (!isOpen || !searchResult) return null;

  return (
    <>
      {/* Backdrop */}
      <div
        className="fixed inset-0 bg-black/20 backdrop-blur-sm z-40"
        onClick={onClose}
      />
      
      {/* Results Panel */}
      <div className={`results-panel ${isOpen ? 'open' : ''}`}>
        {/* Header */}
        <div className="flex items-center justify-between p-6 border-b border-white/10">
          <div>
            <h2 className="text-lg font-semibold text-primary">
              Search Results
            </h2>
            <p className="text-sm text-secondary">
              {searchResult.documents.length} of {searchResult.total_count || searchResult.documents.length} documents
            </p>
          </div>
          <button
            onClick={onClose}
            className="p-2 rounded-full hover:bg-white/10 transition-colors"
            title="Close results"
          >
            <span className="text-lg text-secondary">✕</span>
          </button>
        </div>

        {/* Search Info */}
        {searchResult.query_text && (
          <div className="p-4 border-b border-white/10">
            <div className="glass rounded-lg p-3">
              <p className="text-sm text-primary">
                <strong>Query:</strong> "{searchResult.query_text}"
              </p>
              <p className="text-xs text-secondary mt-1">
                Click location: ({searchResult.click_coordinates.x.toFixed(2)}, {searchResult.click_coordinates.y.toFixed(2)})
              </p>
            </div>
          </div>
        )}

        {/* Results List */}
        <div className="flex-1 overflow-y-auto">
          {searchResult.documents.length > 0 ? (
            <div className="space-y-0">
              {searchResult.documents.map((doc, index) => (
                <div
                  key={doc.document_id}
                  className={`result-item ${readDocuments.has(doc.document_id) ? 'read' : ''}`}
                  onClick={() => handleDocumentClick(doc)}
                >
                  {/* Cluster Color Dot */}
                  <div
                    className="cluster-dot"
                    style={{ backgroundColor: getClusterColor(doc) }}
                  />

                  {/* Document Info */}
                  <div className="flex-1 min-w-0">
                    <div className="flex items-center gap-2 mb-1">
                      <span className="text-xs font-bold text-secondary bg-white/10 px-2 py-1 rounded">
                        #{index + 1}
                      </span>
                      {readDocuments.has(doc.document_id) && (
                        <span className="text-xs text-accent">✓ Read</span>
                      )}
                      {pinnedDocuments.has(doc.document_id) && (
                        <span className="text-xs text-accent">📌 Pinned</span>
                      )}
                    </div>
                    
                    <h3 className="text-sm font-medium text-primary truncate mb-1">
                      {doc.title || doc.document_id}
                    </h3>
                    
                    <div className="flex items-center gap-4 text-xs text-secondary">
                      <span>ID: {doc.document_id.slice(0, 8)}...</span>
                      {doc.similarity_score && (
                        <span>Score: {doc.similarity_score.toFixed(3)}</span>
                      )}
                    </div>
                  </div>

                  {/* Action Icons */}
                  <div className="flex items-center gap-1 opacity-0 group-hover:opacity-100 transition-opacity">
                    <button
                      onClick={(e) => {
                        e.stopPropagation();
                        handleDocumentClick(doc);
                      }}
                      className="p-1 rounded hover:bg-white/10 transition-colors"
                      title="Open document"
                    >
                      <span className="text-sm">📖</span>
                    </button>
                    
                    <button
                      onClick={(e) => {
                        e.stopPropagation();
                        togglePin(doc.document_id);
                      }}
                      className={`p-1 rounded hover:bg-white/10 transition-colors ${
                        pinnedDocuments.has(doc.document_id) ? 'text-accent' : 'text-secondary'
                      }`}
                      title={pinnedDocuments.has(doc.document_id) ? 'Unpin' : 'Pin document'}
                    >
                      <span className="text-sm">📌</span>
                    </button>
                    
                    <button
                      onClick={(e) => {
                        e.stopPropagation();
                        toggleRead(doc.document_id);
                      }}
                      className={`p-1 rounded hover:bg-white/10 transition-colors ${
                        readDocuments.has(doc.document_id) ? 'text-accent' : 'text-secondary'
                      }`}
                      title={readDocuments.has(doc.document_id) ? 'Mark unread' : 'Mark as read'}
                    >
                      <span className="text-sm">✓</span>
                    </button>
                    
                    <button
                      onClick={(e) => {
                        e.stopPropagation();
                        // TODO: Implement note functionality
                      }}
                      className="p-1 rounded hover:bg-white/10 transition-colors text-secondary"
                      title="Add note"
                    >
                      <span className="text-sm">📝</span>
                    </button>
                  </div>
                </div>
              ))}
            </div>
          ) : (
            <div className="flex items-center justify-center h-64 text-center">
              <div>
                <div className="text-4xl mb-4">🔍</div>
                <p className="text-primary font-medium">No results found</p>
                <p className="text-secondary text-sm mt-1">
                  Try adjusting your search query
                </p>
              </div>
            </div>
          )}
        </div>

        {/* Load More Button */}
        {searchResult.has_more && (
          <div className="p-4 border-t border-white/10">
            <button
              onClick={onLoadMore}
              className="w-full py-3 px-4 rounded-lg bg-white/10 hover:bg-white/20 transition-colors text-primary font-medium"
            >
              Load More ({searchResult.total_count ? searchResult.total_count - searchResult.documents.length : '?'} remaining)
            </button>
          </div>
        )}

        {/* Footer */}
        <div className="p-4 border-t border-white/10">
          <div className="text-xs text-muted text-center">
            <p>Click a document to center the map</p>
            <p className="mt-1">Press Esc to close • Swipe right on mobile</p>
          </div>
        </div>
      </div>
    </>
  );
};

export default LiquidGlassResultsPanel;
