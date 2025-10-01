'use client';

import React, { useState, useRef } from 'react';

interface ActionButtonsProps {
  onUpload: (files: File[]) => void;
  onGapFilling: () => void;
  uploadProgress?: number;
  isUploading?: boolean;
}

const ActionButtons: React.FC<ActionButtonsProps> = ({
  onUpload,
  onGapFilling,
  uploadProgress = 0,
  isUploading = false
}) => {
  const [showGapFillingOptions, setShowGapFillingOptions] = useState(false);
  const fileInputRef = useRef<HTMLInputElement>(null);

  const handleUploadClick = () => {
    fileInputRef.current?.click();
  };

  const handleFileChange = (e: React.ChangeEvent<HTMLInputElement>) => {
    const files = Array.from(e.target.files || []);
    if (files.length > 0) {
      onUpload(files);
    }
    // Reset the input
    if (fileInputRef.current) {
      fileInputRef.current.value = '';
    }
  };

  const handleDragOver = (e: React.DragEvent) => {
    e.preventDefault();
    e.stopPropagation();
  };

  const handleDrop = (e: React.DragEvent) => {
    e.preventDefault();
    e.stopPropagation();
    
    const files = Array.from(e.dataTransfer.files);
    if (files.length > 0) {
      onUpload(files);
    }
  };

  const handleGapFillingClick = () => {
    setShowGapFillingOptions(true);
  };

  const handleGapFillingOption = (option: 'suggest-titles' | 'suggest-papers') => {
    setShowGapFillingOptions(false);
    onGapFilling();
  };

  return (
    <div className="flex items-center gap-3">
      {/* Upload Button */}
      <div className="relative">
        <button
          onClick={handleUploadClick}
          onDragOver={handleDragOver}
          onDrop={handleDrop}
          className="action-button relative overflow-hidden"
          title="Upload documents (PDF, DOCX, TXT)"
          disabled={isUploading}
        >
          {isUploading ? (
            <>
              {/* Progress Ring */}
              <div className="absolute inset-0 flex items-center justify-center">
                <svg className="w-6 h-6 transform -rotate-90" viewBox="0 0 24 24">
                  <circle
                    cx="12"
                    cy="12"
                    r="10"
                    stroke="rgba(255,255,255,0.2)"
                    strokeWidth="2"
                    fill="none"
                  />
                  <circle
                    cx="12"
                    cy="12"
                    r="10"
                    stroke="rgba(255,255,255,0.8)"
                    strokeWidth="2"
                    fill="none"
                    strokeDasharray={`${2 * Math.PI * 10}`}
                    strokeDashoffset={`${2 * Math.PI * 10 * (1 - uploadProgress / 100)}`}
                    className="transition-all duration-300"
                  />
                </svg>
              </div>
              <span className="text-xs font-bold text-white">
                {Math.round(uploadProgress)}%
              </span>
            </>
          ) : (
            <span className="text-xl">📁</span>
          )}
        </button>
        
        {/* Hidden file input */}
        <input
          ref={fileInputRef}
          type="file"
          multiple
          accept=".pdf,.docx,.txt,.md"
          onChange={handleFileChange}
          className="hidden"
        />
      </div>

      {/* Gap Filling Button */}
      <div className="relative">
        <button
          onClick={handleGapFillingClick}
          className="action-button gap-filling animate-pulse-glow"
          title="Suggest content to fill gaps"
        >
          <span className="text-xl">✨</span>
        </button>

        {/* Gap Filling Options Popover */}
        {showGapFillingOptions && (
          <>
            {/* Backdrop */}
            <div
              className="fixed inset-0 z-40"
              onClick={() => setShowGapFillingOptions(false)}
            />
            
            {/* Popover */}
            <div className="absolute bottom-full right-0 mb-2 z-50 glass rounded-lg p-4 w-64 animate-spring-in">
              <h3 className="text-sm font-semibold text-primary mb-3">
                Fill Knowledge Gaps
              </h3>
              
              <div className="space-y-2">
                <button
                  onClick={() => handleGapFillingOption('suggest-titles')}
                  className="w-full text-left p-3 rounded-lg hover:bg-white/10 transition-colors"
                >
                  <div className="flex items-center gap-3">
                    <span className="text-lg">📝</span>
                    <div>
                      <div className="text-sm font-medium text-primary">
                        Suggest titles/abstracts
                      </div>
                      <div className="text-xs text-secondary">
                        AI generates candidate content
                      </div>
                    </div>
                  </div>
                </button>
                
                <button
                  onClick={() => handleGapFillingOption('suggest-papers')}
                  className="w-full text-left p-3 rounded-lg hover:bg-white/10 transition-colors"
                >
                  <div className="flex items-center gap-3">
                    <span className="text-lg">🔍</span>
                    <div>
                      <div className="text-sm font-medium text-primary">
                        Suggest existing papers
                      </div>
                      <div className="text-xs text-secondary">
                        Find under-represented topics
                      </div>
                    </div>
                  </div>
                </button>
              </div>
              
              <div className="mt-3 pt-3 border-t border-white/10">
                <div className="text-xs text-muted">
                  <div className="flex items-center gap-1">
                    <span>🤖</span>
                    <span>AI-generated content will be clearly labeled</span>
                  </div>
                </div>
              </div>
            </div>
          </>
        )}
      </div>
    </div>
  );
};

export default ActionButtons;
