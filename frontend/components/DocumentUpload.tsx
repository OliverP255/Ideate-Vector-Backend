'use client';

import React, { useState, useRef } from 'react';

interface DocumentUploadProps {
  userId: string;
  onDocumentUploaded?: (document: any) => void;
}

const DocumentUpload: React.FC<DocumentUploadProps> = ({ userId, onDocumentUploaded }) => {
  const [isUploading, setIsUploading] = useState(false);
  const [uploadStatus, setUploadStatus] = useState<string | null>(null);
  const [uploadedDocuments, setUploadedDocuments] = useState<any[]>([]);
  const fileInputRef = useRef<HTMLInputElement>(null);

  const handleFileUpload = async (event: React.ChangeEvent<HTMLInputElement>) => {
    const file = event.target.files?.[0];
    if (!file) return;

    if (!file.name.toLowerCase().endsWith('.pdf')) {
      setUploadStatus('❌ Only PDF files are supported');
      return;
    }

    setIsUploading(true);
    setUploadStatus('📤 Uploading document...');

    try {
      const formData = new FormData();
      formData.append('file', file);
      formData.append('user_id', userId);

      const response = await fetch('http://localhost:8001/api/upload/document', {
        method: 'POST',
        body: formData,
      });

      if (!response.ok) {
        throw new Error(`Upload failed: ${response.status} ${response.statusText}`);
      }

      const result = await response.json();
      
      if (result.success) {
        setUploadStatus(`✅ ${result.message}`);
        setUploadedDocuments(prev => [...prev, result.document]);
        
        if (onDocumentUploaded) {
          onDocumentUploaded(result.document);
        }
      } else {
        setUploadStatus(`❌ Upload failed: ${result.message}`);
      }
    } catch (error) {
      console.error('Upload error:', error);
      setUploadStatus(`❌ Upload failed: ${error instanceof Error ? error.message : 'Unknown error'}`);
    } finally {
      setIsUploading(false);
      // Clear the file input
      if (fileInputRef.current) {
        fileInputRef.current.value = '';
      }
    }
  };

  const clearUploadedDocuments = async () => {
    try {
      const response = await fetch(`http://localhost:8001/api/upload/documents/${userId}`, {
        method: 'DELETE',
      });

      if (response.ok) {
        setUploadedDocuments([]);
        setUploadStatus('🗑️ Cleared uploaded documents');
      }
    } catch (error) {
      console.error('Error clearing documents:', error);
    }
  };

  return (
    <div className="bg-white bg-opacity-95 rounded-lg p-4 shadow-lg max-w-sm">
      <h3 className="text-lg font-bold mb-3 text-gray-800">Document Upload</h3>
      
      <div className="mb-4">
        <input
          ref={fileInputRef}
          type="file"
          accept=".pdf"
          onChange={handleFileUpload}
          disabled={isUploading}
          className="w-full px-3 py-2 border border-gray-300 rounded-md focus:outline-none focus:ring-2 focus:ring-blue-500 disabled:bg-gray-100"
        />
        <p className="text-xs text-gray-500 mt-1">
          Upload PDF documents to add them to your session
        </p>
      </div>

      {uploadStatus && (
        <div className={`mb-4 p-2 rounded text-sm ${
          uploadStatus.includes('✅') ? 'bg-green-50 text-green-800' :
          uploadStatus.includes('❌') ? 'bg-red-50 text-red-800' :
          'bg-blue-50 text-blue-800'
        }`}>
          {uploadStatus}
        </div>
      )}

      {uploadedDocuments.length > 0 && (
        <div className="mb-4">
          <div className="flex justify-between items-center mb-2">
            <h4 className="font-semibold text-sm text-gray-700">
              Uploaded Documents ({uploadedDocuments.length})
            </h4>
            <button
              onClick={clearUploadedDocuments}
              className="text-xs text-red-600 hover:text-red-800"
            >
              Clear All
            </button>
          </div>
          
          <div className="space-y-2 max-h-32 overflow-y-auto">
            {uploadedDocuments.map((doc, index) => (
              <div key={doc.document_id} className="text-xs p-2 bg-gray-50 rounded">
                <div className="font-medium text-gray-800">{doc.title}</div>
                <div className="text-gray-600">
                  Coords: ({doc.coordinates[0].toFixed(2)}, {doc.coordinates[1].toFixed(2)})
                </div>
              </div>
            ))}
          </div>
        </div>
      )}

      <div className="text-xs text-gray-500">
        <p>• Documents are session-only (not saved)</p>
        <p>• They will appear on the map</p>
        <p>• Refresh page to remove them</p>
      </div>
    </div>
  );
};

export default DocumentUpload;

