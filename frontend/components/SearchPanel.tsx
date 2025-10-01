'use client';

import React, { useState } from 'react';

interface SearchPanelProps {
  onSearch: (query: string, coordinates: { x: number; y: number }, radius: number) => void;
  onClear: () => void;
}

const SearchPanel: React.FC<SearchPanelProps> = ({ onSearch, onClear }) => {
  const [query, setQuery] = useState('');
  const [coordinates, setCoordinates] = useState({ x: 0, y: 0 });
  const [radius, setRadius] = useState(1.0);
  const [loading, setLoading] = useState(false);

  const handleSearch = async () => {
    if (!query.trim()) return;

    setLoading(true);
    try {
      await onSearch(query, coordinates, radius);
    } finally {
      setLoading(false);
    }
  };

  const handleClear = () => {
    setQuery('');
    setCoordinates({ x: 0, y: 0 });
    setRadius(1.0);
    onClear();
  };

  return (
    <div className="bg-white bg-opacity-95 rounded-lg p-4 shadow-lg max-w-sm">
      <h3 className="text-lg font-bold mb-3 text-gray-800">Semantic Search</h3>
      
      {/* Search query input */}
      <div className="mb-4">
        <label className="block text-sm font-medium text-gray-700 mb-1">
          Search Query
        </label>
        <textarea
          value={query}
          onChange={(e) => setQuery(e.target.value)}
          className="w-full px-3 py-2 border border-gray-300 rounded-md focus:outline-none focus:ring-2 focus:ring-blue-500 resize-none"
          rows={3}
          placeholder="Enter your search query (e.g., 'machine learning artificial intelligence')"
        />
      </div>

      {/* Coordinates input */}
      <div className="mb-4">
        <label className="block text-sm font-medium text-gray-700 mb-1">
          Search Center Coordinates
        </label>
        <div className="grid grid-cols-2 gap-2">
          <div>
            <label className="block text-xs text-gray-600 mb-1">X</label>
            <input
              type="number"
              value={coordinates.x}
              onChange={(e) => setCoordinates(prev => ({ ...prev, x: parseFloat(e.target.value) || 0 }))}
              className="w-full px-3 py-2 border border-gray-300 rounded-md focus:outline-none focus:ring-2 focus:ring-blue-500"
              placeholder="0"
              step="0.1"
            />
          </div>
          <div>
            <label className="block text-xs text-gray-600 mb-1">Y</label>
            <input
              type="number"
              value={coordinates.y}
              onChange={(e) => setCoordinates(prev => ({ ...prev, y: parseFloat(e.target.value) || 0 }))}
              className="w-full px-3 py-2 border border-gray-300 rounded-md focus:outline-none focus:ring-2 focus:ring-blue-500"
              placeholder="0"
              step="0.1"
            />
          </div>
        </div>
        <p className="text-xs text-gray-500 mt-1">
          Click on the map to set coordinates automatically
        </p>
      </div>

      {/* Search radius */}
      <div className="mb-4">
        <label className="block text-sm font-medium text-gray-700 mb-1">
          Search Radius
        </label>
        <select 
          value={radius} 
          onChange={(e) => setRadius(parseFloat(e.target.value))}
          className="w-full px-3 py-2 border border-gray-300 rounded-md focus:outline-none focus:ring-2 focus:ring-blue-500"
        >
          <option value="0.5">Small (0.5)</option>
          <option value="1.0">Medium (1.0)</option>
          <option value="2.0">Large (2.0)</option>
        </select>
      </div>

      {/* Action buttons */}
      <div className="flex space-x-2">
        <button
          onClick={handleSearch}
          disabled={loading || !query.trim()}
          className="flex-1 bg-blue-600 text-white px-3 py-2 rounded text-sm hover:bg-blue-700 disabled:bg-gray-400 disabled:cursor-not-allowed"
        >
          {loading ? 'Searching...' : 'Search'}
        </button>
        <button
          onClick={handleClear}
          className="flex-1 bg-gray-600 text-white px-3 py-2 rounded text-sm hover:bg-gray-700"
        >
          Clear
        </button>
      </div>

      {/* Search tips */}
      <div className="mt-4 p-3 bg-blue-50 rounded">
        <h4 className="font-semibold text-blue-800 mb-2">Search Tips</h4>
        <ul className="text-sm text-blue-700 space-y-1">
          <li>• Use specific terms for better results</li>
          <li>• Click on the map to set search center</li>
          <li>• Results are ranked by semantic similarity</li>
          <li>• Yellow highlights show search results</li>
        </ul>
      </div>
    </div>
  );
};

export default SearchPanel;
