'use client';

import React, { useState, useRef, useEffect } from 'react';

interface SearchSuggestion {
  id: string;
  text: string;
  type: 'author' | 'title' | 'topic';
  icon: string;
}

interface LiquidGlassSearchBarProps {
  onSearch: (query: string, coordinates: { x: number; y: number }, radius: number) => void;
  onClear: () => void;
  placeholder?: string;
}

const LiquidGlassSearchBar: React.FC<LiquidGlassSearchBarProps> = ({
  onSearch,
  onClear,
  placeholder = "Search research papers..."
}) => {
  const [query, setQuery] = useState('');
  const [isFocused, setIsFocused] = useState(false);
  const [suggestions, setSuggestions] = useState<SearchSuggestion[]>([]);
  const [selectedIndex, setSelectedIndex] = useState(-1);
  const [searchMode, setSearchMode] = useState<'combined' | 'text' | 'vector'>('combined');
  const inputRef = useRef<HTMLInputElement>(null);

  // Mock suggestions - in a real app, these would come from an API
  const mockSuggestions: SearchSuggestion[] = [
    { id: '1', text: 'machine learning', type: 'topic', icon: '🔬' },
    { id: '2', text: 'neural networks', type: 'topic', icon: '🧠' },
    { id: '3', text: 'deep learning', type: 'topic', icon: '⚡' },
    { id: '4', text: 'author:Smith', type: 'author', icon: '👤' },
    { id: '5', text: 'year:2020..2023', type: 'topic', icon: '📅' },
    { id: '6', text: 'artificial intelligence', type: 'topic', icon: '🤖' },
    { id: '7', text: 'natural language processing', type: 'topic', icon: '💬' },
    { id: '8', text: 'computer vision', type: 'topic', icon: '👁️' },
  ];

  // Filter suggestions based on query
  useEffect(() => {
    if (query.length > 1) {
      const filtered = mockSuggestions.filter(suggestion =>
        suggestion.text.toLowerCase().includes(query.toLowerCase())
      ).slice(0, 6);
      setSuggestions(filtered);
    } else {
      setSuggestions([]);
    }
    setSelectedIndex(-1);
  }, [query]);

  // Handle keyboard navigation
  const handleKeyDown = (e: React.KeyboardEvent) => {
    if (e.key === 'ArrowDown') {
      e.preventDefault();
      setSelectedIndex(prev => 
        prev < suggestions.length - 1 ? prev + 1 : prev
      );
    } else if (e.key === 'ArrowUp') {
      e.preventDefault();
      setSelectedIndex(prev => prev > 0 ? prev - 1 : -1);
    } else if (e.key === 'Enter') {
      e.preventDefault();
      if (selectedIndex >= 0 && suggestions[selectedIndex]) {
        handleSuggestionSelect(suggestions[selectedIndex]);
      } else {
        handleSearch();
      }
    } else if (e.key === 'Escape') {
      setSuggestions([]);
      setSelectedIndex(-1);
      setIsFocused(false);
      inputRef.current?.blur();
    }
  };

  const handleSearch = () => {
    if (query.trim()) {
      // Use center of current viewport as search coordinates
      // In a real app, this would be the center of the map view
      const coordinates = { x: 0, y: 0 }; // Default center
      const radius = 2.0; // Default radius
      
      onSearch(query.trim(), coordinates, radius);
      setSuggestions([]);
      setSelectedIndex(-1);
    }
  };

  const handleSuggestionSelect = (suggestion: SearchSuggestion) => {
    setQuery(suggestion.text);
    setSuggestions([]);
    setSelectedIndex(-1);
    handleSearch();
  };

  const handleModeToggle = () => {
    const modes: Array<'combined' | 'text' | 'vector'> = ['combined', 'text', 'vector'];
    const currentIndex = modes.indexOf(searchMode);
    const nextIndex = (currentIndex + 1) % modes.length;
    setSearchMode(modes[nextIndex]);
  };

  const getModeIcon = () => {
    switch (searchMode) {
      case 'text': return '📝';
      case 'vector': return '🧮';
      default: return '🔍';
    }
  };

  const getModeTooltip = () => {
    switch (searchMode) {
      case 'text': return 'Text search only';
      case 'vector': return 'Vector similarity search only';
      default: return 'Combined text + vector search';
    }
  };

  // Focus search with '/' key
  useEffect(() => {
    const handleKeyPress = (e: KeyboardEvent) => {
      if (e.key === '/' && !isFocused) {
        e.preventDefault();
        inputRef.current?.focus();
      }
    };

    document.addEventListener('keydown', handleKeyPress);
    return () => document.removeEventListener('keydown', handleKeyPress);
  }, [isFocused]);

  return (
    <div className="relative w-full max-w-2xl mx-auto">
      {/* Main Search Bar */}
      <div 
        className={`search-bar flex items-center px-6 transition-all duration-350 ${
          isFocused ? 'scale-105' : 'scale-100'
        }`}
      >
        {/* Search Mode Toggle */}
        <button
          onClick={handleModeToggle}
          className="mr-3 p-1 rounded-full hover:bg-white/10 transition-colors"
          title={getModeTooltip()}
        >
          <span className="text-lg">{getModeIcon()}</span>
        </button>

        {/* Search Input */}
        <input
          ref={inputRef}
          type="text"
          value={query}
          onChange={(e) => setQuery(e.target.value)}
          onKeyDown={handleKeyDown}
          onFocus={() => setIsFocused(true)}
          onBlur={() => {
            // Delay to allow suggestion clicks
            setTimeout(() => setIsFocused(false), 150);
          }}
          placeholder={placeholder}
          className="flex-1 bg-transparent border-none outline-none text-primary text-base font-normal"
          role="search"
          aria-label="Search research papers"
        />

        {/* Clear Button */}
        {query && (
          <button
            onClick={() => {
              setQuery('');
              onClear();
              inputRef.current?.focus();
            }}
            className="ml-3 p-1 rounded-full hover:bg-white/10 transition-colors"
            title="Clear search"
          >
            <span className="text-lg">✕</span>
          </button>
        )}

        {/* Search Button */}
        <button
          onClick={handleSearch}
          className="ml-3 p-2 rounded-full hover:bg-white/10 transition-colors"
          title="Search"
        >
          <span className="text-lg">🔍</span>
        </button>
      </div>

      {/* Search Suggestions */}
      {suggestions.length > 0 && isFocused && (
        <div className="absolute top-full left-0 right-0 mt-2 z-50">
          <div className="glass rounded-lg p-2 max-h-64 overflow-y-auto">
            <div className="flex flex-wrap gap-2">
              {suggestions.map((suggestion, index) => (
                <button
                  key={suggestion.id}
                  onClick={() => handleSuggestionSelect(suggestion)}
                  className={`flex items-center gap-2 px-3 py-2 rounded-full text-sm transition-all duration-200 ${
                    index === selectedIndex
                      ? 'bg-white/20 text-white'
                      : 'bg-white/5 text-secondary hover:bg-white/10 hover:text-primary'
                  }`}
                >
                  <span>{suggestion.icon}</span>
                  <span>{suggestion.text}</span>
                </button>
              ))}
            </div>
          </div>
        </div>
      )}

      {/* Search Mode Indicator */}
      {isFocused && (
        <div className="absolute top-full left-0 mt-2 text-xs text-muted">
          {getModeTooltip()} • Press Enter to search, Esc to close
        </div>
      )}
    </div>
  );
};

export default LiquidGlassSearchBar;
