'use client';

import React, { useState, useEffect } from 'react';

interface UserOverlayProps {
  userId: string;
  onUserChange?: (userId: string) => void;
}

interface UserData {
  user_id: string;
  read_documents: string[];
  read_points?: Array<{
    document_id: string;
    coordinates: number[];
  }>;
  convex_hull?: {
    type: string;
    coordinates: number[][];
  };
  coverage_stats: {
    overall: number;
  };
  total_read: number;
  last_updated: string;
}

const UserOverlay: React.FC<UserOverlayProps> = ({ userId, onUserChange }) => {
  const [userData, setUserData] = useState<UserData | null>(null);
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState<string | null>(null);

  useEffect(() => {
    if (userId) {
      loadUserData();
    }
  }, [userId]);

  const loadUserData = async () => {
    try {
      setLoading(true);
      setError(null);

      // Load user overlay data
      const response = await fetch(`http://localhost:8001/api/user/${userId}/overlay`);
      if (!response.ok) {
        throw new Error('Failed to load user overlay');
      }

      const data = await response.json();
      setUserData(data);
    } catch (err) {
      setError(err instanceof Error ? err.message : 'Failed to load user data');
    } finally {
      setLoading(false);
    }
  };

  const handleUserIdChange = (newUserId: string) => {
    if (onUserChange) {
      onUserChange(newUserId);
    }
  };

  return (
    <div className="bg-white bg-opacity-95 rounded-lg p-4 shadow-lg max-w-sm">
      <h3 className="text-lg font-bold mb-3 text-gray-800">User Overlay</h3>
      
      {/* User ID Input */}
      <div className="mb-4">
        <label className="block text-sm font-medium text-gray-700 mb-1">
          User ID
        </label>
        <input
          type="text"
          value={userId}
          onChange={(e) => handleUserIdChange(e.target.value)}
          className="w-full px-3 py-2 border border-gray-300 rounded-md focus:outline-none focus:ring-2 focus:ring-blue-500"
          placeholder="Enter user ID"
        />
      </div>

      {/* Loading state */}
      {loading && (
        <div className="text-center py-4">
          <div className="inline-block animate-spin rounded-full h-6 w-6 border-b-2 border-blue-600"></div>
          <p className="text-sm text-gray-600 mt-2">Loading user data...</p>
        </div>
      )}

      {/* Error state */}
      {error && (
        <div className="bg-red-100 border border-red-400 text-red-700 px-3 py-2 rounded mb-4">
          <p className="text-sm">{error}</p>
        </div>
      )}

      {/* User data display */}
      {userData && !loading && (
        <div className="space-y-3">
          <div className="bg-gray-50 p-3 rounded">
            <h4 className="font-semibold text-gray-800 mb-2">Coverage Statistics</h4>
            <div className="grid grid-cols-2 gap-2 text-sm">
              <div>
                <span className="text-gray-600">Read:</span>
                <span className="font-medium ml-1">{userData.read_documents?.length || 0}</span>
              </div>
              <div>
                <span className="text-gray-600">Total:</span>
                <span className="font-medium ml-1">{userData.total_read || 0}</span>
              </div>
              <div className="col-span-2">
                <span className="text-gray-600">Coverage:</span>
                <span className="font-medium ml-1">{userData.coverage_stats?.overall?.toFixed(1) || '0.0'}%</span>
              </div>
            </div>
          </div>

          {/* Convex hull info */}
          {userData.convex_hull && (
            <div className="bg-blue-50 p-3 rounded">
              <h4 className="font-semibold text-gray-800 mb-2">Convex Hull</h4>
              <p className="text-sm text-gray-600">
                Hull contains {userData.convex_hull.coordinates?.length || 0} points
              </p>
            </div>
          )}

          {/* Read documents list */}
          <div className="bg-green-50 p-3 rounded">
            <h4 className="font-semibold text-gray-800 mb-2">Read Documents</h4>
            <div className="max-h-32 overflow-y-auto">
              {userData.read_documents && userData.read_documents.length > 0 ? (
                <ul className="text-sm space-y-1">
                  {userData.read_documents.slice(0, 10).map((docId, index) => (
                    <li key={docId} className="text-gray-700">
                      {index + 1}. {docId}
                    </li>
                  ))}
                  {userData.read_documents.length > 10 && (
                    <li className="text-gray-500 italic">
                      ... and {userData.read_documents.length - 10} more
                    </li>
                  )}
                </ul>
              ) : (
                <p className="text-sm text-gray-500 italic">No documents read yet</p>
              )}
            </div>
          </div>
        </div>
      )}

      {/* Actions */}
      <div className="mt-4 flex space-x-2">
        <button
          onClick={loadUserData}
          disabled={loading || !userId}
          className="flex-1 bg-blue-600 text-white px-3 py-2 rounded text-sm hover:bg-blue-700 disabled:bg-gray-400 disabled:cursor-not-allowed"
        >
          Refresh
        </button>
        <button
          onClick={() => handleUserIdChange('')}
          className="flex-1 bg-gray-600 text-white px-3 py-2 rounded text-sm hover:bg-gray-700"
        >
          Clear
        </button>
      </div>
    </div>
  );
};

export default UserOverlay;
