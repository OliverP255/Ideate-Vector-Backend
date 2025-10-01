'use client';

import React, { useState } from 'react';

interface UserAvatarProps {
  userId: string;
  onUserChange?: (newUserId: string) => void;
}

const UserAvatar: React.FC<UserAvatarProps> = ({
  userId,
  onUserChange
}) => {
  const [showMenu, setShowMenu] = useState(false);
  const [isEditing, setIsEditing] = useState(false);
  const [editValue, setEditValue] = useState(userId);

  const handleAvatarClick = () => {
    setShowMenu(!showMenu);
  };

  const handleEditClick = () => {
    setIsEditing(true);
    setEditValue(userId);
    setShowMenu(false);
  };

  const handleSaveEdit = () => {
    if (editValue.trim() && onUserChange) {
      onUserChange(editValue.trim());
    }
    setIsEditing(false);
  };

  const handleCancelEdit = () => {
    setEditValue(userId);
    setIsEditing(false);
  };

  const handleKeyDown = (e: React.KeyboardEvent) => {
    if (e.key === 'Enter') {
      handleSaveEdit();
    } else if (e.key === 'Escape') {
      handleCancelEdit();
    }
  };

  // Generate avatar color based on userId
  const getAvatarColor = (id: string): string => {
    let hash = 0;
    for (let i = 0; i < id.length; i++) {
      hash = id.charCodeAt(i) + ((hash << 5) - hash);
    }
    const hue = Math.abs(hash) % 360;
    return `hsl(${hue}, 70%, 60%)`;
  };

  const getInitials = (id: string): string => {
    return id.slice(0, 2).toUpperCase();
  };

  return (
    <div className="fixed top-4 left-4 z-50">
      {isEditing ? (
        <div className="glass rounded-lg p-3 w-48 animate-spring-in">
          <div className="flex items-center gap-2">
            <input
              type="text"
              value={editValue}
              onChange={(e) => setEditValue(e.target.value)}
              onKeyDown={handleKeyDown}
              onBlur={handleSaveEdit}
              className="flex-1 bg-transparent border-none outline-none text-primary text-sm font-medium"
              placeholder="User ID"
              autoFocus
            />
            <button
              onClick={handleSaveEdit}
              className="p-1 rounded hover:bg-white/10 transition-colors text-accent"
              title="Save"
            >
              ✓
            </button>
            <button
              onClick={handleCancelEdit}
              className="p-1 rounded hover:bg-white/10 transition-colors text-secondary"
              title="Cancel"
            >
              ✕
            </button>
          </div>
        </div>
      ) : (
        <div className="relative">
          <button
            onClick={handleAvatarClick}
            className="w-6 h-6 rounded-full border-2 border-white/20 hover:border-white/40 transition-colors flex items-center justify-center text-xs font-bold text-white"
            style={{ backgroundColor: getAvatarColor(userId) }}
            title={`User: ${userId}`}
          >
            {getInitials(userId)}
          </button>

          {/* Menu */}
          {showMenu && (
            <>
              <div
                className="fixed inset-0 z-40"
                onClick={() => setShowMenu(false)}
              />
              <div className="absolute top-full left-0 mt-2 glass rounded-lg p-2 w-48 animate-spring-in z-50">
                <div className="text-xs text-secondary px-3 py-2 border-b border-white/10">
                  User ID
                </div>
                <div className="text-sm text-primary px-3 py-2 font-mono">
                  {userId}
                </div>
                <button
                  onClick={handleEditClick}
                  className="w-full text-left px-3 py-2 text-sm text-secondary hover:bg-white/10 rounded transition-colors"
                >
                  ✏️ Edit User ID
                </button>
              </div>
            </>
          )}
        </div>
      )}
    </div>
  );
};

export default UserAvatar;
