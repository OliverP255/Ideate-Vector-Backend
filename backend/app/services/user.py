"""
User service for managing user overlays and read documents.
"""

import logging
import json
import numpy as np
from pathlib import Path
from typing import Dict, Any, List, Optional
from datetime import datetime
from scipy.spatial import ConvexHull

from app.services.config import get_settings
from app.services.mapping import MappingService

logger = logging.getLogger(__name__)


class UserService:
    """Service for managing user overlays and read documents."""
    
    def __init__(self):
        self.settings = get_settings()
        self.mapping_service = MappingService()
        self.data_dir = Path("data/users")
        self.data_dir.mkdir(parents=True, exist_ok=True)
    
    async def get_user_overlay(self, user_id: str) -> Dict[str, Any]:
        """
        Get user overlay data including read documents and convex hull.
        
        Args:
            user_id: User identifier
            
        Returns:
            Dict containing user overlay data
        """
        try:
            logger.info(f"Getting user overlay for user: {user_id}")
            
            # Load user data
            user_data = await self._load_user_data(user_id)
            read_documents = user_data.get("read_documents", [])
            
            # Get all document coordinates
            all_coords = await self.mapping_service.get_all_coordinates()
            total_documents = all_coords["total_documents"]
            
            # Calculate coverage
            coverage = (len(read_documents) / total_documents * 100) if total_documents > 0 else 0
            
            # Generate convex hull for read documents
            convex_hull = None
            if len(read_documents) >= 3:
                convex_hull = await self._generate_convex_hull(read_documents, all_coords["coordinates"])
            
            return {
                "user_id": user_id,
                "read_documents": read_documents,
                "read_points": [{"document_id": doc["document_id"], "coordinates": doc.get("coordinates", [0, 0])} for doc in read_documents],
                "convex_hull": convex_hull,
                "coverage_stats": {"overall": coverage},
                "total_read": len(read_documents),
                "last_updated": datetime.now().isoformat()
            }
            
        except Exception as e:
            logger.error(f"Failed to get user overlay for {user_id}: {e}")
            raise
    
    async def mark_document_read(self, user_id: str, document_id: str, read_status: bool = True) -> bool:
        """
        Mark a document as read or unread by the user.
        
        Args:
            user_id: User identifier
            document_id: Document identifier
            read_status: Whether to mark as read (True) or unread (False)
            
        Returns:
            bool: True if successful
        """
        try:
            action = "read" if read_status else "unread"
            logger.info(f"Marking document {document_id} as {action} by user {user_id}")
            
            # Load user data
            user_data = await self._load_user_data(user_id)
            read_documents = user_data.get("read_documents", [])
            
            if read_status:
                # Add document if not already read
                if document_id not in read_documents:
                    read_documents.append(document_id)
            else:
                # Remove document if it exists
                if document_id in read_documents:
                    read_documents.remove(document_id)
            
            user_data["read_documents"] = read_documents
            user_data["last_updated"] = datetime.now().isoformat()
            
            # Save user data
            await self._save_user_data(user_id, user_data)
            
            return True
            
        except Exception as e:
            logger.error(f"Failed to mark document {document_id} as {'read' if read_status else 'unread'} by user {user_id}: {e}")
            return False
    
    async def unmark_document_read(self, user_id: str, document_id: str) -> bool:
        """
        Unmark a document as read by the user.
        
        Args:
            user_id: User identifier
            document_id: Document identifier
            
        Returns:
            bool: True if successful
        """
        try:
            logger.info(f"Unmarking document {document_id} as read by user {user_id}")
            
            # Load user data
            user_data = await self._load_user_data(user_id)
            read_documents = user_data.get("read_documents", [])
            
            # Remove document if it exists
            if document_id in read_documents:
                read_documents.remove(document_id)
                user_data["read_documents"] = read_documents
                user_data["last_updated"] = datetime.now().isoformat()
                
                # Save user data
                await self._save_user_data(user_id, user_data)
            
            return True
            
        except Exception as e:
            logger.error(f"Failed to unmark document {document_id} as read by user {user_id}: {e}")
            return False
    
    async def get_user_read_documents(self, user_id: str) -> List[str]:
        """
        Get list of documents read by the user.
        
        Args:
            user_id: User identifier
            
        Returns:
            List of document IDs
        """
        try:
            user_data = await self._load_user_data(user_id)
            return user_data.get("read_documents", [])
            
        except Exception as e:
            logger.error(f"Failed to get read documents for user {user_id}: {e}")
            return []
    
    async def get_user_profile(self, user_id: str) -> Dict[str, Any]:
        """
        Get user profile information.
        
        Args:
            user_id: User identifier
            
        Returns:
            Dict containing user profile
        """
        try:
            user_data = await self._load_user_data(user_id)
            return {
                "userId": user_id,
                "username": f"user_{user_id}",
                "email": f"{user_id}@example.com",
                "createdAt": user_data.get("created_at", "Unknown"),
                "lastActive": user_data.get("last_updated", "Never")
            }
        except Exception as e:
            logger.error(f"Failed to get user profile for {user_id}: {e}")
            raise

    async def get_user_coverage(self, user_id: str) -> Dict[str, Any]:
        """
        Get user knowledge coverage statistics.
        
        Args:
            user_id: User identifier
            
        Returns:
            Dict containing coverage statistics
        """
        try:
            user_data = await self._load_user_data(user_id)
            read_documents = user_data.get("read_documents", [])
            
            # Get all document coordinates for coverage calculation
            all_coords = await self.mapping_service.get_all_coordinates()
            total_documents = all_coords["total_documents"]
            
            # Calculate overall coverage
            overall_coverage = (len(read_documents) / total_documents * 100) if total_documents > 0 else 0
            
            # Calculate cluster coverage (simplified)
            cluster_coverage = await self._calculate_cluster_coverage(read_documents, all_coords["coordinates"])
            
            return {
                "userId": user_id,
                "overallCoverage": overall_coverage,
                "clusterCoverage": cluster_coverage,
                "readDocuments": len(read_documents),
                "totalDocuments": total_documents,
                "generatedAt": datetime.now().isoformat()
            }
            
        except Exception as e:
            logger.error(f"Failed to get user coverage for {user_id}: {e}")
            raise

    async def get_user_statistics(self, user_id: str) -> Dict[str, Any]:
        """
        Get user statistics.
        
        Args:
            user_id: User identifier
            
        Returns:
            Dict containing user statistics
        """
        try:
            user_data = await self._load_user_data(user_id)
            read_documents = user_data.get("read_documents", [])
            
            # Get all document coordinates for coverage calculation
            all_coords = await self.mapping_service.get_all_coordinates()
            total_documents = all_coords["total_documents"]
            
            # Calculate statistics
            coverage = (len(read_documents) / total_documents * 100) if total_documents > 0 else 0
            
            # Calculate cluster coverage (simplified)
            cluster_coverage = await self._calculate_cluster_coverage(read_documents, all_coords["coordinates"])
            
            return {
                "userId": user_id,
                "readDocuments": len(read_documents),
                "totalDocuments": total_documents,
                "coverage": coverage,
                "clusterCoverage": cluster_coverage,
                "lastActive": user_data.get("last_updated", "Never"),
                "createdAt": user_data.get("created_at", "Unknown")
            }
            
        except Exception as e:
            logger.error(f"Failed to get user statistics for {user_id}: {e}")
            raise
    
    async def _load_user_data(self, user_id: str) -> Dict[str, Any]:
        """Load user data from file."""
        user_file = self.data_dir / f"{user_id}.json"
        
        if user_file.exists():
            with open(user_file, 'r') as f:
                return json.load(f)
        else:
            # Create new user data
            return {
                "user_id": user_id,
                "read_documents": [],
                "created_at": datetime.now().isoformat(),
                "last_updated": datetime.now().isoformat()
            }
    
    async def _save_user_data(self, user_id: str, user_data: Dict[str, Any]) -> None:
        """Save user data to file."""
        user_file = self.data_dir / f"{user_id}.json"
        
        with open(user_file, 'w') as f:
            json.dump(user_data, f, indent=2)
    
    async def _generate_convex_hull(self, read_documents: List[str], all_coordinates: List[Dict[str, Any]]) -> Optional[Dict[str, Any]]:
        """Generate convex hull for read documents."""
        try:
            # Get coordinates for read documents
            read_coords = []
            for coord_data in all_coordinates:
                if coord_data["document_id"] in read_documents:
                    read_coords.append(coord_data["coordinates"])
            
            if len(read_coords) < 3:
                return None
            
            # Convert to numpy array
            points = np.array(read_coords)
            
            # Generate convex hull
            hull = ConvexHull(points)
            
            # Get hull vertices
            hull_points = points[hull.vertices]
            
            # Convert to GeoJSON format
            convex_hull = {
                "type": "Polygon",
                "coordinates": [hull_points.tolist() + [hull_points[0].tolist()]]  # Close the polygon
            }
            
            return convex_hull
            
        except Exception as e:
            logger.error(f"Failed to generate convex hull: {e}")
            return None
    
    async def _calculate_cluster_coverage(self, read_documents: List[str], all_coordinates: List[Dict[str, Any]]) -> float:
        """Calculate cluster coverage (simplified implementation)."""
        try:
            # This is a simplified implementation
            # In a real system, you would use actual clustering results
            
            if not read_documents:
                return 0.0
            
            # For now, return a mock coverage based on document count
            total_docs = len(all_coordinates)
            read_count = len(read_documents)
            
            # Simple coverage calculation
            coverage = min(read_count / max(total_docs, 1) * 100, 100.0)
            
            return coverage
            
        except Exception as e:
            logger.error(f"Failed to calculate cluster coverage: {e}")
            return 0.0