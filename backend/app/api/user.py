"""
User profile and overlay endpoints.
"""

from fastapi import APIRouter, HTTPException
from typing import List, Optional
from app.models.user import UserProfile, UserOverlay, UserRead
from app.services.user import UserService

router = APIRouter()


@router.get("/user/{user_id}/profile", response_model=UserProfile)
async def get_user_profile(user_id: str):
    """
    Get user profile information.
    
    Args:
        user_id: Unique user identifier
        
    Returns:
        UserProfile: User profile data
    """
    try:
        user_service = UserService()
        profile = await user_service.get_user_profile(user_id)
        return profile
    except Exception as e:
        raise HTTPException(status_code=404, detail=f"User not found: {str(e)}")


@router.get("/user/{user_id}/overlay", response_model=UserOverlay)
async def get_user_overlay(user_id: str):
    """
    Get user overlay data for map visualization.
    
    Args:
        user_id: Unique user identifier
        
    Returns:
        UserOverlay: User overlay data with read documents and coverage
    """
    try:
        user_service = UserService()
        overlay = await user_service.get_user_overlay(user_id)
        return overlay
    except Exception as e:
        raise HTTPException(status_code=404, detail=f"User overlay not found: {str(e)}")


@router.post("/user/{user_id}/read")
async def mark_document_read(
    user_id: str,
    document_id: str,
    read_status: bool = True
):
    """
    Mark a document as read or unread by a user.
    
    Args:
        user_id: Unique user identifier
        document_id: Document identifier
        read_status: Whether the document is read (True) or unread (False)
        
    Returns:
        dict: Success status
    """
    try:
        user_service = UserService()
        await user_service.mark_document_read(user_id, document_id, read_status)
        return {"status": "success", "message": f"Document {document_id} marked as {'read' if read_status else 'unread'}"}
    except Exception as e:
        raise HTTPException(status_code=400, detail=f"Failed to update read status: {str(e)}")


@router.get("/user/{user_id}/read", response_model=List[UserRead])
async def get_user_read_documents(user_id: str):
    """
    Get list of documents read by a user.
    
    Args:
        user_id: Unique user identifier
        
    Returns:
        List[UserRead]: List of read documents with metadata
    """
    try:
        user_service = UserService()
        read_documents = await user_service.get_user_read_documents(user_id)
        return read_documents
    except Exception as e:
        raise HTTPException(status_code=404, detail=f"User read documents not found: {str(e)}")


@router.get("/user/{user_id}/coverage")
async def get_user_coverage(user_id: str):
    """
    Get user knowledge coverage statistics.
    
    Args:
        user_id: Unique user identifier
        
    Returns:
        dict: Coverage statistics by cluster and overall
    """
    try:
        user_service = UserService()
        coverage = await user_service.get_user_coverage(user_id)
        return coverage
    except Exception as e:
        raise HTTPException(status_code=404, detail=f"User coverage not found: {str(e)}")
