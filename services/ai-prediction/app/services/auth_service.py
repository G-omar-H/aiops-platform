"""
Authentication Service

Handles JWT token verification and user authentication.
"""

import logging
from typing import Dict, Any, Optional
from datetime import datetime, timedelta

import jwt
from fastapi import HTTPException, status

from app.core.config import get_settings

logger = logging.getLogger(__name__)
settings = get_settings()


async def verify_token(token: str) -> Dict[str, Any]:
    """
    Verify JWT token and return user information.
    
    Args:
        token: JWT token to verify
        
    Returns:
        Dict containing user information
        
    Raises:
        HTTPException: If token is invalid or expired
    """
    try:
        # Decode JWT token
        payload = jwt.decode(
            token,
            settings.SECRET_KEY,
            algorithms=["HS256"]
        )
        
        # Check if token is expired
        exp = payload.get('exp')
        if exp and datetime.utcnow().timestamp() > exp:
            raise HTTPException(
                status_code=status.HTTP_401_UNAUTHORIZED,
                detail="Token has expired",
                headers={"WWW-Authenticate": "Bearer"}
            )
        
        # Extract user information
        user_info = {
            'user_id': payload.get('sub'),
            'username': payload.get('username'),
            'email': payload.get('email'),
            'role': payload.get('role', 'user'),
            'permissions': payload.get('permissions', []),
            'exp': exp
        }
        
        return user_info
        
    except jwt.ExpiredSignatureError:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Token has expired",
            headers={"WWW-Authenticate": "Bearer"}
        )
    except jwt.InvalidTokenError:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Invalid token",
            headers={"WWW-Authenticate": "Bearer"}
        )
    except Exception as e:
        logger.error(f"Token verification failed: {e}")
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Token verification failed",
            headers={"WWW-Authenticate": "Bearer"}
        )


def create_access_token(data: Dict[str, Any], expires_delta: Optional[timedelta] = None) -> str:
    """
    Create a new JWT access token.
    
    Args:
        data: Data to encode in the token
        expires_delta: Token expiration time
        
    Returns:
        Encoded JWT token
    """
    to_encode = data.copy()
    
    if expires_delta:
        expire = datetime.utcnow() + expires_delta
    else:
        expire = datetime.utcnow() + timedelta(hours=24)
    
    to_encode.update({"exp": expire})
    
    encoded_jwt = jwt.encode(to_encode, settings.SECRET_KEY, algorithm="HS256")
    return encoded_jwt


async def check_permissions(user: Dict[str, Any], required_permission: str) -> bool:
    """
    Check if user has required permission.
    
    Args:
        user: User information from JWT token
        required_permission: Required permission string
        
    Returns:
        True if user has permission, False otherwise
    """
    user_permissions = user.get('permissions', [])
    user_role = user.get('role', 'user')
    
    # Admin role has all permissions
    if user_role == 'admin':
        return True
    
    # Check specific permission
    return required_permission in user_permissions 