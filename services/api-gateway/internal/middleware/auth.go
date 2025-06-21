package middleware

import (
	"net/http"
	"strings"
	"time"

	"github.com/gin-gonic/gin"
	"github.com/golang-jwt/jwt/v5"
	"github.com/gorilla/websocket"
)

// Claims represents the JWT claims
type Claims struct {
	UserID    string   `json:"user_id"`
	Username  string   `json:"username"`
	Email     string   `json:"email"`
	Roles     []string `json:"roles"`
	IssuedAt  int64    `json:"iat"`
	ExpiresAt int64    `json:"exp"`
	jwt.RegisteredClaims
}

// JWTAuth creates a JWT authentication middleware
func JWTAuth(secretKey string) gin.HandlerFunc {
	return func(c *gin.Context) {
		// Extract token from header
		authHeader := c.GetHeader("Authorization")
		if authHeader == "" {
			c.JSON(http.StatusUnauthorized, gin.H{
				"error":   "Unauthorized",
				"message": "Authorization header is required",
				"code":    "AUTH_HEADER_MISSING",
			})
			c.Abort()
			return
		}

		// Check if it starts with "Bearer "
		tokenString := ""
		if strings.HasPrefix(authHeader, "Bearer ") {
			tokenString = authHeader[7:]
		} else {
			c.JSON(http.StatusUnauthorized, gin.H{
				"error":   "Unauthorized",
				"message": "Authorization header must start with 'Bearer '",
				"code":    "INVALID_AUTH_FORMAT",
			})
			c.Abort()
			return
		}

		// Parse and validate token
		claims, err := ValidateToken(tokenString, secretKey)
		if err != nil {
			c.JSON(http.StatusUnauthorized, gin.H{
				"error":   "Unauthorized",
				"message": err.Error(),
				"code":    "INVALID_TOKEN",
			})
			c.Abort()
			return
		}

		// Check if token is expired
		if time.Now().Unix() > claims.ExpiresAt {
			c.JSON(http.StatusUnauthorized, gin.H{
				"error":   "Unauthorized",
				"message": "Token has expired",
				"code":    "TOKEN_EXPIRED",
			})
			c.Abort()
			return
		}

		// Store user information in context
		c.Set("user_id", claims.UserID)
		c.Set("username", claims.Username)
		c.Set("email", claims.Email)
		c.Set("roles", claims.Roles)
		c.Set("claims", claims)

		c.Next()
	}
}

// WSAuth creates a WebSocket authentication middleware
func WSAuth(secretKey string) gin.HandlerFunc {
	return func(c *gin.Context) {
		// Get token from query parameter for WebSocket connections
		tokenString := c.Query("token")
		if tokenString == "" {
			c.JSON(http.StatusUnauthorized, gin.H{
				"error":   "Unauthorized",
				"message": "Token query parameter is required for WebSocket connections",
				"code":    "WS_TOKEN_MISSING",
			})
			c.Abort()
			return
		}

		// Validate token
		claims, err := ValidateToken(tokenString, secretKey)
		if err != nil {
			c.JSON(http.StatusUnauthorized, gin.H{
				"error":   "Unauthorized",
				"message": err.Error(),
				"code":    "INVALID_WS_TOKEN",
			})
			c.Abort()
			return
		}

		// Check if token is expired
		if time.Now().Unix() > claims.ExpiresAt {
			c.JSON(http.StatusUnauthorized, gin.H{
				"error":   "Unauthorized",
				"message": "Token has expired",
				"code":    "WS_TOKEN_EXPIRED",
			})
			c.Abort()
			return
		}

		// Store user information in context
		c.Set("user_id", claims.UserID)
		c.Set("username", claims.Username)
		c.Set("email", claims.Email)
		c.Set("roles", claims.Roles)
		c.Set("claims", claims)

		c.Next()
	}
}

// RoleAuth creates a role-based authorization middleware
func RoleAuth(requiredRoles ...string) gin.HandlerFunc {
	return func(c *gin.Context) {
		// Get user roles from context
		userRoles, exists := c.Get("roles")
		if !exists {
			c.JSON(http.StatusForbidden, gin.H{
				"error":   "Forbidden",
				"message": "User roles not found in context",
				"code":    "ROLES_NOT_FOUND",
			})
			c.Abort()
			return
		}

		roles, ok := userRoles.([]string)
		if !ok {
			c.JSON(http.StatusForbidden, gin.H{
				"error":   "Forbidden",
				"message": "Invalid roles format",
				"code":    "INVALID_ROLES_FORMAT",
			})
			c.Abort()
			return
		}

		// Check if user has any of the required roles
		hasRequiredRole := false
		for _, userRole := range roles {
			for _, requiredRole := range requiredRoles {
				if userRole == requiredRole || userRole == "admin" { // admin has access to everything
					hasRequiredRole = true
					break
				}
			}
			if hasRequiredRole {
				break
			}
		}

		if !hasRequiredRole {
			c.JSON(http.StatusForbidden, gin.H{
				"error":   "Forbidden",
				"message": "Insufficient permissions",
				"code":    "INSUFFICIENT_PERMISSIONS",
				"required_roles": requiredRoles,
			})
			c.Abort()
			return
		}

		c.Next()
	}
}

// APIKeyAuth creates an API key authentication middleware
func APIKeyAuth(validAPIKeys map[string]string) gin.HandlerFunc {
	return func(c *gin.Context) {
		apiKey := c.GetHeader("X-API-Key")
		if apiKey == "" {
			c.JSON(http.StatusUnauthorized, gin.H{
				"error":   "Unauthorized",
				"message": "API key is required",
				"code":    "API_KEY_MISSING",
			})
			c.Abort()
			return
		}

		// Validate API key
		clientName, exists := validAPIKeys[apiKey]
		if !exists {
			c.JSON(http.StatusUnauthorized, gin.H{
				"error":   "Unauthorized",
				"message": "Invalid API key",
				"code":    "INVALID_API_KEY",
			})
			c.Abort()
			return
		}

		// Store client information in context
		c.Set("api_client", clientName)
		c.Set("auth_method", "api_key")

		c.Next()
	}
}

// GenerateToken generates a new JWT token
func GenerateToken(userID, username, email string, roles []string, secretKey string, expiration time.Duration) (string, error) {
	now := time.Now()
	claims := &Claims{
		UserID:    userID,
		Username:  username,
		Email:     email,
		Roles:     roles,
		IssuedAt:  now.Unix(),
		ExpiresAt: now.Add(expiration).Unix(),
		RegisteredClaims: jwt.RegisteredClaims{
			Issuer:    "aiops-platform",
			Subject:   userID,
			IssuedAt:  jwt.NewNumericDate(now),
			ExpiresAt: jwt.NewNumericDate(now.Add(expiration)),
		},
	}

	token := jwt.NewWithClaims(jwt.SigningMethodHS256, claims)
	tokenString, err := token.SignedString([]byte(secretKey))
	if err != nil {
		return "", err
	}

	return tokenString, nil
}

// ValidateToken validates a JWT token and returns claims
func ValidateToken(tokenString, secretKey string) (*Claims, error) {
	token, err := jwt.ParseWithClaims(tokenString, &Claims{}, func(token *jwt.Token) (interface{}, error) {
		// Validate the signing method
		if _, ok := token.Method.(*jwt.SigningMethodHMAC); !ok {
			return nil, jwt.ErrSignatureInvalid
		}
		return []byte(secretKey), nil
	})

	if err != nil {
		return nil, err
	}

	if claims, ok := token.Claims.(*Claims); ok && token.Valid {
		return claims, nil
	}

	return nil, jwt.ErrTokenInvalidClaims
}

// RefreshToken generates a new token from an existing valid token
func RefreshToken(tokenString, secretKey string, newExpiration time.Duration) (string, error) {
	claims, err := ValidateToken(tokenString, secretKey)
	if err != nil {
		return "", err
	}

	// Generate new token with extended expiration
	return GenerateToken(claims.UserID, claims.Username, claims.Email, claims.Roles, secretKey, newExpiration)
}

// GetUserFromContext extracts user information from Gin context
func GetUserFromContext(c *gin.Context) (*Claims, error) {
	claims, exists := c.Get("claims")
	if !exists {
		return nil, jwt.ErrTokenNotValidYet
	}

	userClaims, ok := claims.(*Claims)
	if !ok {
		return nil, jwt.ErrTokenInvalidClaims
	}

	return userClaims, nil
}

// IsAdmin checks if the current user has admin role
func IsAdmin(c *gin.Context) bool {
	roles, exists := c.Get("roles")
	if !exists {
		return false
	}

	userRoles, ok := roles.([]string)
	if !ok {
		return false
	}

	for _, role := range userRoles {
		if role == "admin" {
			return true
		}
	}

	return false
}

// HasRole checks if the current user has a specific role
func HasRole(c *gin.Context, requiredRole string) bool {
	roles, exists := c.Get("roles")
	if !exists {
		return false
	}

	userRoles, ok := roles.([]string)
	if !ok {
		return false
	}

	for _, role := range userRoles {
		if role == requiredRole || role == "admin" {
			return true
		}
	}

	return false
}

// WebSocketUpgrader for WebSocket connections
var WebSocketUpgrader = websocket.Upgrader{
	ReadBufferSize:  1024,
	WriteBufferSize: 1024,
	CheckOrigin: func(r *http.Request) bool {
		// In production, implement proper origin checking
		return true
	},
}

// AuthenticateWebSocket authenticates WebSocket connections
func AuthenticateWebSocket(c *gin.Context, secretKey string) (*Claims, error) {
	// Get token from query parameter
	tokenString := c.Query("token")
	if tokenString == "" {
		return nil, jwt.ErrTokenNotValidYet
	}

	// Validate token
	claims, err := ValidateToken(tokenString, secretKey)
	if err != nil {
		return nil, err
	}

	// Check if token is expired
	if time.Now().Unix() > claims.ExpiresAt {
		return nil, jwt.ErrTokenExpired
	}

	return claims, nil
} 