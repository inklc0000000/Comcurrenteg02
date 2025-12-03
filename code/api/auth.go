package main

import (
	"context"
	"encoding/json"
	"errors"
	"fmt"
	"log"
	"net/http"
	"os"
	"strings"
	"time"

	"github.com/golang-jwt/jwt/v5"
	"go.mongodb.org/mongo-driver/bson"
	"go.mongodb.org/mongo-driver/mongo"
)

type contextKey string

const (
	ctxUserID contextKey = "user_id"
	ctxRole   contextKey = "role"
)

var jwtSecret []byte

// Se llama desde main() para inicializar la clave
func initJWT() {
	secret := os.Getenv("JWT_SECRET")
	if secret == "" {
		// Valor por defecto solo para desarrollo
		secret = "dev-secret-change-me"
	}
	jwtSecret = []byte(secret)
}

type JWTClaims struct {
	UserID int    `json:"user_id"`
	Role   string `json:"role"`
	jwt.RegisteredClaims
}

func generateToken(userID int, role string) (string, error) {
	now := time.Now()
	claims := JWTClaims{
		UserID: userID,
		Role:   role,
		RegisteredClaims: jwt.RegisteredClaims{
			IssuedAt:  jwt.NewNumericDate(now),
			ExpiresAt: jwt.NewNumericDate(now.Add(12 * time.Hour)),
		},
	}

	token := jwt.NewWithClaims(jwt.SigningMethodHS256, claims)
	return token.SignedString(jwtSecret)
}

func parseToken(tokenStr string) (*JWTClaims, error) {
	tok, err := jwt.ParseWithClaims(tokenStr, &JWTClaims{}, func(t *jwt.Token) (interface{}, error) {
		if _, ok := t.Method.(*jwt.SigningMethodHMAC); !ok {
			return nil, errors.New("algoritmo inesperado")
		}
		return jwtSecret, nil
	})
	if err != nil {
		return nil, err
	}

	claims, ok := tok.Claims.(*JWTClaims)
	if !ok || !tok.Valid {
		return nil, errors.New("token inválido")
	}
	return claims, nil
}

func authMiddleware(next http.Handler) http.Handler {
	return http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		// Rutas sin JWT: health, login y websockets (se validan dentro)
		if r.URL.Path == "/health" || r.URL.Path == "/auth/login" || strings.HasPrefix(r.URL.Path, "/ws/") {
			next.ServeHTTP(w, r)
			return
		}

		authHeader := r.Header.Get("Authorization")
		if authHeader == "" || !strings.HasPrefix(authHeader, "Bearer ") {
			http.Error(w, "token JWT requerido", http.StatusUnauthorized)
			return
		}

		raw := strings.TrimSpace(strings.TrimPrefix(authHeader, "Bearer"))
		claims, err := parseToken(raw)
		if err != nil {
			log.Printf("[AUTH] token inválido: %v\n", err)
			http.Error(w, "token inválido", http.StatusUnauthorized)
			return
		}

		ctx := context.WithValue(r.Context(), ctxUserID, claims.UserID)
		ctx = context.WithValue(ctx, ctxRole, claims.Role)
		next.ServeHTTP(w, r.WithContext(ctx))
	})
}

// Para envolver endpoints solo admin (cluster, métricas, etc.)
func adminOnly(next http.Handler) http.Handler {
	return http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		role, _ := r.Context().Value(ctxRole).(string)
		if role != "admin" {
			http.Error(w, "solo admin", http.StatusForbidden)
			return
		}
		next.ServeHTTP(w, r)
	})
}

// Helpers por si en el futuro querés validar user_id vs path, etc.
func getTokenUserID(ctx context.Context) (int, bool) {
	v := ctx.Value(ctxUserID)
	id, ok := v.(int)
	return id, ok
}

func getTokenRole(ctx context.Context) (string, bool) {
	v := ctx.Value(ctxRole)
	role, ok := v.(string)
	return role, ok
}

// -----------------------------
// Handler de login
// -----------------------------

type LoginRequest struct {
	UserID   int    `json:"user_id"`
	AdminKey string `json:"admin_key"`
}

type LoginResponse struct {
	Token  string `json:"token"`
	UserID int    `json:"user_id"`
	Role   string `json:"role"`
}

func loginHandler() http.HandlerFunc {
	// Clave de admin por env var
	adminKeyEnv := os.Getenv("ADMIN_KEY")
	if adminKeyEnv == "" {
		adminKeyEnv = "admin-key-dev"
	}

	return func(w http.ResponseWriter, r *http.Request) {
		if r.Method != http.MethodPost {
			http.Error(w, "solo POST", http.StatusMethodNotAllowed)
			return
		}

		var req LoginRequest
		if err := json.NewDecoder(r.Body).Decode(&req); err != nil {
			http.Error(w, "JSON inválido", http.StatusBadRequest)
			return
		}

		role := "user"
		if req.AdminKey != "" && req.AdminKey == adminKeyEnv {
			role = "admin"
		}

		// Validar usuario solo si es rol user
		if role == "user" {
			if req.UserID <= 0 {
				http.Error(w, "user_id debe ser entero positivo", http.StatusBadRequest)
				return
			}

			if mongoUsersColl != nil {
				ctx, cancel := context.WithTimeout(r.Context(), 2*time.Second)
				defer cancel()

				err := mongoUsersColl.FindOne(ctx, bson.M{"user_id": req.UserID}).Err()
				if err == mongo.ErrNoDocuments {
					http.Error(w, "usuario no encontrado", http.StatusNotFound)
					return
				}
				if err != nil {
					log.Printf("[AUTH] error validando usuario: %v\n", err)
					http.Error(w, "error interno", http.StatusInternalServerError)
					return
				}
			}
		}

		token, err := generateToken(req.UserID, role)
		if err != nil {
			log.Printf("[AUTH] error generando token: %v\n", err)
			http.Error(w, "error interno", http.StatusInternalServerError)
			return
		}
		// Auditoría de login
		var uidPtr *int
		if role == "user" && req.UserID > 0 {
			uid := req.UserID
			uidPtr = &uid
		}
		loginKind := "de usuario"
		if role == "admin" {
			loginKind = "de administrador"
		}

		detail := fmt.Sprintf(
			"Login %s. Paso 1: la API recibe POST /auth/login con user_id=%d. "+
				"Paso 2: si es rol=user valida que el user_id exista en MongoDB (colección users). "+
				"Paso 3: genera un JWT firmado con rol=%s que el frontend usará en las siguientes peticiones.",
			loginKind, req.UserID, role,
		)
		logAuditEvent(r.Context(), "login", uidPtr, detail)
		resp := LoginResponse{
			Token:  token,
			UserID: req.UserID,
			Role:   role,
		}

		w.Header().Set("Content-Type", "application/json")
		if err := json.NewEncoder(w).Encode(resp); err != nil {
			log.Println("[AUTH] error codificando respuesta de login:", err)
		}
	}
}
