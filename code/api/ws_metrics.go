package main

import (
	"log"
	"net/http"
	"runtime"
	"time"

	"github.com/gorilla/websocket"
	"github.com/shirou/gopsutil/v3/cpu"
)

var metricsWSUpgrader = websocket.Upgrader{
	ReadBufferSize:  1024,
	WriteBufferSize: 1024,
	// Control sencillo de origen: solo aceptamos el frontend Vite
	CheckOrigin: func(r *http.Request) bool {
		origin := r.Header.Get("Origin")
		return origin == "http://localhost:5173" || origin == ""
	},
}

// Cuenta los nodos configurados para mostrarlos en la métrica
func countNodesFromConfig(cfg nodeConfig) int {
	n := 0
	if cfg.Node1Addr != "" {
		n++
	}
	if cfg.Node2Addr != "" {
		n++
	}
	if cfg.Node3Addr != "" {
		n++
	}
	return n
}

// Enviamos por WebSocket exactamente la misma estructura que usa /metrics/system:
// SystemMetricsResponse (cpu_percent, num_goroutines, num_nodes, timestamp)
func metricsWSHandler(cfg nodeConfig) http.HandlerFunc {
	return func(w http.ResponseWriter, r *http.Request) {
		// JWT recibido en query string: ?token=...
		tokenStr := r.URL.Query().Get("token")
		if tokenStr == "" {
			http.Error(w, "token requerido", http.StatusUnauthorized)
			return
		}

		claims, err := parseToken(tokenStr)
		if err != nil {
			log.Printf("[WS] token inválido: %v\n", err)
			http.Error(w, "token inválido", http.StatusUnauthorized)
			return
		}
		if claims.Role != "admin" {
			http.Error(w, "solo admin", http.StatusForbidden)
			return
		}

		conn, err := metricsWSUpgrader.Upgrade(w, r, nil)
		if err != nil {
			log.Printf("[WS] error en upgrade: %v\n", err)
			return
		}
		defer conn.Close()

		log.Printf("[WS] conexión /ws/metrics abierta para admin user_id=%d\n", claims.UserID)

		ticker := time.NewTicker(3 * time.Second)
		defer ticker.Stop()

		for {
			select {
			case <-ticker.C:
				// Snapshot de métricas del sistema (similar a /metrics/system)
				cpuPercents, err := cpu.Percent(0, false)
				var cpuPercent float64
				if err != nil || len(cpuPercents) == 0 {
					cpuPercent = 0
				} else {
					cpuPercent = cpuPercents[0]
				}
				resp := SystemMetricsResponse{
					CPUPercent:    cpuPercent,
					NumGoroutines: runtime.NumGoroutine(),
					NumNodes:      countNodesFromConfig(cfg),
					Timestamp:     time.Now(),
				}

				if err := conn.WriteJSON(resp); err != nil {
					log.Printf("[WS] error enviando métricas: %v\n", err)
					return
				}
			}
		}
	}
}
