package main

import (
	"encoding/json"
	"fmt"
	"log"
	"net"
	"os"
	"strconv"
	"time"
)

// Reutilizamos desde cf_user-based.go:
//
//  - Constantes:
//      KNeighbors, MinOverlap, TopN, NumWorkers, RatingsPath, ColdThresh
//
//  - Tipos:
//      type Pair struct { ID int; Val float64 }
//      type Sim  struct { V int; S float64; Cnt int }
//
//  - Funciones:
//      readRatings(path string, max int) [][3]int
//      buildUserRaw(ratings [][3]int) map[int]map[int]int
//      userMeans(userRaw map[int]map[int]int) map[int]float64
//      buildUserNorm(userRaw map[int]map[int]int, means map[int]float64) map[int]map[int]float64
//      buildInverted(userRaw map[int]map[int]int) map[int][]int
//      globalTopMovies(userRaw map[int]map[int]int) []Pair
//      topKNeighbors(userNorm map[int]map[int]float64, inv map[int][]int) map[int][]Sim
//      predictForUsers(userRaw map[int]map[int]int, means map[int]float64, neigh map[int][]Sim, global []Pair) map[int][]Pair

// ---------------------------------------------------------------------
// Estado del modelo cargado en memoria del nodo
// ---------------------------------------------------------------------

var (
	modelUserRaw   map[int]map[int]int
	modelMeans     map[int]float64
	modelNeighbors map[int][]Sim
	modelGlobal    []Pair
)

var (
	shardMod int = 1 // 1 = sin partición
	shardRem int = 0
)

// initModel construye el modelo una sola vez al arrancar el nodo.
func initModel(maxRatings int) {
	start := time.Now()
	fmt.Println("[NODE] Cargando ratings desde", RatingsPath, "...")
	ratings := readRatings(RatingsPath, maxRatings)
	fmt.Printf("[NODE] Ratings cargados: %d\n", len(ratings))

	// Mismas etapas que en el main para los experimentos
	modelUserRaw = buildUserRaw(ratings)
	modelMeans = userMeans(modelUserRaw)
	userNorm := buildUserNorm(modelUserRaw, modelMeans)
	inv := buildInverted(modelUserRaw)
	modelGlobal = globalTopMovies(modelUserRaw)
	modelNeighbors = topKNeighbors(userNorm, inv)

	fmt.Printf("[NODE] Modelo construido en %.2fs\n", time.Since(start).Seconds())
}

// ---------------------------------------------------------------------
// Protocolo TCP (API <-> Nodo)
// ---------------------------------------------------------------------

// Request que envía la API al nodo
type Request struct {
	UserID int `json:"user_id"`
	TopN   int `json:"top_n"`
}

// Item recomendado
type Item struct {
	MovieID int     `json:"movie_id"`
	Score   float64 `json:"score"`
}

// Response que devuelve el nodo a la API
type Response struct {
	UserID int    `json:"user_id"`
	Items  []Item `json:"items"`
}

// recommendForUser usa la lógica de CF para un solo usuario,
// aplicando sharding por ítems según shardMod / shardRem.
func recommendForUser(userID, topN int) []Item {
	// Usuario no conocido en el modelo → devolver top global del shard
	if _, ok := modelUserRaw[userID]; !ok {
		items := make([]Item, 0, topN)
		for _, p := range modelGlobal {
			if shardMod > 1 && (p.ID%shardMod) != shardRem {
				continue
			}
			items = append(items, Item{
				MovieID: p.ID,
				Score:   p.Val,
			})
			if topN > 0 && len(items) >= topN {
				break
			}
		}
		return items
	}

	// Subconjunto de datos solo para este usuario
	userRawSub := map[int]map[int]int{
		userID: modelUserRaw[userID],
	}
	meansSub := map[int]float64{
		userID: modelMeans[userID],
	}
	neighSub := map[int][]Sim{
		userID: modelNeighbors[userID],
	}

	// Reutilizamos la función concurrente existente
	res := predictForUsers(userRawSub, meansSub, neighSub, modelGlobal)
	pairs, ok := res[userID]
	if !ok || len(pairs) == 0 {
		// fallback: top global del shard
		items := make([]Item, 0, topN)
		for _, p := range modelGlobal {
			if shardMod > 1 && (p.ID%shardMod) != shardRem {
				continue
			}
			items = append(items, Item{
				MovieID: p.ID,
				Score:   p.Val,
			})
			if topN > 0 && len(items) >= topN {
				break
			}
		}
		return items
	}

	// Filtrar por shard antes de aplicar TopN
	filtered := make([]Pair, 0, len(pairs))
	for _, p := range pairs {
		if shardMod > 1 && (p.ID%shardMod) != shardRem {
			continue
		}
		filtered = append(filtered, p)
	}

	if topN > 0 && len(filtered) > topN {
		filtered = filtered[:topN]
	}

	items := make([]Item, len(filtered))
	for i, p := range filtered {
		items[i] = Item{
			MovieID: p.ID,
			Score:   p.Val,
		}
	}
	return items
}

// handleConn atiende una conexión TCP (una request por conexión).
func handleConn(conn net.Conn) {
	defer conn.Close()

	dec := json.NewDecoder(conn)
	enc := json.NewEncoder(conn)

	var req Request
	if err := dec.Decode(&req); err != nil {
		log.Println("[NODE] error decodificando request:", err)
		return
	}

	if req.TopN <= 0 {
		req.TopN = 20
	}

	items := recommendForUser(req.UserID, req.TopN)
	resp := Response{
		UserID: req.UserID,
		Items:  items,
	}

	if err := enc.Encode(&resp); err != nil {
		log.Println("[NODE] error enviando respuesta:", err)
		return
	}
}

// ---------------------------------------------------------------------
// main del nodo ML
// ---------------------------------------------------------------------

func main() {
	// Leer configuración desde variables de entorno
	port := os.Getenv("PORT")
	if port == "" {
		port = "9001"
	}
	nodeID := os.Getenv("NODE_ID")
	if nodeID == "" {
		nodeID = "1"
	}

	// Configuración de sharding por ítems
	if v := os.Getenv("SHARD_MOD"); v != "" {
		if n, err := strconv.Atoi(v); err == nil && n > 0 {
			shardMod = n
		}
	}
	if v := os.Getenv("SHARD_REM"); v != "" {
		if n, err := strconv.Atoi(v); err == nil && n >= 0 {
			shardRem = n
		}
	}

	// Se puede ajustar este límite según las pruebas de rendimiento previas.
	// Para MovieLens 1M usamos 1000000, para 10M solo se usa hasta la mitad por el alto costo de computo y la PC se cuelga.
	maxRatings := 1000000

	fmt.Printf("[NODE %s] Inicializando modelo...\n", nodeID)
	initModel(maxRatings)
	fmt.Printf("[NODE %s] Modelo listo. Escuchando en puerto %s\n", nodeID, port)

	ln, err := net.Listen("tcp", ":"+port)
	if err != nil {
		log.Fatal("[NODE] error al escuchar:", err)
	}
	defer ln.Close()

	for {
		conn, err := ln.Accept()
		if err != nil {
			log.Println("[NODE] error aceptando conexión:", err)
			continue
		}
		go handleConn(conn)
	}
}
