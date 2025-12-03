// main.go
package main

import (
	"context"
	"encoding/json"
	"fmt"
	"log"
	"net"
	"net/http"
	"os"
	"runtime"
	"sort"
	"strconv"
	"strings"
	"sync"
	"time"

	"go.mongodb.org/mongo-driver/bson"
	"go.mongodb.org/mongo-driver/bson/primitive"
	"go.mongodb.org/mongo-driver/mongo"
	"go.mongodb.org/mongo-driver/mongo/options"

	redis "github.com/redis/go-redis/v9"
	"github.com/shirou/gopsutil/v3/cpu"
)

// -------------------------------------------------------------------
// Tipos compartidos con el nodo (mismo JSON que usa node.go)
// -------------------------------------------------------------------

type nodeRequest struct {
	UserID int `json:"user_id"`
	TopN   int `json:"top_n"`
}

type Item struct {
	MovieID int      `json:"movie_id" bson:"movie_id"`
	Score   float64  `json:"score" bson:"score"`
	Nodes   []string `json:"nodes,omitempty" bson:"nodes,omitempty"`
}

type NodeResponse struct {
	UserID int    `json:"user_id"`
	Items  []Item `json:"items"`
}

// -------------------------------------------------------------------
// MongoDB
// -------------------------------------------------------------------

var (
	mongoClient     *mongo.Client
	mongoRecColl    *mongo.Collection // historial de recomendaciones
	mongoMoviesColl *mongo.Collection // catálogo de películas
	mongoUsersColl  *mongo.Collection // usuarios (MovieLens 1M)
	mongoAuditColl  *mongo.Collection // eventos de auditoría (bitácora)
	redisClient     *redis.Client     // caché de recomendaciones
)

// Documento de historial
type RecommendationRecord struct {
	UserID   int       `bson:"user_id" json:"user_id"`
	TopN     int       `bson:"top_n" json:"top_n"`
	Items    []Item    `bson:"items" json:"items"`
	NodeAddr string    `bson:"node_addr" json:"node_addr"`
	Created  time.Time `bson:"created_at" json:"created_at"`
}

// Documento de película en Mongo
type MovieDoc struct {
	MovieID int    `bson:"movie_id"`
	Title   string `bson:"title"`
	Genres  string `bson:"genres"` // "Comedy|Drama|..."
}

// Documento de usuario en Mongo (MovieLens 10M)
type UserDoc struct {
	UserID     int    `bson:"user_id" json:"user_id"`
	Name       string `bson:"name" json:"name"` // NUEVO, CREADO SINTETICAMENTE
	Gender     string `bson:"gender" json:"gender"`
	Age        int    `bson:"age" json:"age"`
	Occupation int    `bson:"occupation" json:"occupation"`
	ZipCode    string `bson:"zip_code" json:"zip_code"`
}

// Evento de auditoría para bitácora end-to-end
type AuditEvent struct {
	ID        primitive.ObjectID `bson:"_id,omitempty" json:"id"`
	Timestamp time.Time          `bson:"timestamp" json:"timestamp"`
	Type      string             `bson:"type" json:"type"`                 // "login", "recommendation", etc.
	UserID    *int               `bson:"user_id,omitempty" json:"user_id"` // opcional
	Detail    string             `bson:"detail" json:"detail"`             // descripción en texto
}

// logAuditEvent inserta un evento de auditoría en Mongo (si está configurado)
func logAuditEvent(ctx context.Context, typ string, userID *int, detail string) {
	if mongoAuditColl == nil {
		// Sin Mongo para auditoría: solo logueamos en consola
		log.Printf("[AUDIT:NO-MONGO] type=%s user_id=%v detail=%s\n", typ, userID, detail)
		return
	}

	evt := AuditEvent{
		Timestamp: time.Now(),
		Type:      typ,
		UserID:    userID,
		Detail:    detail,
	}

	ctxIns, cancel := context.WithTimeout(ctx, 2*time.Second)
	defer cancel()

	if _, err := mongoAuditColl.InsertOne(ctxIns, evt); err != nil {
		log.Printf("[AUDIT] error insertando evento: %v\n", err)
	}
}

// Respuesta enriquecida para HTTP
type MovieWithDetails struct {
	MovieID int      `json:"movie_id"`
	Title   string   `json:"title"`
	Genres  []string `json:"genres"`
	Score   float64  `json:"score"`
	Nodes   []string `json:"nodes,omitempty"`
}

type RecommendHTTPResponse struct {
	UserID int                `json:"user_id"`
	Items  []MovieWithDetails `json:"items"`
}

// -------------------------------------------------------------------
// Tipos adicionales para /cluster/status y /metrics/benchmark
// -------------------------------------------------------------------

// Estado de un nodo individual del cluster
type ClusterNodeStatus struct {
	Addr          string   `json:"addr"`
	Up            bool     `json:"up"`
	LastLatencyMS *float64 `json:"last_latency_ms,omitempty"`
}

// Respuesta de /cluster/status
type ClusterStatusResponse struct {
	Nodes     []ClusterNodeStatus `json:"nodes"`
	CheckedAt time.Time           `json:"checked_at"`
}

// Request para /metrics/benchmark
type MetricsBenchmarkRequest struct {
	Requests    int `json:"requests"`
	Concurrency int `json:"concurrency"`
	UserID      int `json:"user_id"`
	TopN        int `json:"top_n"`
}

// Response de /metrics/benchmark
type MetricsBenchmarkResponse struct {
	TotalRequests int     `json:"total_requests"`
	OK            int     `json:"ok"`
	Errors        int     `json:"errors"`
	TotalTimeSec  float64 `json:"total_time_sec"`
	LatencyAvgMS  float64 `json:"latency_avg_ms"`
	LatencyMinMS  float64 `json:"latency_min_ms"`
	LatencyMaxMS  float64 `json:"latency_max_ms"`
	RPS           float64 `json:"rps"`
}

type RedisMetricsResponse struct {
	DBSize           int64 `json:"db_size"`
	UsedMemoryBytes  int64 `json:"used_memory_bytes"`
	ConnectedClients int64 `json:"connected_clients"`
	KeyspaceHits     int64 `json:"keyspace_hits"`
	KeyspaceMisses   int64 `json:"keyspace_misses"`
}

// -------------------------------------------------------------------
// Tipo para /metrics/system
// -------------------------------------------------------------------

type SystemMetricsResponse struct {
	CPUPercent    float64   `json:"cpu_percent"`
	NumGoroutines int       `json:"num_goroutines"`
	NumNodes      int       `json:"num_nodes"`
	Timestamp     time.Time `json:"timestamp"`
}

func initMongo(ctx context.Context) error {
	uri := os.Getenv("MONGO_URI")
	if uri == "" {
		uri = "mongodb://localhost:27017"
	}
	dbName := os.Getenv("MONGO_DB")
	if dbName == "" {
		dbName = "recomendador"
	}

	recCollName := os.Getenv("MONGO_COLL")
	if recCollName == "" {
		recCollName = "recommendations"
	}
	moviesCollName := os.Getenv("MONGO_MOVIES_COLL")
	if moviesCollName == "" {
		moviesCollName = "movies"
	}
	usersCollName := os.Getenv("MONGO_USERS_COLL")
	if usersCollName == "" {
		usersCollName = "users"
	}
	auditCollName := os.Getenv("MONGO_AUDIT_COLL") // NUEVO
	if auditCollName == "" {
		auditCollName = "audit_events"
	}

	client, err := mongo.Connect(ctx, options.Client().ApplyURI(uri))
	if err != nil {
		return fmt.Errorf("error conectando a Mongo: %w", err)
	}
	if err := client.Ping(ctx, nil); err != nil {
		return fmt.Errorf("error haciendo ping a Mongo: %w", err)
	}

	mongoClient = client
	db := client.Database(dbName)
	mongoRecColl = db.Collection(recCollName)
	mongoMoviesColl = db.Collection(moviesCollName)
	mongoUsersColl = db.Collection(usersCollName)
	mongoAuditColl = db.Collection(auditCollName)

	log.Printf("[API] Conectado a MongoDB en %s, DB=%s, RecColl=%s, MoviesColl=%s, UsersColl=%s, AuditColl=%s\n",
		uri, dbName, recCollName, moviesCollName, usersCollName, auditCollName)
	return nil
}

func initRedis(ctx context.Context) error {
	addr := os.Getenv("REDIS_ADDR")
	if addr == "" {
		addr = "localhost:6379"
	}

	password := os.Getenv("REDIS_PASSWORD") // normalmente vacío en la fase de desarrollo (se pondria password en caso de llevarlo a produccion)
	dbIndex := 0
	if v := os.Getenv("REDIS_DB"); v != "" {
		if n, err := strconv.Atoi(v); err == nil && n >= 0 {
			dbIndex = n
		}
	}

	redisClient = redis.NewClient(&redis.Options{
		Addr:     addr,
		Password: password,
		DB:       dbIndex,
	})

	if err := redisClient.Ping(ctx).Err(); err != nil {
		// Si falla, dejamos redisClient en nil y no matamos la API
		redisClient = nil
		return fmt.Errorf("error conectando a Redis en %s: %w", addr, err)
	}

	log.Printf("[API] Conectado a Redis en %s (db=%d)\n", addr, dbIndex)
	return nil
}

// -------------------------------------------------------------------
// Configuración de nodos y utilidades
// -------------------------------------------------------------------

type nodeConfig struct {
	Node1Addr string
	Node2Addr string
	Node3Addr string
}

func loadNodeConfig() nodeConfig {
	cfg := nodeConfig{
		Node1Addr: os.Getenv("NODE1_ADDR"),
		Node2Addr: os.Getenv("NODE2_ADDR"),
		Node3Addr: os.Getenv("NODE3_ADDR"),
	}
	if cfg.Node1Addr == "" {
		cfg.Node1Addr = "localhost:9001"
	}
	if cfg.Node2Addr == "" {
		cfg.Node2Addr = "localhost:9002"
	}
	if cfg.Node3Addr == "" {
		cfg.Node3Addr = "localhost:9003"
	}
	return cfg
}

// llama a un nodo ML por TCP usando el protocolo JSON definido
func callNode(addr string, userID, topN int) (*NodeResponse, error) {
	conn, err := net.DialTimeout("tcp", addr, 5*time.Second)
	if err != nil {
		return nil, fmt.Errorf("error conectando al nodo %s: %w", addr, err)
	}
	defer conn.Close()

	enc := json.NewEncoder(conn)
	dec := json.NewDecoder(conn)

	req := nodeRequest{
		UserID: userID,
		TopN:   topN,
	}
	if err := enc.Encode(&req); err != nil {
		return nil, fmt.Errorf("error enviando request al nodo: %w", err)
	}

	var resp NodeResponse
	if err := dec.Decode(&resp); err != nil {
		return nil, fmt.Errorf("error leyendo respuesta del nodo: %w", err)
	}

	return &resp, nil
}

// llama a todos los nodos en paralelo y combina las respuestas
func callAllNodes(cfg nodeConfig, userID, topN int) ([]Item, error) {
	addrs := []string{cfg.Node1Addr, cfg.Node2Addr, cfg.Node3Addr}

	type nodeResult struct {
		addr string
		resp *NodeResponse
		err  error
	}

	ch := make(chan nodeResult, len(addrs))

	for _, addr := range addrs {
		addr := addr
		go func() {
			resp, err := callNode(addr, userID, topN)
			ch <- nodeResult{addr: addr, resp: resp, err: err}
		}()
	}

	var allItems []Item
	var hadSuccess bool

	for i := 0; i < len(addrs); i++ {
		r := <-ch
		if r.err != nil {
			log.Printf("[API] error llamando al nodo %s: %v\n", r.addr, r.err)
			continue
		}
		if r.resp == nil {
			continue
		}
		hadSuccess = true

		// etiquetamos cada ítem con el nodo que lo generó
		for _, it := range r.resp.Items {
			allItems = append(allItems, Item{
				MovieID: it.MovieID,
				Score:   it.Score,
				Nodes:   []string{r.addr},
			})
		}
	}

	if !hadSuccess {
		return nil, fmt.Errorf("ningún nodo pudo responder")
	}

	// combinar y ordenar globalmente
	if len(allItems) == 0 {
		return allItems, nil
	}

	// eliminar duplicados por movie_id, quedarnos con el mayor score
	// y acumular el conjunto de nodos que aportaron esa película
	bestByID := make(map[int]Item)
	for _, it := range allItems {
		if prev, ok := bestByID[it.MovieID]; !ok {
			bestByID[it.MovieID] = it
		} else {
			// unificar nodos
			nodeSet := make(map[string]struct{})
			for _, n := range prev.Nodes {
				nodeSet[n] = struct{}{}
			}
			for _, n := range it.Nodes {
				nodeSet[n] = struct{}{}
			}
			mergedNodes := make([]string, 0, len(nodeSet))
			for n := range nodeSet {
				mergedNodes = append(mergedNodes, n)
			}
			sort.Strings(mergedNodes)

			if it.Score > prev.Score {
				prev.Score = it.Score
			}
			prev.Nodes = mergedNodes
			bestByID[it.MovieID] = prev
		}
	}

	merged := make([]Item, 0, len(bestByID))
	for _, it := range bestByID {
		merged = append(merged, it)
	}

	sort.Slice(merged, func(i, j int) bool {
		return merged[i].Score > merged[j].Score
	})

	if topN > 0 && len(merged) > topN {
		merged = merged[:topN]
	}

	return merged, nil
}

// -------------------------------------------------------------------
// Enriquecimiento con detalles de películas
// -------------------------------------------------------------------

// -------------------------------------------------------------------
// Enriquecimiento con detalles de películas
// -------------------------------------------------------------------

// -------------------------------------------------------------------
// Enriquecimiento con detalles de películas
// -------------------------------------------------------------------

func enrichWithMovieDetails(ctx context.Context, items []Item) []MovieWithDetails {
	// si no hay colección de películas, devolvemos solo IDs y scores
	if mongoMoviesColl == nil || len(items) == 0 {
		out := make([]MovieWithDetails, len(items))
		for i, it := range items {
			out[i] = MovieWithDetails{
				MovieID: it.MovieID,
				Title:   "",
				Genres:  nil,
				Score:   it.Score,
				Nodes:   it.Nodes,
			}
		}
		return out
	}

	// construir lista de IDs
	ids := make([]int, 0, len(items))
	seen := make(map[int]struct{})
	for _, it := range items {
		if _, ok := seen[it.MovieID]; !ok {
			seen[it.MovieID] = struct{}{}
			ids = append(ids, it.MovieID)
		}
	}

	ctxQ, cancel := context.WithTimeout(ctx, 3*time.Second)
	defer cancel()

	filter := bson.M{"movie_id": bson.M{"$in": ids}}
	cur, err := mongoMoviesColl.Find(ctxQ, filter)
	if err != nil {
		log.Printf("[API] error consultando movies en Mongo: %v\n", err)
		// fallback: solo IDs y scores
		out := make([]MovieWithDetails, len(items))
		for i, it := range items {
			out[i] = MovieWithDetails{
				MovieID: it.MovieID,
				Score:   it.Score,
				Nodes:   it.Nodes,
			}
		}
		return out
	}
	defer cur.Close(ctxQ)

	// mapear movie_id -> doc
	movieMap := make(map[int]MovieDoc)
	for cur.Next(ctxQ) {
		var m MovieDoc
		if err := cur.Decode(&m); err != nil {
			log.Printf("[API] error decodificando movie doc: %v\n", err)
			continue
		}
		movieMap[m.MovieID] = m
	}

	out := make([]MovieWithDetails, len(items))
	for i, it := range items {
		md, ok := movieMap[it.MovieID]
		if !ok {
			out[i] = MovieWithDetails{
				MovieID: it.MovieID,
				Score:   it.Score,
				Nodes:   it.Nodes,
			}
			continue
		}
		var genres []string
		if md.Genres != "" && md.Genres != "(no genres listed)" {
			genres = strings.Split(md.Genres, "|")
		}
		out[i] = MovieWithDetails{
			MovieID: it.MovieID,
			Title:   md.Title,
			Genres:  genres,
			Score:   it.Score,
			Nodes:   it.Nodes,
		}
	}
	return out
}

// -------------------------------------------------------------------
// Handlers HTTP
// -------------------------------------------------------------------

func healthHandler(w http.ResponseWriter, r *http.Request) {
	w.WriteHeader(http.StatusOK)
	_, _ = w.Write([]byte("ok"))
}

// GET /recommend/{user_id}?top_n=20
func recommendHandler(cfg nodeConfig) http.HandlerFunc {
	return func(w http.ResponseWriter, r *http.Request) {
		path := strings.TrimPrefix(r.URL.Path, "/recommend/")
		if path == "" {
			http.Error(w, "user_id requerido en la ruta /recommend/{user_id}", http.StatusBadRequest)
			return
		}

		userID, err := strconv.Atoi(path)
		if err != nil || userID <= 0 {
			http.Error(w, "user_id debe ser un entero positivo", http.StatusBadRequest)
			return
		}

		// Validar que el usuario exista en Mongo (si la colección está configurada)
		if mongoUsersColl != nil {
			ctxUser, cancelUser := context.WithTimeout(r.Context(), 2*time.Second)
			err := mongoUsersColl.FindOne(ctxUser, bson.M{"user_id": userID}).Err()
			cancelUser()

			if err == mongo.ErrNoDocuments {
				http.Error(w, "usuario no encontrado en la base de datos", http.StatusNotFound)
				return
			}
			if err != nil {
				log.Printf("[API] error consultando usuario en Mongo: %v\n", err)
				http.Error(w, "error interno consultando usuario", http.StatusInternalServerError)
				return
			}
		}

		topN := 20
		if qs := r.URL.Query().Get("top_n"); qs != "" {
			if v, err := strconv.Atoi(qs); err == nil && v > 0 {
				topN = v
			}
		}

		// --- 1) Intentar cache Redis antes de llamar a los nodos ---
		cacheKey := fmt.Sprintf("rec:%d:%d", userID, topN)

		if redisClient != nil {
			ctxCache, cancelCache := context.WithTimeout(r.Context(), 80*time.Millisecond)
			cached, err := redisClient.Get(ctxCache, cacheKey).Result()
			cancelCache()

			if err == nil {
				log.Printf("[API] cache HIT Redis para key=%s\n", cacheKey)

				// Auditoría: recomendaciones servidas desde cache
				uid := userID
				detail := fmt.Sprintf(
					"Recomendaciones (cache HIT). Paso 1: la API recibe GET /recommend/%d?top_n=%d y valida el JWT del usuario. "+
						"Paso 2: consulta Redis con key=%s y encuentra un Top-N precalculado. "+
						"Paso 3: devuelve ese Top-N directamente al frontend sin llamar al cluster de nodos ML ni escribir nuevo historial en MongoDB.",
					userID, topN, cacheKey,
				)
				logAuditEvent(r.Context(), "recommendation", &uid, detail)

				w.Header().Set("Content-Type", "application/json")
				_, _ = w.Write([]byte(cached))
				return
			} else if err != redis.Nil {
				// redis.Nil = clave no encontrada; otros errores los logueamos
				log.Printf("[API] error leyendo caché Redis: %v\n", err)
			}
		}

		// --- 2) Flujo normal: llamar a los nodos ML ---
		start := time.Now()
		items, err := callAllNodes(cfg, userID, topN)
		if err != nil {
			log.Printf("[API] error llamando a nodos: %v\n", err)
			http.Error(w, "error interno al obtener recomendaciones", http.StatusInternalServerError)
			return
		}

		log.Printf(
			"[API] user_id=%d top_n=%d nodos=[%s,%s,%s] tiempo=%.3fs\n",
			userID, topN,
			cfg.Node1Addr, cfg.Node2Addr, cfg.Node3Addr,
			time.Since(start).Seconds(),
		)

		// --- 3) Guardar historial en Mongo ---
		if mongoRecColl != nil {
			ctxSave, cancelSave := context.WithTimeout(r.Context(), 3*time.Second)

			rec := RecommendationRecord{
				UserID:   userID,
				TopN:     topN,
				Items:    items,
				NodeAddr: "cluster",
				Created:  time.Now(),
			}

			if _, err := mongoRecColl.InsertOne(ctxSave, rec); err != nil {
				log.Printf("[API] error guardando recomendación en Mongo: %v\n", err)
			}
			cancelSave()
		}

		// --- 4) Enriquecer con títulos y géneros desde Mongo (movies) ---
		ctxEnrich, cancelEnrich := context.WithTimeout(r.Context(), 3*time.Second)
		enriched := enrichWithMovieDetails(ctxEnrich, items)
		cancelEnrich()

		httpResp := RecommendHTTPResponse{
			UserID: userID,
			Items:  enriched,
		}

		// --- 5) Serializar una sola vez para Redis + respuesta HTTP ---
		respBytes, err := json.Marshal(httpResp)
		if err != nil {
			log.Println("[API] error serializando respuesta HTTP:", err)
			http.Error(w, "error interno al preparar respuesta", http.StatusInternalServerError)
			return
		}

		// --- 6) Guardar en Redis (cache miss) ---
		if redisClient != nil {
			ctxSet, cancelSet := context.WithTimeout(r.Context(), 80*time.Millisecond)
			if err := redisClient.Set(ctxSet, cacheKey, respBytes, 10*time.Minute).Err(); err != nil {
				log.Printf("[API] error guardando en Redis: %v\n", err)
			}
			cancelSet()
		}

		// Auditoría: recomendaciones calculadas vía nodos ML y sharding por ítems
		uid := userID
		detail := fmt.Sprintf(
			"Recomendaciones (cache MISS). Paso 1: la API recibe GET /recommend/%d?top_n=%d y valida el JWT del usuario. "+
				"Paso 2: consulta Redis con key=%s y no encuentra resultados. "+
				"Paso 3: realiza fan-out al cluster de nodos ML [%s,%s,%s]. Cada nodo almacena solo un shard de las películas "+
				"(partición por ítems usando movie_id %% num_shards) y calcula el Top-N parcial sobre su shard. "+
				"Paso 4: la API combina los Top-N parciales en un Top-N global, ordena por score y enriquece los resultados con títulos y géneros desde MongoDB (colección movies). "+
				"Paso 5: guarda la recomendación en MongoDB (colección recommendations) como historial del usuario y devuelve el Top-N global al frontend.",
			userID, topN, cacheKey, cfg.Node1Addr, cfg.Node2Addr, cfg.Node3Addr,
		)
		logAuditEvent(r.Context(), "recommendation", &uid, detail)

		// --- 7) Enviar respuesta al cliente ---
		w.Header().Set("Content-Type", "application/json")
		if _, err := w.Write(respBytes); err != nil {
			log.Println("[API] error enviando respuesta HTTP:", err)
		}
	}
}

// GET /history/{user_id}
func historyHandler() http.HandlerFunc {
	return func(w http.ResponseWriter, r *http.Request) {
		if mongoRecColl == nil {
			http.Error(w, "MongoDB no está configurado", http.StatusServiceUnavailable)
			return
		}

		path := strings.TrimPrefix(r.URL.Path, "/history/")
		if path == "" {
			http.Error(w, "user_id requerido en la ruta /history/{user_id}", http.StatusBadRequest)
			return
		}
		userID, err := strconv.Atoi(path)
		if err != nil || userID <= 0 {
			http.Error(w, "user_id debe ser un entero positivo", http.StatusBadRequest)
			return
		}

		ctx, cancel := context.WithTimeout(r.Context(), 3*time.Second)
		defer cancel()

		filter := bson.M{"user_id": userID}
		opts := options.Find().SetSort(bson.D{{Key: "created_at", Value: -1}}).SetLimit(10)

		cur, err := mongoRecColl.Find(ctx, filter, opts)
		if err != nil {
			log.Printf("[API] error consultando historial en Mongo: %v\n", err)
			http.Error(w, "error interno consultando historial", http.StatusInternalServerError)
			return
		}
		defer cur.Close(ctx)

		var records []RecommendationRecord
		if err := cur.All(ctx, &records); err != nil {
			log.Printf("[API] error leyendo cursor de Mongo: %v\n", err)
			http.Error(w, "error interno consultando historial", http.StatusInternalServerError)
			return
		}

		// Asegurar que no se encodee como null
		if records == nil {
			records = []RecommendationRecord{}
		}
		// Auditoría: consulta de historial
		uid := userID
		detail := fmt.Sprintf(
			"Consulta de historial. Paso 1: la API recibe GET /history/%d y valida el JWT del usuario. "+
				"Paso 2: lee en MongoDB (colección recommendations) las últimas %d recomendaciones del usuario, ordenadas por fecha descendente. "+
				"Paso 3: devuelve la lista de recomendaciones al frontend.",
			userID, len(records),
		)
		logAuditEvent(r.Context(), "history", &uid, detail)
		w.Header().Set("Content-Type", "application/json")
		if err := json.NewEncoder(w).Encode(records); err != nil {
			log.Println("[API] error codificando historial HTTP:", err)
		}
	}
}

// GET /audit/latest?limit=50
// Devuelve los últimos eventos de auditoría, ordenados por timestamp desc.
func latestAuditHandler() http.HandlerFunc {
	return func(w http.ResponseWriter, r *http.Request) {
		if mongoAuditColl == nil {
			http.Error(w, "auditoría no configurada", http.StatusServiceUnavailable)
			return
		}

		limit := int64(50)
		if qs := r.URL.Query().Get("limit"); qs != "" {
			if v, err := strconv.Atoi(qs); err == nil && v > 0 {
				limit = int64(v)
			}
		}

		ctx, cancel := context.WithTimeout(r.Context(), 3*time.Second)
		defer cancel()

		opts := options.Find().
			SetSort(bson.D{{Key: "timestamp", Value: -1}}).
			SetLimit(limit)

		cur, err := mongoAuditColl.Find(ctx, bson.D{}, opts)
		if err != nil {
			log.Printf("[AUDIT] error consultando eventos: %v\n", err)
			http.Error(w, "error interno consultando auditoría", http.StatusInternalServerError)
			return
		}
		defer cur.Close(ctx)

		var events []AuditEvent
		if err := cur.All(ctx, &events); err != nil {
			log.Printf("[AUDIT] error leyendo cursor: %v\n", err)
			http.Error(w, "error interno leyendo auditoría", http.StatusInternalServerError)
			return
		}

		w.Header().Set("Content-Type", "application/json")
		if err := json.NewEncoder(w).Encode(events); err != nil {
			log.Printf("[AUDIT] error codificando respuesta: %v\n", err)
		}
	}
}

// GET /users/{user_id}
func userHandler() http.HandlerFunc {
	return func(w http.ResponseWriter, r *http.Request) {
		if mongoUsersColl == nil {
			http.Error(w, "MongoDB no está configurado para usuarios", http.StatusServiceUnavailable)
			return
		}

		path := strings.TrimPrefix(r.URL.Path, "/users/")
		if path == "" {
			http.Error(w, "user_id requerido en la ruta /users/{user_id}", http.StatusBadRequest)
			return
		}

		userID, err := strconv.Atoi(path)
		if err != nil || userID <= 0 {
			http.Error(w, "user_id debe ser un entero positivo", http.StatusBadRequest)
			return
		}

		ctx, cancel := context.WithTimeout(r.Context(), 3*time.Second)
		defer cancel()

		var user UserDoc
		err = mongoUsersColl.FindOne(ctx, bson.M{"user_id": userID}).Decode(&user)
		if err == mongo.ErrNoDocuments {
			http.Error(w, "usuario no encontrado", http.StatusNotFound)
			return
		}
		if err != nil {
			log.Printf("[API] error consultando usuario en Mongo: %v\n", err)
			http.Error(w, "error interno consultando usuario", http.StatusInternalServerError)
			return
		}

		w.Header().Set("Content-Type", "application/json")
		if err := json.NewEncoder(w).Encode(user); err != nil {
			log.Println("[API] error codificando usuario HTTP:", err)
		}
	}
}

// GET /cluster/status
// Opcional: query ?user_id=X para probar con ese usuario (por defecto 1)
func clusterStatusHandler(cfg nodeConfig) http.HandlerFunc {
	return func(w http.ResponseWriter, r *http.Request) {
		sampleUserID := 1
		if qs := r.URL.Query().Get("user_id"); qs != "" {
			if v, err := strconv.Atoi(qs); err == nil && v > 0 {
				sampleUserID = v
			}
		}
		topN := 1

		addrs := []string{cfg.Node1Addr, cfg.Node2Addr, cfg.Node3Addr}

		type nodeResult struct {
			addr      string
			up        bool
			latencyMS float64
		}

		ch := make(chan nodeResult, len(addrs))

		for _, addr := range addrs {
			addr := addr
			go func() {
				start := time.Now()
				_, err := callNode(addr, sampleUserID, topN)
				if err != nil {
					log.Printf("[API] error en cluster-status con nodo %s: %v\n", addr, err)
					ch <- nodeResult{addr: addr, up: false, latencyMS: 0}
					return
				}
				dur := time.Since(start)
				ch <- nodeResult{
					addr:      addr,
					up:        true,
					latencyMS: dur.Seconds() * 1000,
				}
			}()
		}

		var nodes []ClusterNodeStatus
		for i := 0; i < len(addrs); i++ {
			r := <-ch
			var latPtr *float64
			if r.up {
				lat := r.latencyMS
				latPtr = &lat
			}
			nodes = append(nodes, ClusterNodeStatus{
				Addr:          r.addr,
				Up:            r.up,
				LastLatencyMS: latPtr,
			})
		}

		resp := ClusterStatusResponse{
			Nodes:     nodes,
			CheckedAt: time.Now(),
		}

		w.Header().Set("Content-Type", "application/json")
		if err := json.NewEncoder(w).Encode(resp); err != nil {
			log.Println("[API] error codificando cluster status:", err)
		}
	}
}

// POST /metrics/benchmark
// Body JSON: { "requests": 100, "concurrency": 10, "user_id": 1, "top_n": 20 }
func metricsBenchmarkHandler(apiBaseURL string) http.HandlerFunc {
	return func(w http.ResponseWriter, r *http.Request) {
		if r.Method != http.MethodPost {
			http.Error(w, "método no permitido, use POST", http.StatusMethodNotAllowed)
			return
		}

		var req MetricsBenchmarkRequest
		if err := json.NewDecoder(r.Body).Decode(&req); err != nil {
			http.Error(w, "body JSON inválido", http.StatusBadRequest)
			return
		}

		// Valores por defecto y límites razonables
		if req.Requests <= 0 {
			req.Requests = 50
		}
		if req.Requests > 1000 {
			req.Requests = 1000
		}
		if req.Concurrency <= 0 {
			req.Concurrency = 10
		}
		if req.Concurrency > req.Requests {
			req.Concurrency = req.Requests
		}
		if req.TopN <= 0 {
			req.TopN = 20
		}
		if req.UserID <= 0 {
			req.UserID = 1
		}

		totalRequests := req.Requests
		concurrency := req.Concurrency

		type benchResult struct {
			ok       bool
			duration time.Duration
			err      error
			status   int
		}

		results := make([]benchResult, totalRequests)
		jobs := make(chan int)

		client := &http.Client{
			Timeout: 15 * time.Second,
		}

		// Reutilizamos el mismo Authorization que llegó a /metrics/benchmark
		authHeader := r.Header.Get("Authorization")

		var wg sync.WaitGroup
		startAll := time.Now()

		// Lanzar workers
		for wID := 0; wID < concurrency; wID++ {
			wg.Add(1)
			go func() {
				defer wg.Done()
				for i := range jobs {
					url := fmt.Sprintf("%s/recommend/%d?top_n=%d", apiBaseURL, req.UserID, req.TopN)

					httpReq, err := http.NewRequest(http.MethodGet, url, nil)
					if err != nil {
						results[i] = benchResult{ok: false, duration: 0, err: err, status: 0}
						continue
					}
					if authHeader != "" {
						httpReq.Header.Set("Authorization", authHeader)
					}

					startReq := time.Now()
					resp, err := client.Do(httpReq)
					dur := time.Since(startReq)

					res := benchResult{
						ok:       false,
						duration: dur,
						err:      err,
						status:   0,
					}

					if err == nil {
						res.status = resp.StatusCode
						if resp.StatusCode == http.StatusOK {
							res.ok = true
						}
						_ = resp.Body.Close()
					}

					results[i] = res
				}
			}()
		}

		// Enviar trabajos
		for i := 0; i < totalRequests; i++ {
			jobs <- i
		}
		close(jobs)

		wg.Wait()
		totalTime := time.Since(startAll)

		var (
			okCount  int
			errCount int
			sumDur   time.Duration
			minDur   time.Duration
			maxDur   time.Duration
			first    bool
		)

		for _, rRes := range results {
			if rRes.err != nil || !rRes.ok {
				errCount++
				continue
			}
			okCount++
			sumDur += rRes.duration
			if !first {
				minDur = rRes.duration
				maxDur = rRes.duration
				first = true
			} else {
				if rRes.duration < minDur {
					minDur = rRes.duration
				}
				if rRes.duration > maxDur {
					maxDur = rRes.duration
				}
			}
		}

		resp := MetricsBenchmarkResponse{
			TotalRequests: totalRequests,
			OK:            okCount,
			Errors:        errCount,
			TotalTimeSec:  totalTime.Seconds(),
			LatencyAvgMS:  0,
			LatencyMinMS:  0,
			LatencyMaxMS:  0,
			RPS:           0,
		}

		if okCount > 0 {
			avg := sumDur / time.Duration(okCount)
			resp.LatencyAvgMS = avg.Seconds() * 1000
			resp.LatencyMinMS = minDur.Seconds() * 1000
			resp.LatencyMaxMS = maxDur.Seconds() * 1000
			resp.RPS = float64(okCount) / totalTime.Seconds()
		}

		log.Printf("[API] benchmark: req=%d conc=%d user_id=%d top_n=%d OK=%d ERR=%d total=%.3fs avg=%.3fms\n",
			totalRequests, concurrency, req.UserID, req.TopN, okCount, errCount, totalTime.Seconds(), resp.LatencyAvgMS,
		)

		w.Header().Set("Content-Type", "application/json")
		if err := json.NewEncoder(w).Encode(resp); err != nil {
			log.Println("[API] error codificando respuesta benchmark:", err)
		}
	}
}

// GET /metrics/system
// Devuelve métricas simples del sistema: uso CPU, goroutines, número de nodos.
func systemMetricsHandler(cfg nodeConfig) http.HandlerFunc {
	return func(w http.ResponseWriter, r *http.Request) {
		// Medir % CPU usando gopsutil. Intervalo 0 → cálculo inmediato desde la última llamada.
		// Es suficiente para el propósito del proyecto.
		cpuPercents, err := cpu.Percent(0, false)
		var cpuPercent float64
		if err != nil || len(cpuPercents) == 0 {
			log.Printf("[API] error obteniendo CPU percent: %v\n", err)
			cpuPercent = 0
		} else {
			cpuPercent = cpuPercents[0]
		}

		// Número de goroutines de la API
		numGoroutines := runtime.NumGoroutine()

		// Número de nodos configurados en la API
		numNodes := 0
		if cfg.Node1Addr != "" {
			numNodes++
		}
		if cfg.Node2Addr != "" {
			numNodes++
		}
		if cfg.Node3Addr != "" {
			numNodes++
		}

		resp := SystemMetricsResponse{
			CPUPercent:    cpuPercent,
			NumGoroutines: numGoroutines,
			NumNodes:      numNodes,
			Timestamp:     time.Now(),
		}

		w.Header().Set("Content-Type", "application/json")
		if err := json.NewEncoder(w).Encode(resp); err != nil {
			log.Println("[API] error codificando system metrics:", err)
		}
	}
}

func redisMetricsHandler() http.HandlerFunc {
	return func(w http.ResponseWriter, r *http.Request) {
		if redisClient == nil {
			http.Error(w, "Redis no está configurado", http.StatusServiceUnavailable)
			return
		}

		ctx, cancel := context.WithTimeout(r.Context(), 2*time.Second)
		defer cancel()

		dbSize, err := redisClient.DBSize(ctx).Result()
		if err != nil {
			log.Printf("[API] error obteniendo DBSize de Redis: %v\n", err)
			http.Error(w, "error consultando Redis", http.StatusInternalServerError)
			return
		}

		infoStr, err := redisClient.Info(ctx, "memory", "stats").Result()
		if err != nil {
			log.Printf("[API] error obteniendo INFO de Redis: %v\n", err)
			http.Error(w, "error consultando Redis", http.StatusInternalServerError)
			return
		}

		metrics := RedisMetricsResponse{
			DBSize: dbSize,
		}

		lines := strings.Split(infoStr, "\n")
		for _, line := range lines {
			line = strings.TrimSpace(line)
			if line == "" || strings.HasPrefix(line, "#") {
				continue
			}

			parts := strings.SplitN(line, ":", 2)
			if len(parts) != 2 {
				continue
			}

			key := parts[0]
			valStr := parts[1]

			switch key {
			case "used_memory":
				if v, err := strconv.ParseInt(valStr, 10, 64); err == nil {
					metrics.UsedMemoryBytes = v
				}
			case "connected_clients":
				if v, err := strconv.ParseInt(valStr, 10, 64); err == nil {
					metrics.ConnectedClients = v
				}
			case "keyspace_hits":
				if v, err := strconv.ParseInt(valStr, 10, 64); err == nil {
					metrics.KeyspaceHits = v
				}
			case "keyspace_misses":
				if v, err := strconv.ParseInt(valStr, 10, 64); err == nil {
					metrics.KeyspaceMisses = v
				}
			}
		}

		w.Header().Set("Content-Type", "application/json")
		if err := json.NewEncoder(w).Encode(metrics); err != nil {
			log.Println("[API] error codificando Redis metrics:", err)
		}
	}
}

// Middleware simple de CORS para permitir llamadas desde el frontend (localhost:5173)
func withCORS(h http.Handler) http.Handler {
	return http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		// Origen permitido: tu frontend Vite
		w.Header().Set("Access-Control-Allow-Origin", "http://localhost:5173")
		w.Header().Set("Access-Control-Allow-Methods", "GET, POST, OPTIONS")
		w.Header().Set("Access-Control-Allow-Headers", "Content-Type, Authorization")

		// Manejo de preflight (OPTIONS)
		if r.Method == http.MethodOptions {
			w.WriteHeader(http.StatusNoContent)
			return
		}

		h.ServeHTTP(w, r)
	})
}

// -------------------------------------------------------------------
// main de la API
// -------------------------------------------------------------------

func main() {
	cfg := loadNodeConfig()
	initJWT()
	port := os.Getenv("PORT")
	if port == "" {
		port = "8000"
	}

	ctx, cancel := context.WithTimeout(context.Background(), 10*time.Second)
	defer cancel()
	if err := initMongo(ctx); err != nil {
		log.Printf("[API] Advertencia: no se pudo conectar a Mongo: %v\n", err)
	}
	if err := initRedis(ctx); err != nil {
		log.Printf("[API] Advertencia: no se pudo conectar a Redis: %v\n", err)
	}

	// Base URL para el benchmark (normalmente http://localhost:PORT)
	apiBaseURL := os.Getenv("BENCHMARK_BASE_URL")
	if apiBaseURL == "" {
		apiBaseURL = "http://localhost:" + port
	}

	mux := http.NewServeMux()
	mux.HandleFunc("/health", healthHandler)
	mux.Handle("/auth/login", loginHandler())

	// rutas protegidas por JWT (via authMiddleware global)
	mux.Handle("/recommend/", recommendHandler(cfg))
	mux.Handle("/history/", historyHandler())
	mux.Handle("/users/", userHandler())
	mux.Handle("/cluster/status", adminOnly(clusterStatusHandler(cfg)))
	mux.Handle("/metrics/benchmark", adminOnly(metricsBenchmarkHandler(apiBaseURL)))
	mux.Handle("/metrics/system", adminOnly(systemMetricsHandler(cfg)))
	mux.Handle("/metrics/redis", adminOnly(redisMetricsHandler()))
	mux.Handle("/audit/latest", adminOnly(latestAuditHandler()))

	// NUEVA ruta WebSocket (solo admin, se valida dentro del handler vía token query)
	mux.Handle("/ws/metrics", metricsWSHandler(cfg))

	handler := authMiddleware(mux)
	handler = withCORS(handler)
	addr := ":" + port
	log.Printf("[API] Iniciando API en %s\n", addr)
	log.Printf("[API] Nodo1: %s, Nodo2: %s, Nodo3: %s\n",
		cfg.Node1Addr, cfg.Node2Addr, cfg.Node3Addr)

	if err := http.ListenAndServe(addr, handler); err != nil {
		log.Fatal("[API] error en ListenAndServe:", err)
	}
}
