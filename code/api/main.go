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
	"sort"
	"strconv"
	"strings"
	"time"

	"go.mongodb.org/mongo-driver/bson"
	"go.mongodb.org/mongo-driver/mongo"
	"go.mongodb.org/mongo-driver/mongo/options"

	redis "github.com/redis/go-redis/v9"
)

// -------------------------------------------------------------------
// Tipos compartidos con el nodo (mismo JSON que usa node.go)
// -------------------------------------------------------------------

type nodeRequest struct {
	UserID int `json:"user_id"`
	TopN   int `json:"top_n"`
}

type Item struct {
	MovieID int     `json:"movie_id" bson:"movie_id"`
	Score   float64 `json:"score" bson:"score"`
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

// Documento de usuario en Mongo (MovieLens 1M users.dat)
type UserDoc struct {
	UserID     int    `bson:"user_id" json:"user_id"`
	Gender     string `bson:"gender" json:"gender"`
	Age        int    `bson:"age" json:"age"`
	Occupation int    `bson:"occupation" json:"occupation"`
	ZipCode    string `bson:"zip_code" json:"zip_code"`
}

// Respuesta enriquecida para HTTP
type MovieWithDetails struct {
	MovieID int      `json:"movie_id"`
	Title   string   `json:"title"`
	Genres  []string `json:"genres"`
	Score   float64  `json:"score"`
}

type RecommendHTTPResponse struct {
	UserID int                `json:"user_id"`
	Items  []MovieWithDetails `json:"items"`
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

	log.Printf("[API] Conectado a MongoDB en %s, DB=%s, RecColl=%s, MoviesColl=%s, UsersColl=%s\n",
		uri, dbName, recCollName, moviesCollName, usersCollName)
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
		allItems = append(allItems, r.resp.Items...)
	}

	if !hadSuccess {
		return nil, fmt.Errorf("ningún nodo pudo responder")
	}

	// combinar y ordenar globalmente
	if len(allItems) == 0 {
		return allItems, nil
	}

	// eliminar duplicados por movie_id y quedarnos con mayor score
	bestByID := make(map[int]Item)
	for _, it := range allItems {
		if prev, ok := bestByID[it.MovieID]; !ok || it.Score > prev.Score {
			bestByID[it.MovieID] = it
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

		w.Header().Set("Content-Type", "application/json")
		if err := json.NewEncoder(w).Encode(records); err != nil {
			log.Println("[API] error codificando historial HTTP:", err)
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

// -------------------------------------------------------------------
// main de la API
// -------------------------------------------------------------------

func main() {
	cfg := loadNodeConfig()

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

	mux := http.NewServeMux()
	mux.HandleFunc("/health", healthHandler)
	mux.Handle("/recommend/", recommendHandler(cfg))
	mux.Handle("/history/", historyHandler())
	mux.Handle("/users/", userHandler())

	addr := ":" + port
	log.Printf("[API] Iniciando API en %s\n", addr)
	log.Printf("[API] Nodo1: %s, Nodo2: %s, Nodo3: %s\n",
		cfg.Node1Addr, cfg.Node2Addr, cfg.Node3Addr)

	if err := http.ListenAndServe(addr, mux); err != nil {
		log.Fatal("[API] error en ListenAndServe:", err)
	}
}
