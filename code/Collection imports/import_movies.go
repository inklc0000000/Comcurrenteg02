package main

import (
	"bufio"
	"context"
	"fmt"
	"log"
	"os"
	"strconv"
	"strings"
	"time"

	"go.mongodb.org/mongo-driver/mongo"
	"go.mongodb.org/mongo-driver/mongo/options"
)

const MoviesPath = "../ml-10m/movies.dat"

type MovieDoc struct {
	MovieID int    `bson:"movie_id"`
	Title   string `bson:"title"`
	Genres  string `bson:"genres"`
}

func main() {
	fmt.Println("=== Importador de películas a MongoDB ===")

	uri := os.Getenv("MONGO_URI")
	if uri == "" {
		uri = "mongodb://localhost:27017"
	}
	dbName := os.Getenv("MONGO_DB")
	if dbName == "" {
		dbName = "recomendador"
	}
	collName := os.Getenv("MONGO_MOVIES_COLL")
	if collName == "" {
		collName = "movies"
	}

	fmt.Printf("Usando MONGO_URI=%s\n", uri)
	fmt.Printf("DB=%s, Coll=%s\n", dbName, collName)
	fmt.Printf("Leyendo archivo de películas: %s\n", MoviesPath)

	// Conectar a Mongo
	// Damos más tiempo porque la importación completa puede tardar bastante
	ctx, cancel := context.WithTimeout(context.Background(), 5*time.Minute)
	defer cancel()

	client, err := mongo.Connect(ctx, options.Client().ApplyURI(uri))
	if err != nil {
		log.Fatalf("Error conectando a Mongo: %v", err)
	}
	defer client.Disconnect(ctx)

	if err := client.Ping(ctx, nil); err != nil {
		log.Fatalf("Error haciendo ping a Mongo: %v", err)
	}

	db := client.Database(dbName)
	coll := db.Collection(collName)

	// Limpiar colección antes de importar
	if err := coll.Drop(ctx); err != nil {
		log.Printf("Advertencia: no se pudo hacer Drop de %s: %v\n", collName, err)
	} else {
		log.Printf("Colección %s limpiada (Drop OK)\n", collName)
	}

	// Abrir archivo movies.dat
	file, err := os.Open(MoviesPath)
	if err != nil {
		log.Fatalf("No se pudo abrir %s: %v", MoviesPath, err)
	}
	defer file.Close()

	scanner := bufio.NewScanner(file)

	var docs []interface{}
	count := 0

	for scanner.Scan() {
		line := scanner.Text()
		// Formato MovieLens: MovieID::Title::Genres
		parts := strings.SplitN(line, "::", 3)
		if len(parts) != 3 {
			log.Printf("Línea ignorada (formato inválido): %s\n", line)
			continue
		}

		idStr := parts[0]
		title := parts[1]
		genres := parts[2]

		movieID, err := strconv.Atoi(idStr)
		if err != nil {
			log.Printf("ID inválido (%s) en línea: %s\n", idStr, line)
			continue
		}

		doc := MovieDoc{
			MovieID: movieID,
			Title:   title,
			Genres:  genres,
		}
		docs = append(docs, doc)
		count++

		// Insertar en lotes para no saturar
		if len(docs) >= 1000 {
			if err := insertBatch(ctx, coll, docs); err != nil {
				log.Fatalf("Error al insertar batch: %v", err)
			}
			docs = docs[:0]
		}
	}

	if err := scanner.Err(); err != nil {
		log.Fatalf("Error leyendo archivo: %v", err)
	}

	// Insertar lo que quedó
	if len(docs) > 0 {
		if err := insertBatch(ctx, coll, docs); err != nil {
			log.Fatalf("Error al insertar último batch: %v", err)
		}
	}

	fmt.Printf("Importación completada. Películas insertadas: %d\n", count)
	fmt.Println("=== Fin importador ===")
}

func insertBatch(ctx context.Context, coll *mongo.Collection, docs []interface{}) error {
	if len(docs) == 0 {
		return nil
	}
	// damos un timeout por batch
	ctxBatch, cancel := context.WithTimeout(ctx, 10*time.Second)
	defer cancel()

	_, err := coll.InsertMany(ctxBatch, docs)
	if err != nil {
		return fmt.Errorf("InsertMany falló: %w", err)
	}
	log.Printf("Batch insertado: %d documentos\n", len(docs))
	return nil
}
