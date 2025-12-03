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

	"go.mongodb.org/mongo-driver/bson"
	"go.mongodb.org/mongo-driver/mongo"
	"go.mongodb.org/mongo-driver/mongo/options"
)

type UserDoc struct {
	UserID     int       `bson:"user_id"`
	Name       string    `bson:"name"`
	Gender     string    `bson:"gender"`
	Age        int       `bson:"age"`
	Occupation int       `bson:"occupation"`
	ZipCode    string    `bson:"zip_code"`
	CreatedAt  time.Time `bson:"created_at"`
}

// syntheticName genera un nombre determinista a partir del userID
func syntheticName(userID int) string {
	firstNames := []string{"Alex", "María", "Luis", "Carla", "Diego", "Lucía", "Javier", "Ana"}
	lastNames := []string{"García", "Pérez", "Rodríguez", "López", "Fernández", "Sánchez", "Torres", "Ramírez"}

	i := userID % len(firstNames)
	j := (userID / len(firstNames)) % len(lastNames)

	return fmt.Sprintf("%s %s", firstNames[i], lastNames[j])
}

func main() {
	ctx := context.Background()

	// Mismos defaults que usa la API
	uri := os.Getenv("MONGO_URI")
	if uri == "" {
		uri = "mongodb://localhost:27017"
	}
	dbName := os.Getenv("MONGO_DB")
	if dbName == "" {
		dbName = "recomendador"
	}
	usersCollName := os.Getenv("MONGO_USERS_COLL")
	if usersCollName == "" {
		usersCollName = "users"
	}

	log.Printf("[import_users] Conectando a MongoDB: %s, DB=%s, Coll=%s\n", uri, dbName, usersCollName)

	client, err := mongo.Connect(ctx, options.Client().ApplyURI(uri))
	if err != nil {
		log.Fatalf("error conectando a Mongo: %v", err)
	}
	defer func() {
		_ = client.Disconnect(context.Background())
	}()

	if err := client.Ping(ctx, nil); err != nil {
		log.Fatalf("error haciendo ping a Mongo: %v", err)
	}

	coll := client.Database(dbName).Collection(usersCollName)

	// Vaciar colección para reimportar limpia
	{
		ctxClear, cancel := context.WithTimeout(ctx, 10*time.Second)
		defer cancel()
		res, err := coll.DeleteMany(ctxClear, bson.D{})
		if err != nil {
			log.Fatalf("error limpiando colección users: %v", err)
		}
		log.Printf("[import_users] Colección limpiada, documentos eliminados: %d\n", res.DeletedCount)
	}

	// Ruta del archivo
	path := "../ml-10m/users_synth.dat"
	if len(os.Args) > 1 {
		path = os.Args[1]
	}

	f, err := os.Open(path)
	if err != nil {
		log.Fatalf("no se pudo abrir %s: %v", path, err)
	}
	defer f.Close()

	scanner := bufio.NewScanner(f)
	batch := make([]interface{}, 0, 1000)

	totalLines := 0
	totalDocs := 0

	for scanner.Scan() {
		line := strings.TrimSpace(scanner.Text())
		if line == "" {
			continue
		}
		totalLines++

		parts := strings.Split(line, "::")
		if len(parts) != 5 {
			log.Printf("[WARN] línea %d con formato inesperado: %q\n", totalLines, line)
			continue
		}

		userID, err := strconv.Atoi(parts[0])
		if err != nil {
			log.Printf("[WARN] user_id inválido en línea %d: %q\n", totalLines, line)
			continue
		}
		gender := parts[1]
		age, err := strconv.Atoi(parts[2])
		if err != nil {
			log.Printf("[WARN] edad inválida en línea %d: %q\n", totalLines, line)
			continue
		}
		occ, err := strconv.Atoi(parts[3])
		if err != nil {
			log.Printf("[WARN] ocupación inválida en línea %d: %q\n", totalLines, line)
			continue
		}
		zipCode := parts[4]

		doc := UserDoc{
			UserID:     userID,
			Name:       syntheticName(userID),
			Gender:     gender,
			Age:        age,
			Occupation: occ,
			ZipCode:    zipCode,
			CreatedAt:  time.Now(),
		}

		batch = append(batch, doc)
		if len(batch) >= 1000 {
			if err := insertBatch(ctx, coll, batch); err != nil {
				log.Fatalf("error insertando batch: %v", err)
			}
			totalDocs += len(batch)
			batch = batch[:0]
		}
	}
	if err := scanner.Err(); err != nil {
		log.Fatalf("error leyendo archivo %s: %v", path, err)
	}

	if len(batch) > 0 {
		if err := insertBatch(ctx, coll, batch); err != nil {
			log.Fatalf("error insertando batch final: %v", err)
		}
		totalDocs += len(batch)
	}

	log.Printf("[import_users] Importación completada. Líneas leídas: %d, documentos insertados: %d\n", totalLines, totalDocs)
}

func insertBatch(ctx context.Context, coll *mongo.Collection, docs []interface{}) error {
	if len(docs) == 0 {
		return nil
	}
	ctxBatch, cancel := context.WithTimeout(ctx, 10*time.Second)
	defer cancel()

	_, err := coll.InsertMany(ctxBatch, docs)
	if err != nil {
		return fmt.Errorf("InsertMany falló: %w", err)
	}
	log.Printf("[import_users] Batch insertado: %d documentos\n", len(docs))
	return nil
}
