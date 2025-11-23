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

const UsersPath = "../ml-1m/users.dat"

type UserDoc struct {
	UserID     int    `bson:"user_id"`
	Gender     string `bson:"gender"`
	Age        int    `bson:"age"`
	Occupation int    `bson:"occupation"`
	ZipCode    string `bson:"zip_code"`
}

func main() {
	fmt.Println("=== Importador de usuarios a MongoDB ===")

	uri := os.Getenv("MONGO_URI")
	if uri == "" {
		uri = "mongodb://localhost:27017"
	}
	dbName := os.Getenv("MONGO_DB")
	if dbName == "" {
		dbName = "recomendador"
	}
	collName := os.Getenv("MONGO_USERS_COLL")
	if collName == "" {
		collName = "users"
	}

	fmt.Printf("Usando MONGO_URI=%s\n", uri)
	fmt.Printf("DB=%s, Coll=%s\n", dbName, collName)
	fmt.Printf("Leyendo archivo de usuarios: %s\n", UsersPath)

	// Conexión a MongoDB (más tiempo porque es un proceso de importación)
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

	file, err := os.Open(UsersPath)
	if err != nil {
		log.Fatalf("No se pudo abrir %s: %v", UsersPath, err)
	}
	defer file.Close()

	scanner := bufio.NewScanner(file)

	var docs []interface{}
	count := 0

	for scanner.Scan() {
		line := scanner.Text()
		// Formato MovieLens 1M: UserID::Gender::Age::Occupation::Zip-code
		parts := strings.SplitN(line, "::", 5)
		if len(parts) != 5 {
			log.Printf("Línea ignorada (formato inválido): %s\n", line)
			continue
		}

		idStr := parts[0]
		gender := parts[1]
		ageStr := parts[2]
		occStr := parts[3]
		zip := parts[4]

		userID, err := strconv.Atoi(idStr)
		if err != nil {
			log.Printf("UserID inválido (%s) en línea: %s\n", idStr, line)
			continue
		}

		age, err := strconv.Atoi(ageStr)
		if err != nil {
			log.Printf("Age inválido (%s) en línea: %s\n", ageStr, line)
			continue
		}

		occ, err := strconv.Atoi(occStr)
		if err != nil {
			log.Printf("Occupation inválido (%s) en línea: %s\n", occStr, line)
			continue
		}

		doc := UserDoc{
			UserID:     userID,
			Gender:     gender,
			Age:        age,
			Occupation: occ,
			ZipCode:    zip,
		}

		docs = append(docs, doc)
		count++

		// Insertar en lotes de 1000 (igual que import_movies.go)
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

	fmt.Printf("Importación completada. Usuarios insertados: %d\n", count)
	fmt.Println("=== Fin importador ===")
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
	log.Printf("Batch insertado: %d documentos\n", len(docs))
	return nil
}
