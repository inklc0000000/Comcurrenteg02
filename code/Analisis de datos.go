package main

import (
	"bufio"
	"fmt"
	"os"
	"strconv"
	"strings"
	"sync"
	"time"
)

type Rating struct {
	UserID  int
	MovieID int
	Score   int
}

type User struct {
	ID     int
	Gender string
	Age    int
}

type Movie struct {
	ID     int
	Title  string
	Genres []string
}

// -----------------------------
// 1. LECTURA Y LIMPIEZA DE DATOS
// -----------------------------
func readRatings(path string) []Rating {
	file, err := os.Open(path)
	if err != nil {
		panic(err)
	}
	defer file.Close()

	var ratings []Rating
	scanner := bufio.NewScanner(file)
	for scanner.Scan() {
		parts := strings.Split(scanner.Text(), "::")
		if len(parts) != 4 {
			continue
		}
		userID, _ := strconv.Atoi(parts[0])
		movieID, _ := strconv.Atoi(parts[1])
		score, _ := strconv.Atoi(parts[2])
		if score >= 1 && score <= 5 {
			ratings = append(ratings, Rating{userID, movieID, score})
		}
	}
	return ratings
}

func readUsers(path string) []User {
	file, err := os.Open(path)
	if err != nil {
		panic(err)
	}
	defer file.Close()

	var users []User
	scanner := bufio.NewScanner(file)
	for scanner.Scan() {
		parts := strings.Split(scanner.Text(), "::")
		if len(parts) != 5 {
			continue
		}
		id, _ := strconv.Atoi(parts[0])
		age, _ := strconv.Atoi(parts[2])
		users = append(users, User{id, parts[1], age})
	}
	return users
}

func readMovies(path string) []Movie {
	file, err := os.Open(path)
	if err != nil {
		panic(err)
	}
	defer file.Close()

	var movies []Movie
	scanner := bufio.NewScanner(file)
	for scanner.Scan() {
		parts := strings.Split(scanner.Text(), "::")
		if len(parts) != 3 {
			continue
		}
		id, _ := strconv.Atoi(parts[0])
		genres := strings.Split(parts[2], "|")
		movies = append(movies, Movie{id, parts[1], genres})
	}
	return movies
}

// -----------------------------
// 2. MATRIZ USUARIO-PELÍCULA (CONCURRENTE)
// -----------------------------
func findMaxIDs(ratings []Rating) (int, int) {
	maxUser, maxMovie := 0, 0
	for _, r := range ratings {
		if r.UserID > maxUser {
			maxUser = r.UserID
		}
		if r.MovieID > maxMovie {
			maxMovie = r.MovieID
		}
	}
	return maxUser, maxMovie
}

func buildMatrix(ratings []Rating, numUsers, numMovies int) [][]int {
	matrix := make([][]int, numUsers+1)
	for i := range matrix {
		matrix[i] = make([]int, numMovies+1)
	}

	var wg sync.WaitGroup
	numWorkers := 8
	block := len(ratings) / numWorkers

	for w := 0; w < numWorkers; w++ {
		start := w * block
		end := start + block
		if w == numWorkers-1 {
			end = len(ratings)
		}

		wg.Add(1)
		go func(start, end int) {
			defer wg.Done()
			for i := start; i < end; i++ {
				r := ratings[i]
				if r.UserID <= numUsers && r.MovieID <= numMovies {
					matrix[r.UserID][r.MovieID] = r.Score
				}
			}
		}(start, end)
	}

	wg.Wait()
	return matrix
}

// -----------------------------
// 3. NORMALIZACIÓN (MEDIA CENTRADA)
// -----------------------------
func normalizeMatrix(matrix [][]int) [][]float64 {
	norm := make([][]float64, len(matrix))
	for i := range matrix {
		norm[i] = make([]float64, len(matrix[i]))
		sum := 0
		count := 0
		for _, val := range matrix[i] {
			if val > 0 {
				sum += val
				count++
			}
		}
		if count == 0 {
			continue
		}
		mean := float64(sum) / float64(count)
		for j, val := range matrix[i] {
			if val > 0 {
				norm[i][j] = float64(val) - mean
			}
		}
	}
	return norm
}

// -----------------------------
// 4. CALCULO DE LA SPARSITY (PORCENTAJE DE CELDAS VACIAS EN LA MATRIZ)
// -----------------------------

func calculateSparsity(ratingsCount, numUsers, numMovies int) float64 {
	total := float64(numUsers) * float64(numMovies)
	filled := float64(ratingsCount)
	sparsity := 1 - (filled / total)
	return sparsity * 100
}

// -----------------------------
// MAIN
// -----------------------------
func main() {
	start := time.Now()
	fmt.Println("=== Análisis de Datos: MovieLens 1M ===")

	ratings := readRatings("ml-1m/ratings.dat")
	users := readUsers("ml-1m/users.dat")
	movies := readMovies("ml-1m/movies.dat")

	// Determinar IDs máximos reales
	maxUser, maxMovie := findMaxIDs(ratings)

	fmt.Println("Usuarios:", len(users), "(ID máximo:", maxUser, ")")
	fmt.Println("Películas:", len(movies), "(ID máximo:", maxMovie, ")")
	fmt.Println("Ratings:", len(ratings))

	fmt.Println("\nGenerando matriz usuario-película de forma concurrente...")
	t1 := time.Now()
	matrix := buildMatrix(ratings, maxUser, maxMovie)
	fmt.Printf("Matriz generada en %.2f segundos\n", time.Since(t1).Seconds())

	fmt.Println("\nNormalizando matriz por usuario...")
	t2 := time.Now()
	normMatrix := normalizeMatrix(matrix)
	fmt.Printf("Normalización completada en %.2f segundos\n", time.Since(t2).Seconds())

	fmt.Println("\nEjemplo de usuario 1 (primeras 1000 películas):")
	fmt.Println(normMatrix[1][:1000])

	sparsity := calculateSparsity(len(ratings), maxUser, maxMovie)
	fmt.Printf("\nSparsity de la matriz: %.2f%%\n", sparsity)

	fmt.Printf("\nProceso total completado en %.2f segundos\n", time.Since(start).Seconds())
	fmt.Println("===========================================")
}
