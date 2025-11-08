package main

import (
	"bufio"
	"fmt"
	"os"
	"sort"
	"strconv"
	"strings"
	"sync"
	"time"
)

const (
	KNeighbors  = 20
	MinOverlap  = 5
	TopN        = 500
	NumWorkers  = 16
	RatingsPath = "ml-1m/ratings.dat"
	ColdThresh  = 10 // menos de 10 ratings = cold-start
)

// estructuras
type Pair struct {
	ID  int
	Val float64
}

type Sim struct {
	V   int
	S   float64
	Cnt int
}

// Lectura ratings.dat
func readRatings(path string) [][3]int {
	f, err := os.Open(path)
	if err != nil {
		panic(err)
	}
	defer f.Close()

	sc := bufio.NewScanner(f)
	out := make([][3]int, 0, 1000209)
	for sc.Scan() {
		parts := strings.Split(sc.Text(), "::")
		if len(parts) != 4 {
			continue
		}
		u, _ := strconv.Atoi(parts[0])
		i, _ := strconv.Atoi(parts[1])
		r, _ := strconv.Atoi(parts[2])
		if r >= 1 && r <= 5 {
			out = append(out, [3]int{u, i, r})
		}
	}
	return out
}

// estructuras base (Matriz dispersa)
func buildUserRaw(ratings [][3]int) map[int]map[int]int {
	userRaw := make(map[int]map[int]int)
	var mu sync.Mutex
	var wg sync.WaitGroup

	numWorkers := NumWorkers
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
			local := make(map[int]map[int]int)
			for _, t := range ratings[start:end] {
				u, i, r := t[0], t[1], t[2]
				m, ok := local[u]
				if !ok {
					m = make(map[int]int)
					local[u] = m
				}
				m[i] = r
			}

			mu.Lock()
			for u, items := range local {
				m, ok := userRaw[u]
				if !ok {
					m = make(map[int]int)
					userRaw[u] = m
				}
				for i, r := range items {
					m[i] = r
				}
			}
			mu.Unlock()
		}(start, end)
	}

	wg.Wait()
	return userRaw
}

func userMeans(userRaw map[int]map[int]int) map[int]float64 {
	means := make(map[int]float64, len(userRaw))
	for u, items := range userRaw {
		sum := 0
		for _, r := range items {
			sum += r
		}
		means[u] = float64(sum) / float64(len(items))
	}
	return means
}

// Normalizamos la matriz
func buildUserNorm(userRaw map[int]map[int]int, means map[int]float64) map[int]map[int]float64 {
	userNorm := make(map[int]map[int]float64)
	var mu sync.Mutex
	var wg sync.WaitGroup

	users := make([]int, 0, len(userRaw))
	for u := range userRaw {
		users = append(users, u)
	}

	numWorkers := NumWorkers
	block := len(users) / numWorkers

	for w := 0; w < numWorkers; w++ {
		start := w * block
		end := start + block
		if w == numWorkers-1 {
			end = len(users)
		}

		wg.Add(1)
		go func(start, end int) {
			defer wg.Done()
			local := make(map[int]map[int]float64)
			for _, u := range users[start:end] {
				m := make(map[int]float64)
				muU := means[u]
				for i, r := range userRaw[u] {
					m[i] = float64(r) - muU
				}
				local[u] = m
			}
			mu.Lock()
			for u, m := range local {
				userNorm[u] = m
			}
			mu.Unlock()
		}(start, end)
	}

	wg.Wait()
	return userNorm
}

// Generamos la matriz con indice inverso
func buildInverted(userRaw map[int]map[int]int) map[int][]int {
	inv := make(map[int][]int)
	var mu sync.Mutex
	var wg sync.WaitGroup

	users := make([]int, 0, len(userRaw))
	for u := range userRaw {
		users = append(users, u)
	}

	numWorkers := NumWorkers
	block := len(users) / numWorkers

	for w := 0; w < numWorkers; w++ {
		start := w * block
		end := start + block
		if w == numWorkers-1 {
			end = len(users)
		}

		wg.Add(1)
		go func(start, end int) {
			defer wg.Done()
			local := make(map[int][]int)
			for _, u := range users[start:end] {
				for i := range userRaw[u] {
					local[i] = append(local[i], u)
				}
			}

			mu.Lock()
			for i, us := range local {
				inv[i] = append(inv[i], us...)
			}
			mu.Unlock()
		}(start, end)
	}

	wg.Wait()
	return inv
}

// Pearson
func pearson(uMap, vMap map[int]float64) (float64, int) {
	num, denU, denV := 0.0, 0.0, 0.0
	cnt := 0
	if len(uMap) > len(vMap) {
		uMap, vMap = vMap, uMap
	}
	for i, ru := range uMap {
		if rv, ok := vMap[i]; ok {
			num += ru * rv
			denU += ru * ru
			denV += rv * rv
			cnt++
		}
	}
	if cnt < MinOverlap || denU == 0 || denV == 0 {
		return 0, cnt
	}
	return num / (sqrt(denU) * sqrt(denV)), cnt
}

func sqrt(x float64) float64 {
	if x <= 0 {
		return 0
	}
	z := x
	for k := 0; k < 20; k++ {
		z = 0.5 * (z + x/z)
	}
	return z
}

// top-K vecinos concurrentes
func topKNeighbors(userNorm map[int]map[int]float64, inv map[int][]int) map[int][]Sim {
	users := make([]int, 0, len(userNorm))
	for u := range userNorm {
		users = append(users, u)
	}

	out := make(map[int][]Sim, len(users))
	var mu sync.Mutex
	tasks := make(chan int, len(users))
	var wg sync.WaitGroup

	worker := func() {
		defer wg.Done()
		for u := range tasks {
			candSet := make(map[int]struct{})
			for i := range userNorm[u] {
				for _, v := range inv[i] {
					if v != u {
						candSet[v] = struct{}{}
					}
				}
			}
			sims := make([]Sim, 0, 64)
			for v := range candSet {
				s, cnt := pearson(userNorm[u], userNorm[v])
				if s != 0 {
					sims = append(sims, Sim{V: v, S: s, Cnt: cnt})
				}
			}
			sort.Slice(sims, func(i, j int) bool {
				if sims[i].S == sims[j].S {
					return sims[i].Cnt > sims[j].Cnt
				}
				return sims[i].S > sims[j].S
			})
			if len(sims) > KNeighbors {
				sims = sims[:KNeighbors]
			}
			mu.Lock()
			out[u] = sims
			mu.Unlock()
		}
	}

	wg.Add(NumWorkers)
	for w := 0; w < NumWorkers; w++ {
		go worker()
	}
	for _, u := range users {
		tasks <- u
	}
	close(tasks)
	wg.Wait()
	return out
}

// popularidad global de las peliculas (para manejar el cold-start)
func globalTopMovies(userRaw map[int]map[int]int) []Pair {
	counts := make(map[int]int)
	sums := make(map[int]int)
	for _, items := range userRaw {
		for i, r := range items {
			counts[i]++
			sums[i] += r
		}
	}
	pop := make([]Pair, 0, len(counts))
	for i := range counts {
		pop = append(pop, Pair{ID: i, Val: float64(sums[i]) / float64(counts[i])})
	}
	sort.Slice(pop, func(i, j int) bool { return pop[i].Val > pop[j].Val })
	if len(pop) > TopN {
		pop = pop[:TopN]
	}
	return pop
}

// Aqui hace la prediccion concurrente
func predictForUsers(userRaw map[int]map[int]int, means map[int]float64, neigh map[int][]Sim, global []Pair) map[int][]Pair {
	users := make([]int, 0, len(userRaw))
	for u := range userRaw {
		users = append(users, u)
	}

	res := make(map[int][]Pair, len(users))
	var mu sync.Mutex
	tasks := make(chan int, len(users))
	var wg sync.WaitGroup

	worker := func() {
		defer wg.Done()
		for u := range tasks {
			if len(userRaw[u]) < ColdThresh {
				// si el usuario tiene pocas calificaciones → usar globalTopMovies
				mu.Lock()
				res[u] = global
				mu.Unlock()
				continue
			}

			cand := make(map[int]struct{})
			seen := userRaw[u]
			for _, nb := range neigh[u] {
				for i := range userRaw[nb.V] {
					if _, ok := seen[i]; !ok {
						cand[i] = struct{}{}
					}
				}
			}
			preds := make([]Pair, 0, len(cand))
			muU := means[u]
			for i := range cand {
				num, den := 0.0, 0.0
				for _, nb := range neigh[u] {
					v := nb.V
					if rv, ok := userRaw[v][i]; ok {
						num += nb.S * (float64(rv) - means[v])
						den += abs(nb.S)
					}
				}
				if den > 0 {
					rhat := muU + num/den
					// recorte al rango válido de MovieLens (1–5)
					if rhat > 5 {
						rhat = 5
					} else if rhat < 1 {
						rhat = 1
					}
					preds = append(preds, Pair{ID: i, Val: rhat})
				}
			}
			sort.Slice(preds, func(i, j int) bool { return preds[i].Val > preds[j].Val })
			if len(preds) > TopN {
				preds = preds[:TopN]
			}
			mu.Lock()
			res[u] = preds
			mu.Unlock()
		}
	}

	wg.Add(NumWorkers)
	for w := 0; w < NumWorkers; w++ {
		go worker()
	}
	for _, u := range users {
		tasks <- u
	}
	close(tasks)
	wg.Wait()
	return res
}

func abs(x float64) float64 {
	if x < 0 {
		return -x
	}
	return x
}

// main
func main() {
	t0 := time.Now()
	fmt.Println("=== User-Based CF ===")

	ratings := readRatings(RatingsPath)
	fmt.Printf("Ratings: %d\n", len(ratings))

	userRaw := buildUserRaw(ratings)
	means := userMeans(userRaw)
	userNorm := buildUserNorm(userRaw, means)
	inv := buildInverted(userRaw)
	global := globalTopMovies(userRaw)
	fmt.Printf("Usuarios: %d | Ítems distintos: %d\n", len(userRaw), len(inv))
	fmt.Printf("Parámetros → NumWorkers=%d | TopN=%d | MinOverlap=%d | KNeighbors=%d\n", NumWorkers, TopN, MinOverlap, KNeighbors)

	t1 := time.Now()
	neighbors := topKNeighbors(userNorm, inv)
	fmt.Printf("Top-K vecinos calculados en %.2fs\n", time.Since(t1).Seconds())

	t2 := time.Now()
	recs := predictForUsers(userRaw, means, neighbors, global)
	fmt.Printf("Predicciones y Top-%d en %.2fs\n", TopN, time.Since(t2).Seconds())

	if rs, ok := recs[1]; ok {
		fmt.Println("\nTop-N usuario 1:")
		for k, p := range rs {
			fmt.Printf("%2d) MovieID=%d  r̂=%.3f\n", k+1, p.ID, p.Val)
		}
	}

	fmt.Printf("\nTiempo total: %.2fs\n", time.Since(t0).Seconds())
}
