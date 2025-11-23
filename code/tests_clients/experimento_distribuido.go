package main

import (
	"fmt"
	"net/http"
	"sync"
	"time"
)

// Configuración del experimento
const (
	baseURL       = "http://localhost:8000" // API
	totalRequests = 100                     // número total de requests
	concurrency   = 10                      // número de goroutines concurrentes
	userStart     = 1                       // primer user_id a usar
	userRange     = 300                     // rango de usuarios (1..300, por ejemplo)
	topN          = 20                      // top_n en la query
)

// Resultado de un request
type result struct {
	ok       bool
	duration time.Duration
	err      error
	status   int
}

func main() {
	fmt.Println("=== Experimento distribuido /recommend ===")
	fmt.Printf("Base URL    : %s\n", baseURL)
	fmt.Printf("Requests    : %d\n", totalRequests)
	fmt.Printf("Concurrency : %d\n", concurrency)
	fmt.Printf("Users range : [%d .. %d]\n", userStart, userStart+userRange-1)

	results := make([]result, totalRequests)

	var wg sync.WaitGroup
	jobs := make(chan int)

	client := &http.Client{
		Timeout: 10 * time.Second,
	}

	startAll := time.Now()

	// Lanzar workers
	for w := 0; w < concurrency; w++ {
		wg.Add(1)
		go func(id int) {
			defer wg.Done()
			for i := range jobs {
				// Elegir un user_id dentro del rango
				userID := userStart + (i % userRange)
				url := fmt.Sprintf("%s/recommend/%d?top_n=%d", baseURL, userID, topN)

				startReq := time.Now()
				resp, err := client.Get(url)
				dur := time.Since(startReq)

				res := result{
					ok:       false,
					duration: dur,
					err:      err,
					status:   0,
				}

				if err == nil {
					res.status = resp.StatusCode
					_ = resp.Body.Close()
					if resp.StatusCode == http.StatusOK {
						res.ok = true
					}
				}

				results[i] = res
			}
		}(w)
	}

	// Enviar trabajos
	for i := 0; i < totalRequests; i++ {
		jobs <- i
	}
	close(jobs)

	wg.Wait()
	totalTime := time.Since(startAll)

	// Analizar resultados
	var (
		okCount  int
		errCount int
		sumDur   time.Duration
		minDur   time.Duration
		maxDur   time.Duration
		firstDur bool
	)

	for _, r := range results {
		if r.err != nil || !r.ok {
			errCount++
			continue
		}
		okCount++
		sumDur += r.duration
		if !firstDur {
			minDur = r.duration
			maxDur = r.duration
			firstDur = true
		} else {
			if r.duration < minDur {
				minDur = r.duration
			}
			if r.duration > maxDur {
				maxDur = r.duration
			}
		}
	}

	fmt.Println("\n=== Resultados ===")
	fmt.Printf("Total requests       : %d\n", totalRequests)
	fmt.Printf("OK (status=200)      : %d\n", okCount)
	fmt.Printf("Errores              : %d\n", errCount)
	fmt.Printf("Tiempo total batch   : %.3fs\n", totalTime.Seconds())
	if okCount > 0 {
		avg := sumDur / time.Duration(okCount)
		fmt.Printf("Latencia promedio OK : %.3f ms\n", avg.Seconds()*1000)
		fmt.Printf("Latencia mínima OK   : %.3f ms\n", minDur.Seconds()*1000)
		fmt.Printf("Latencia máxima OK   : %.3f ms\n", maxDur.Seconds()*1000)
		fmt.Printf("Requests/seg (aprox) : %.2f\n", float64(okCount)/totalTime.Seconds())
	} else {
		fmt.Println("No hubo respuestas OK para calcular latencias.")
	}
	fmt.Println("=== Fin experimento ===")
}
