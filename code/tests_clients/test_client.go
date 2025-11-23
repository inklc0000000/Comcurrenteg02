package main

import (
	"encoding/json"
	"fmt"
	"net"
)

type Request struct {
	UserID int `json:"user_id"`
	TopN   int `json:"top_n"`
}

type Item struct {
	MovieID int     `json:"movie_id"`
	Score   float64 `json:"score"`
}

type Response struct {
	UserID int    `json:"user_id"`
	Items  []Item `json:"items"`
}

func main() {
	// Conectarse al nodo 1 en localhost:9001
	conn, err := net.Dial("tcp", "127.0.0.1:9001")
	if err != nil {
		panic(err)
	}
	defer conn.Close()

	enc := json.NewEncoder(conn)
	dec := json.NewDecoder(conn)

	// Probar con el usuario 10 y top 5
	req := Request{UserID: 10, TopN: 5}
	if err := enc.Encode(&req); err != nil {
		panic(err)
	}

	var resp Response
	if err := dec.Decode(&resp); err != nil {
		panic(err)
	}

	fmt.Printf("Recomendaciones para user %d:\n", resp.UserID)
	for _, it := range resp.Items {
		fmt.Printf("  movie_id=%d score=%.3f\n", it.MovieID, it.Score)
	}
}
