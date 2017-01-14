package main

import (
	"log"
	"math"
	"math/rand"
	"time"

	"github.com/NOX73/go-neural"
	"github.com/NOX73/go-neural/learn"
)

func main() {
	network := neural.NewNetwork(2, []int{5, 5, 1})
	network.RandomizeSynapses()

	ch := make(chan float64, 300)

	go func() {
		tick := time.Tick(5 * time.Second)
		sampleSize := 1000
		sample := make([]float64, sampleSize)
		index := 0

		for {
			select {
			// Feed delta in to sample, taking only the latest N samples
			case v := <-ch:
				sample[index] = v
				index++
				if index == len(sample) {
					index = 0
				}
			// Ouput Average error for current sample every 5 seconds
			case <-tick:
				var sum float64
				for _, val := range sample {
					sum += val
				}

				log.Printf("Avarege error: %f", sum/float64(sampleSize))

			}
		}

	}()

	count := 100000000
	for i := 0; i < count; i++ {
		test := []float64{rand.Float64(), rand.Float64()}
		result := network.Calculate(test)[0]

		ch <- math.Abs(test[0]*test[1] - result)

		// Commented out to reduce noise, but can be uncommeted to see what input
		// and expected output are being introduced to the network
		// log.Printf("Learning: %f * %f = %f", test[0], test[1], test[0]*test[1])

		learn.Learn(network, test, []float64{test[0] * test[1]}, 0.2)
	}

	log.Println("Quality:")
	e := learn.Evaluation(network, []float64{5, 5}, []float64{25})
	log.Println(e)

	log.Println("Calculation of 0.2 * 0.4:")
	out := network.Calculate([]float64{0.2, 0.4})
	log.Println(out[0])
}
