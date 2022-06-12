package ps

import (
	"fmt"
	"os"
	"text/tabwriter"
	"time"

	"github.com/auxten/dnn-ranker/nn"
)

// StatsPrinter prints training progress
type StatsPrinter struct {
	w *tabwriter.Writer
}

// NewStatsPrinter creates a StatsPrinter
func NewStatsPrinter() *StatsPrinter {
	return &StatsPrinter{tabwriter.NewWriter(os.Stdout, 16, 0, 3, ' ', 0)}
}

// Init initializes printer
func (p *StatsPrinter) Init(n *nn.Neural) {
	fmt.Fprintf(p.w, "Epochs\tElapsed\tLoss (%s)\t", n.Config.Loss)
	if n.Config.Mode == nn.ModeMultiClass {
		fmt.Fprintf(p.w, "Accuracy\t\n---\t---\t---\t---\t\n")
	} else {
		fmt.Fprintf(p.w, "\n---\t---\t---\t\n")
	}
}

// PrintProgress prints the current state of training
func (p *StatsPrinter) PrintProgress(n *nn.Neural, validation Samples, elapsed time.Duration, iteration int) {
	fmt.Fprintf(p.w, "%d\t%s\t%.4f\t%s\n",
		iteration,
		elapsed.String(),
		crossValidate(n, validation),
		formatAccuracy(n, validation))
	p.w.Flush()
}

func formatAccuracy(n *nn.Neural, validation Samples) string {
	if n.Config.Mode == nn.ModeMultiClass {
		return fmt.Sprintf("%.2f\t", accuracy(n, validation))
	}
	return ""
}

func accuracy(n *nn.Neural, validation Samples) float64 {
	correct := 0
	for _, e := range validation {
		est := n.Predict(e.Input)
		if nn.ArgMax(e.Response) == nn.ArgMax(est) {
			correct++
		}
	}
	return float64(correct) / float64(len(validation))
}

func crossValidate(n *nn.Neural, validation Samples) float64 {
	predictions, responses := make([][]float64, len(validation)), make([][]float64, len(validation))
	for i := 0; i < len(validation); i++ {
		predictions[i] = n.Predict(validation[i].Input)
		responses[i] = validation[i].Response
	}

	return nn.GetLoss(n.Config.Loss).F(predictions, responses)
}
