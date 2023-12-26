package main

import (
	"fmt"
	"github.com/gofiber/fiber/v2"
	"github.com/google/uuid"
	"log"
	"os"
	"os/exec"
	"path/filepath"
)

func main() {
	app := fiber.New()

	app.Post("/", func(c *fiber.Ctx) error {
		var request struct {
			AudioPath string `json:"audio_path"`
			VideoPath string `json:"video_path"`
		}

		if err := c.BodyParser(&request); err != nil {
			return err
		}

		println(request.VideoPath)
		println(request.AudioPath)

		out := fmt.Sprintf("./results/%s.mp4", uuid.New().String())

		cmd := exec.Command(
			"python", "inference.py",
			"--checkpoint_path", "checkpoints/wav2lip.pth",
			"--face", request.VideoPath,
			"--audio", request.AudioPath,
			"--outfile", out,
		)

		cmd.Stdout = os.Stdout
		cmd.Stderr = os.Stderr

		if err := cmd.Run(); err != nil {
			return err
		}

		if path, err := filepath.Abs(out); err != nil {
			return err
		} else {
			return c.SendString(path)
		}
	})

	log.Fatal(app.Listen(":6873"))
}
