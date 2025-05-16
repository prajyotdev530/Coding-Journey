import pygame

# Initialize pygame
pygame.init()

# Game Constants
WIDTH, HEIGHT = 500, 500
WHITE = (255, 255, 255)
BLUE = (0, 0, 255)
SPEED = 5

# Create Game Window
screen = pygame.display.set_mode((WIDTH, HEIGHT))
pygame.display.set_caption("Simple Pygame - Move the Square")

# Player Setup
player_size = 50
player_x = WIDTH // 2
player_y = HEIGHT // 2

# Game Loop
running = True
while running:
    pygame.time.delay(30)  # Control game speed

    # Check for Quit Event
    for event in pygame.event.get():
        if event.type == pygame.QUIT:
            running = False

    # Get Key Presses
    keys = pygame.key.get_pressed()
    if keys[pygame.K_LEFT] and player_x > 0:
        player_x -= SPEED
    if keys[pygame.K_RIGHT] and player_x < WIDTH - player_size:
        player_x += SPEED
    if keys[pygame.K_UP] and player_y > 0:
        player_y -= SPEED
    if keys[pygame.K_DOWN] and player_y < HEIGHT - player_size:
        player_y += SPEED

    # Draw Everything
    screen.fill(WHITE)
    pygame.draw.rect(screen, BLUE, (player_x, player_y, player_size, player_size))
    pygame.display.update()

# Quit Pygame
pygame.quit()