import pygame
import random

# Initialize Pygame
pygame.init()

# Game Constants
WIDTH, HEIGHT = 800, 600
WHITE = (255, 255, 255)
BLACK = (0, 0, 0)
GREEN = (34, 139, 34)

# Set up display
screen = pygame.display.set_mode((WIDTH, HEIGHT))
pygame.display.set_caption("Blackjack Game")

# Load Fonts
font = pygame.font.Font(None, 36)

# Card deck where Ace (11) is included
cards = [11, 2, 3, 4, 5, 6, 7, 8, 9, 10, 10, 10, 10]

# Function to draw a card
def draw_card():
    return random.choice(cards)

# Function to calculate sum of cards with Ace handling
def calculate_sum(hand):
    total = sum(hand)
    if 11 in hand and total > 21:
        hand[hand.index(11)] = 1  # Convert Ace from 11 to 1 if total exceeds 21
        total = sum(hand)
    return total

# Draw a button
def draw_button(text, x, y, w, h):
    pygame.draw.rect(screen, BLACK, (x, y, w, h), border_radius=10)
    text_surface = font.render(text, True, WHITE)
    screen.blit(text_surface, (x + 10, y + 10))

# Function to display cards
def draw_cards(cards, x, y):
    for i, card in enumerate(cards):
        pygame.draw.rect(screen, WHITE, (x + i * 60, y, 50, 70), border_radius=10)
        card_text = font.render(str(card), True, BLACK)
        screen.blit(card_text, (x + i * 60 + 15, y + 20))

# Game Variables
player_cards = [draw_card(), draw_card()]
computer_cards = [draw_card(), draw_card()]
player_turn = True
game_over = False
message = ""

# Main Game Loop
running = True
while running:
    screen.fill(GREEN)

    # Display cards
    draw_cards(player_cards, 100, 400)
    draw_cards(computer_cards, 100, 100 if game_over else 100)  # Hide computer's second card unless game over

    # Show sum of cards
    player_sum = calculate_sum(player_cards)
    computer_sum = calculate_sum(computer_cards) if game_over else "?"
    screen.blit(font.render(f"Your Sum: {player_sum}", True, WHITE), (100, 500))
    screen.blit(font.render(f"Computer Sum: {computer_sum}", True, WHITE), (100, 200))

    # Draw Buttons
    if not game_over:
        draw_button("Hit", 600, 400, 100, 50)
        draw_button("Stand", 600, 460, 100, 50)
    
    # Display message
    screen.blit(font.render(message, True, WHITE), (100, 550))

    pygame.display.flip()

    # Event handling
    for event in pygame.event.get():
        if event.type == pygame.QUIT:
            running = False
        elif event.type == pygame.MOUSEBUTTONDOWN and player_turn:
            x, y = event.pos

            # "Hit" button clicked
            if 600 <= x <= 700 and 400 <= y <= 450:
                player_cards.append(draw_card())
                player_sum = calculate_sum(player_cards)
                if player_sum > 21:
                    message = "You Busted! You Lose!"
                    game_over = True

            # "Stand" button clicked
            elif 600 <= x <= 700 and 460 <= y <= 510:
                player_turn = False
                while calculate_sum(computer_cards) < 17:
                    computer_cards.append(draw_card())

                # Final sum calculations
                player_sum = calculate_sum(player_cards)
                computer_sum = calculate_sum(computer_cards)
                
                # Determine winner
                if computer_sum > 21:
                    message = "Computer Busted! You Win!"
                elif computer_sum > player_sum:
                    message = "Computer Wins!"
                elif computer_sum < player_sum:
                    message = "You Win!"
                else:
                    message = "It's a Draw!"
                
                game_over = True

pygame.quit()