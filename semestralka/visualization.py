import pygame
import numpy as np

def pendulum_visualization(x_history, t_grid, u_history=None, d_array=None):

    if isinstance(x_history, list):
        x_array = np.array(x_history)
    else:
        x_array = x_history

    pygame.init()
    WIDTH, HEIGHT = 1200, 720
    screen = pygame.display.set_mode((WIDTH, HEIGHT))
    pygame.display.set_caption("Inverted Pendulum - Simulation Replay")
    clock = pygame.time.Clock()

    # Colors
    BLACK = (0, 0, 0)
    WHITE = (255, 255, 255)
    RED   = (220, 20, 60)
    BLUE  = (30, 144, 255)
    GRAY  = (70, 70, 70)
    GREEN = (0, 220, 0)
    BUTTON_COLOR = (0, 160, 0)
    BUTTON_HOVER = (0, 200, 0)

    # Scaling
    SCALE = 320.0
    CART_WIDTH, CART_HEIGHT = 90, 35
    PENDULUM_LENGTH = 1.0
    PENDULUM_THICKNESS = 14

    font = pygame.font.SysFont("consolas", 22)
    small_font = pygame.font.SysFont("consolas", 18)

    def draw_frame(i):
        state = x_array[i]
        cart_x = state[0]
        theta = state[2]

        screen.fill(BLACK)

        origin_x = WIDTH // 2
        origin_y = HEIGHT // 2 + 80

        # Rails
        pygame.draw.line(screen, GRAY, (100, origin_y + 40), (WIDTH - 100, origin_y + 40), 10)

        # Cart
        cart_screen_x = origin_x + int(cart_x * SCALE)
        pygame.draw.rect(screen, BLUE,
                         (cart_screen_x - CART_WIDTH//2, origin_y - CART_HEIGHT//2,
                          CART_WIDTH, CART_HEIGHT), border_radius=8)

        # Pendulum drawing (theta ≈ π = upright)
        draw_angle = np.pi - theta
        dx = PENDULUM_LENGTH * np.sin(draw_angle) * SCALE
        dy = PENDULUM_LENGTH * np.cos(draw_angle) * SCALE

        pend_x = cart_screen_x + dx
        pend_y = origin_y - dy

        pygame.draw.line(screen, RED, (cart_screen_x, origin_y), (pend_x, pend_y), PENDULUM_THICKNESS)
        pygame.draw.circle(screen, WHITE, (int(pend_x), int(pend_y)), 18)

        # Upright reference
        pygame.draw.circle(screen, GREEN, (cart_screen_x, origin_y - int(PENDULUM_LENGTH * SCALE)), 8, 2)

        # Info
        t = t_grid[min(i, len(t_grid)-1)]
        u_val = u_history[i] if u_history is not None else 0.0
        d_val = d_array[i] if d_array is not None else 0.0

        texts = [
            f"Time: {t:.2f} s",
            f"Cart Position: {cart_x:.3f} m",
            f"Angle: {np.degrees(theta):.1f}°  (180° = upright)",
            f"Control: {u_val:+.2f} N",
            f"Disturbance: {d_val:+.2f} N"
        ]

        for idx, text in enumerate(texts):
            surf = font.render(text, True, WHITE)
            screen.blit(surf, (30, 30 + idx * 34))

        # Progress bar
        progress = i / (len(x_array) - 1) if len(x_array) > 1 else 0
        pygame.draw.rect(screen, (40, 40, 40), (250, 20, 700, 8))
        pygame.draw.rect(screen, RED, (250, 20, int(700 * progress), 8))

    # ====================== MAIN ANIMATION ======================
    running = True
    current_frame = 0
    playing = True

    while running:
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                running = False

            elif event.type == pygame.MOUSEBUTTONDOWN and not playing:
                mx, my = pygame.mouse.get_pos()
                if 480 <= mx <= 720 and 380 <= my <= 460:   # Replay button
                    current_frame = 0
                    playing = True

        if playing:
            if current_frame < len(x_array):
                draw_frame(current_frame)
                current_frame += 1
                clock.tick(75)          # ← change this number for speed (60 = slower, 100 = faster)
            else:
                playing = False         # finished playing

        else:
            # Show final frame + Replay button
            draw_frame(len(x_array) - 1)

            # Replay Button
            button_rect = pygame.Rect(480, 380, 240, 80)
            mouse_pos = pygame.mouse.get_pos()
            color = BUTTON_HOVER if button_rect.collidepoint(mouse_pos) else BUTTON_COLOR

            pygame.draw.rect(screen, color, button_rect, border_radius=12)
            pygame.draw.rect(screen, WHITE, button_rect, width=4, border_radius=12)

            txt = font.render("REPLAY", True, WHITE)
            screen.blit(txt, txt.get_rect(center=button_rect.center))

            help_txt = small_font.render("Click REPLAY to watch the animation again", True, (200, 200, 200))
            screen.blit(help_txt, help_txt.get_rect(centerx=WIDTH//2, y=500))

            clock.tick(30)

        pygame.display.flip()

    pygame.quit()