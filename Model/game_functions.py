import pygame
import math
import numpy as np
from Box2D import b2Vec2, b2World, b2PolygonShape, b2FixtureDef, b2BodyDef, b2_dynamicBody
from robot import Robot
from constants import *

def create_walls(world):
    walls = [
        # Right wall
        [((FIELD_MARGIN + FIELD_WIDTH - BOUNDARY_THICKNESS)/PPM, FIELD_MARGIN/PPM), 
         ((FIELD_MARGIN + FIELD_WIDTH)/PPM, FIELD_MARGIN/PPM),
         ((FIELD_MARGIN + FIELD_WIDTH)/PPM, (FIELD_MARGIN + FIELD_HEIGHT)/PPM),
         ((FIELD_MARGIN + FIELD_WIDTH - BOUNDARY_THICKNESS)/PPM, (FIELD_MARGIN + FIELD_HEIGHT)/PPM)],
        # Bottom wall
        [(FIELD_MARGIN/PPM, (FIELD_MARGIN + FIELD_HEIGHT - BOUNDARY_THICKNESS)/PPM),
         ((FIELD_MARGIN + FIELD_WIDTH)/PPM, (FIELD_MARGIN + FIELD_HEIGHT - BOUNDARY_THICKNESS)/PPM),
         ((FIELD_MARGIN + FIELD_WIDTH)/PPM, (FIELD_MARGIN + FIELD_HEIGHT)/PPM),
         (FIELD_MARGIN/PPM, (FIELD_MARGIN + FIELD_HEIGHT)/PPM)],
        # Left wall
        [(FIELD_MARGIN/PPM, FIELD_MARGIN/PPM),
         ((FIELD_MARGIN + BOUNDARY_THICKNESS)/PPM, FIELD_MARGIN/PPM),
         ((FIELD_MARGIN + BOUNDARY_THICKNESS)/PPM, (FIELD_MARGIN + FIELD_HEIGHT)/PPM),
         (FIELD_MARGIN/PPM, (FIELD_MARGIN + FIELD_HEIGHT)/PPM)]
    ]
    
    for wall_vertices in walls:
        body = world.CreateStaticBody(
            shapes=b2PolygonShape(vertices=wall_vertices)
        )
        body.fixtures[0].friction = 0.7
        body.fixtures[0].restitution = 0.1

def check_win_condition(robot_blue, game_start_time):
    # Get the top-most point of the blue robot
    blue_top = robot_blue.body.position.y * PPM - (robot_blue.size * PPM / 2)
    
    # Check if the entire robot has crossed above the top field boundary
    if blue_top < FIELD_MARGIN:
        return "BLUE WINS!"
    
    # Check if time is up
    if pygame.time.get_ticks() - game_start_time >= GAME_DURATION * 1000:
        return "RED WINS!"
    
    return None

def draw_timer(screen, game_start_time):
    elapsed_time = (pygame.time.get_ticks() - game_start_time) / 1000
    remaining_time = max(0, GAME_DURATION - elapsed_time)
    
    font = pygame.font.Font(None, 74)
    timer_text = font.render(f"{remaining_time:.1f}", True, (0, 0, 0))  # Change from (255, 255, 255) to black
    timer_rect = timer_text.get_rect(center=(WINDOW_WIDTH // 2, 30))
    screen.blit(timer_text, timer_rect)
    
def draw_game_over(screen, message):
    font = pygame.font.Font(None, 100)
    text = font.render(message, True, (255, 255, 0))
    rect = text.get_rect(center=(WINDOW_WIDTH // 2, WINDOW_HEIGHT // 2))
    
    # Draw semi-transparent background
    s = pygame.Surface((WINDOW_WIDTH, WINDOW_HEIGHT))
    s.set_alpha(128)
    s.fill((0, 0, 0))
    screen.blit(s, (0, 0))
    
    screen.blit(text, rect)

def draw_goal_area(screen):
    # Create a surface for the goal area
    goal_surface = pygame.Surface((FIELD_WIDTH, GOAL_HEIGHT))
    goal_surface.fill(GOAL_COLOR_1)
    
    # Draw diagonal stripes
    num_stripes = int(FIELD_WIDTH / STRIPE_WIDTH) + 2
    points = []
    
    for i in range(num_stripes):
        # Calculate stripe positions with movement
        x = i * STRIPE_WIDTH - STRIPE_WIDTH
        points.append([(x, 0), (x + GOAL_HEIGHT, GOAL_HEIGHT),
                      (x + GOAL_HEIGHT + STRIPE_WIDTH/2, GOAL_HEIGHT),
                      (x + STRIPE_WIDTH/2, 0)])
    
    # Draw the stripes
    for stripe in points:
        pygame.draw.polygon(goal_surface, GOAL_COLOR_2, stripe)
    
    # Draw the goal area on the screen
    screen.blit(goal_surface, (FIELD_MARGIN, FIELD_MARGIN))
    
    # Draw thin lines at top and bottom of goal area
    pygame.draw.line(screen, BOUNDARY_COLOR,
                    (FIELD_MARGIN, FIELD_MARGIN + GOAL_HEIGHT),
                    (FIELD_MARGIN + FIELD_WIDTH, FIELD_MARGIN + GOAL_HEIGHT),
                    2)
