import pygame
import math
import numpy as np
from Box2D import b2Vec2, b2World, b2PolygonShape, b2FixtureDef, b2BodyDef, b2_dynamicBody
from robot import Robot
from constants import *
from game_functions import *

def main():
    pygame.init()
    screen = pygame.display.set_mode((WINDOW_WIDTH, WINDOW_HEIGHT))
    pygame.display.set_caption("FRC Swerve Drive Game")
    clock = pygame.time.Clock()
    
    # Create Box2D world with no gravity
    world = b2World(gravity=(0, 0), doSleep=True)
    
    # Create walls
    create_walls(world)
    
    # Create robots with new starting positions
    robot_red = Robot(WINDOW_WIDTH//2, RED_START_Y, (200, 30, 30), world)
    robot_blue = Robot(WINDOW_WIDTH//2, BLUE_START_Y, (30, 30, 200), world)
    
    game_start_time = pygame.time.get_ticks()
    game_over = False
    winner_message = None
    
    running = True
    while running:
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                running = False
            elif event.type == pygame.KEYDOWN:
                if event.key == pygame.K_ESCAPE:
                    running = False
                elif event.key == pygame.K_r and game_over:
                    # Reset game
                    game_over = False
                    game_start_time = pygame.time.get_ticks()
                    
                    # Reset red robot position, velocity, and orientation
                    robot_red.body.position = (WINDOW_WIDTH/2/PPM, RED_START_Y/PPM)
                    robot_red.body.linearVelocity = b2Vec2(0, 0)
                    robot_red.body.angularVelocity = 0
                    robot_red.body.angle = 0  # Reset angle to 0
                    
                    # Reset blue robot position, velocity, and orientation
                    robot_blue.body.position = (WINDOW_WIDTH/2/PPM, BLUE_START_Y/PPM)
                    robot_blue.body.linearVelocity = b2Vec2(0, 0)
                    robot_blue.body.angularVelocity = 0
                    robot_blue.body.angle = 0  # Reset angle to 0
                    
                    # Reset any accumulated velocities in the Robot class
                    robot_red.current_vx = 0
                    robot_red.current_vy = 0
                    robot_red.current_omega = 0
                    robot_red.target_vx = 0
                    robot_red.target_vy = 0
                    robot_red.target_omega = 0
                    
                    robot_blue.current_vx = 0
                    robot_blue.current_vy = 0
                    robot_blue.current_omega = 0
                    robot_blue.target_vx = 0
                    robot_blue.target_vy = 0
                    robot_blue.target_omega = 0
                    
                    # Reset swerve modules
                    for module in robot_red.modules:
                        module.wheel_angle = 0
                        module.wheel_speed = 0
                        module.last_wheel_angle = 0
                    
                    for module in robot_blue.modules:
                        module.wheel_angle = 0
                        module.wheel_speed = 0
                        module.last_wheel_angle = 0
        
        # Handle input and physics only if game is not over
        if not game_over:
            keys = pygame.key.get_pressed()
            
            # Handle input for red robot
            red_vx = (keys[pygame.K_d] - keys[pygame.K_a]) * robot_red.max_speed
            red_vy = (keys[pygame.K_s] - keys[pygame.K_w]) * robot_red.max_speed
            red_omega = (keys[pygame.K_e] - keys[pygame.K_q]) * robot_red.max_omega
            
            # Handle input for blue robot
            blue_vx = (keys[pygame.K_RIGHT] - keys[pygame.K_LEFT]) * robot_blue.max_speed
            blue_vy = (keys[pygame.K_DOWN] - keys[pygame.K_UP]) * robot_blue.max_speed
            blue_omega = (keys[pygame.K_RIGHTBRACKET] - keys[pygame.K_LEFTBRACKET]) * robot_blue.max_omega
            
            # Update robots
            dt = 1/60.0
            robot_red.apply_movement(red_vx, red_vy, red_omega, field_oriented=True, dt=dt)
            robot_blue.apply_movement(blue_vx, blue_vy, blue_omega, field_oriented=True, dt=dt)
            
            # Step physics world
            world.Step(dt, 8, 3)
            
            # Check win condition
            winner_message = check_win_condition(robot_blue, game_start_time)
            if winner_message:
                game_over = True
        
        # Draw
        screen.fill((255, 255, 255))
        
        # Draw field
        pygame.draw.rect(screen, FIELD_COLOR, 
                        (FIELD_MARGIN, FIELD_MARGIN, 
                         FIELD_WIDTH, FIELD_HEIGHT))
        
        # Draw goal area
        draw_goal_area(screen)
        
        # Draw remaining walls
        pygame.draw.line(screen, BOUNDARY_COLOR,
                        (FIELD_MARGIN, FIELD_MARGIN + FIELD_HEIGHT),
                        (FIELD_MARGIN + FIELD_WIDTH, FIELD_MARGIN + FIELD_HEIGHT),
                        BOUNDARY_THICKNESS)
        pygame.draw.line(screen, BOUNDARY_COLOR,
                        (FIELD_MARGIN, FIELD_MARGIN),
                        (FIELD_MARGIN, FIELD_MARGIN + FIELD_HEIGHT),
                        BOUNDARY_THICKNESS)
        pygame.draw.line(screen, BOUNDARY_COLOR,
                        (FIELD_MARGIN + FIELD_WIDTH, FIELD_MARGIN),
                        (FIELD_MARGIN + FIELD_WIDTH, FIELD_MARGIN + FIELD_HEIGHT),
                        BOUNDARY_THICKNESS)
        
        # Draw robots
        robot_red.draw(screen)
        robot_blue.draw(screen)
        
        # Draw timer
        draw_timer(screen, game_start_time)
        
        # Draw game over message if game is over
        if game_over:
            draw_game_over(screen, winner_message)
            font = pygame.font.Font(None, 36)
            restart_text = font.render("Press R to Restart", True, (255, 255, 0))
            restart_rect = restart_text.get_rect(center=(WINDOW_WIDTH // 2, WINDOW_HEIGHT // 2 + 50))
            screen.blit(restart_text, restart_rect)
        
        pygame.display.flip()
        clock.tick(60)
    
    pygame.quit()

if __name__ == "__main__":
    main()