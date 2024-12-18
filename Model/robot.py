import pygame
import math
import numpy as np
from Box2D import b2Vec2, b2World, b2PolygonShape, b2FixtureDef, b2BodyDef, b2_dynamicBody
from constants import *

class SwerveModule:
    def __init__(self, x_offset, y_offset):
        self.x_offset = x_offset
        self.y_offset = y_offset
        
        # Physical properties
        self.max_module_speed = 4.42  # ~14.5 ft/s for MK4i
        self.max_rotation_speed = math.pi * 2
        
        # Module state
        self.wheel_angle = 0
        self.wheel_speed = 0
        self.last_wheel_angle = 0
        
        # Module characteristics
        self.drive_motor_max_torque = 1.5
        self.rotation_motor_max_torque = 1.0
        self.wheel_inertia = 0.001
        
        # Add visualization properties
        self.module_color = (100, 100, 100)
        self.direction_color = (255, 255, 0)  # Yellow for better visibility
        self.speed_indicator_color = (0, 255, 0)  # Green for speed
        
        # Physical properties
        self.wheel_width = 4 * PIXELS_PER_INCH
        self.wheel_height = 2 * PIXELS_PER_INCH
        self.circle_radius = 10  # Base circle size


    def optimize_angle(self, target_angle, current_speed):
        # Normalize angles to -pi to pi
        target = ((target_angle + math.pi) % (2 * math.pi)) - math.pi
        current = ((self.wheel_angle + math.pi) % (2 * math.pi)) - math.pi
        
        # Find the shortest path
        diff = target - current
        if abs(diff) > math.pi:
            diff = diff - (2 * math.pi) * (diff / abs(diff))
        
        # If turn is more than 90 degrees, reverse direction and flip angle
        if abs(diff) > math.pi/2:
            if target > 0:
                target -= math.pi
            else:
                target += math.pi
            current_speed = -current_speed
        
        # Store the optimized angle
        self.last_wheel_angle = target
        
        return target, current_speed
    
    def update(self, dt):
        # Get current velocity
        vel = self.body.linearVelocity
        velocity = [vel.x, vel.y]
        
        # Calculate forces from friction model
        forces = self.friction_model.calculate_wheel_forces(
            velocity,
            self.wheel_angle,
            self.normal_force
        )
        
        # Apply forces
        force_angle = self.wheel_angle + math.atan2(vel.y, vel.x)
        force_x = forces['friction_force'] * math.cos(force_angle)
        force_y = forces['friction_force'] * math.sin(force_angle)
        
        # Apply rolling resistance opposite to velocity
        if vel.length > 0:
            resistance_x = -forces['rolling_resistance'] * vel.x / vel.length
            resistance_y = -forces['rolling_resistance'] * vel.y / vel.length
            force_x += resistance_x
            force_y += resistance_y
        
        self.body.ApplyForce(b2Vec2(force_x, force_y), self.body.worldCenter, True)

    def draw(self, screen, body):
        # Get position relative to robot body
        angle = body.angle
        cos_h = math.cos(angle)
        sin_h = math.sin(angle)
        
        # Convert module offsets from meters to pixels
        x_offset_pixels = self.x_offset * PPM
        y_offset_pixels = self.y_offset * PPM
        
        pos = body.position
        # Calculate module position
        x = pos.x * PPM + (x_offset_pixels * cos_h - y_offset_pixels * sin_h)
        y = pos.y * PPM + (x_offset_pixels * sin_h + y_offset_pixels * cos_h)
        
        # Draw module base (gray circle)
        pygame.draw.circle(screen, (100, 100, 100), (int(x), int(y)), self.circle_radius)
        
        # Calculate total angle for the wheel
        total_angle = angle + self.wheel_angle
        
        # Draw wheel rectangle
        wheel_points = []
        half_width = self.wheel_width / 2
        half_height = self.wheel_height / 2
        
        # Calculate wheel corners
        corners = [
            (-half_width, -half_height),
            (half_width, -half_height),
            (half_width, half_height),
            (-half_width, half_height)
        ]
        
        # Transform wheel corners
        for wx, wy in corners:
            # Rotate around wheel center
            rotated_x = wx * math.cos(total_angle) - wy * math.sin(total_angle)
            rotated_y = wx * math.sin(total_angle) + wy * math.cos(total_angle)
            # Translate to module position
            wheel_points.append((
                int(x + rotated_x),
                int(y + rotated_y)
            ))
        
        # Draw wheel with speed-based color
        speed_intensity = min(255, max(0, int(255 * abs(self.wheel_speed))))
        wheel_color = (0, speed_intensity, 0)
        pygame.draw.polygon(screen, wheel_color, wheel_points)


class Robot:
    def __init__(self, x, y, color, world, size_multiplier=1.0):
        self.color = color
        
        # Determine if this is the red robot based on color
        is_red = color[0] > color[2]  # Red has higher R value than B value
        
        # Robot dimensions in pixels (30 inches * 2 pixels per inch)
        self.robot_width_pixels = 30 * PIXELS_PER_INCH  # 60 pixels
        self.robot_length_pixels = 30 * PIXELS_PER_INCH  # 60 pixels
        
        # Module spacing in pixels (20 inches * 2 pixels per inch)
        module_spacing_pixels = 20 * PIXELS_PER_INCH  # 40 pixels
        
        # Convert to meters for Box2D
        self.robot_width = (self.robot_width_pixels / PIXELS_PER_INCH) * INCHES_TO_METERS
        self.robot_length = (self.robot_length_pixels / PIXELS_PER_INCH) * INCHES_TO_METERS
        module_spacing = (module_spacing_pixels / PIXELS_PER_INCH) * INCHES_TO_METERS
        
        # Create Box2D body
        bodyDef = b2BodyDef()
        bodyDef.type = b2_dynamicBody
        bodyDef.position = (x / PPM, y / PPM)
        bodyDef.linearDamping = 0.85
        bodyDef.angularDamping = 0.95
        self.body = world.CreateBody(bodyDef)
        
        # Create box shape - using robot dimensions in meters
        shape = b2PolygonShape(box=(self.robot_width/2, self.robot_length/2))
        
        # Adjust physics properties based on robot color
        density = 1.5 if is_red else 1.0  # Red robot is 50% heavier
        friction = 0.9 if is_red else 0.8  # Red robot has better traction
        
        # Create fixture with adjusted properties
        fixtureDef = b2FixtureDef(
            shape=shape,
            density=density,
            friction=friction,
            restitution=0.1,
            userData={'type': 'robot'}
        )
        self.body.CreateFixture(fixtureDef)
        
        # Swerve modules setup
        module_offset = module_spacing / 2
        self.modules = [
            SwerveModule(-module_offset, -module_offset),  # Back Left
            SwerveModule(module_offset, -module_offset),   # Back Right
            SwerveModule(-module_offset, module_offset),   # Front Left
            SwerveModule(module_offset, module_offset)     # Front Right
        ]
        
        # Movement properties with realistic MK4i values
        self.max_speed = 4.42  # ~14.5 ft/s for MK4i
        self.max_omega = math.pi * 2  # Maximum rotation speed
        
        # Base acceleration values
        base_acceleration = 8.0  # About 2g acceleration
        base_deceleration = 10.0
        
        # Acceleration parameters - adjusted for red robot
        self.linear_acceleration = base_acceleration * (1.2 if is_red else 1.0)  # Red accelerates 20% faster
        self.linear_deceleration = base_deceleration * (1.2 if is_red else 1.0)
        self.angular_acceleration = math.pi * 2 * (1.2 if is_red else 1.0)
        self.angular_deceleration = math.pi * 2.4 * (1.2 if is_red else 1.0)
        
        # Current velocity tracking
        self.current_vx = 0
        self.current_vy = 0
        self.current_omega = 0
        
        # Target velocity tracking
        self.target_vx = 0
        self.target_vy = 0
        self.target_omega = 0
        
        # Physics properties
        self.traction_factor = 1.0
        self.weight_transfer = 0.0
        self.base_friction = friction  # Use the color-based friction value
        
        # Store size for drawing
        self.size = self.robot_width  # Size in meters



    def _apply_acceleration(self, current, target, accel, decel, dt):
        if abs(target) > abs(current):
            # Accelerating
            if target > current:
                current = min(target, current + accel * dt)
            else:
                current = max(target, current - accel * dt)
        else:
            # Decelerating
            if current > target:
                current = max(target, current - decel * dt)
            else:
                current = min(target, current + decel * dt)
        return current

    def update_physics(self, dt):
        # Calculate acceleration
        current_velocity = b2Vec2(self.current_vx, self.current_vy)
        acceleration = (current_velocity - self.body.linearVelocity) / dt
        
        # Calculate weight transfer
        accel_magnitude = acceleration.length
        self.weight_transfer = min(accel_magnitude / self.linear_acceleration, 1.0)
        
        # Calculate traction factor based on speed and weight transfer
        speed = current_velocity.length
        base_traction = 1.0 - (speed / self.max_speed) * 0.3
        self.traction_factor = base_traction * (1.0 - self.weight_transfer * 0.2)
        self.traction_factor = max(0.3, min(1.0, self.traction_factor))
        
        # Update fixture friction based on conditions
        for fixture in self.body.fixtures:
            fixture.friction = self.base_friction * self.traction_factor

    def _update_modules(self, vx, vy, omega):
        for module in self.modules:
            # Calculate velocity contribution from rotation
            rot_x = -module.y_offset * omega
            rot_y = module.x_offset * omega
            
            # Combine translation and rotation velocities
            final_vx = vx + rot_x
            final_vy = vy + rot_y
            
            # Calculate desired angle and speed
            desired_speed = math.sqrt(final_vx**2 + final_vy**2)
            
            # Update module angle even during pure rotation
            if desired_speed > 0.001:  # Lower threshold for better rotation visualization
                desired_angle = math.atan2(final_vy, final_vx)
                
                # Optimize wheel angle and direction
                optimized_angle, optimized_speed = module.optimize_angle(
                    desired_angle, 
                    desired_speed
                )
                
                # Apply maximum speed limit
                optimized_speed = min(optimized_speed, module.max_module_speed)
                
                # Apply traction factor
                optimized_speed *= self.traction_factor
                
                # Update module state
                module.wheel_angle = optimized_angle
                module.wheel_speed = optimized_speed / self.max_speed
            else:
                # During pure rotation, point wheels tangent to rotation
                desired_angle = math.atan2(rot_y, rot_x)
                if abs(omega) > 0.001:
                    module.wheel_angle = desired_angle
                    # Set speed proportional to rotation
                    module.wheel_speed = (abs(omega) / self.max_omega) * self.traction_factor
                else:
                    module.wheel_speed = 0
    def apply_movement(self, vx, vy, omega, field_oriented=True, dt=1/60.0):
        # Store target omega
        self.target_omega = omega

        if field_oriented:
            # Store field-oriented velocities as targets
            self.target_vx = vx
            self.target_vy = vy
            
            # Apply acceleration to field-oriented velocities
            self.current_vx = self._apply_acceleration(
                self.current_vx,
                self.target_vx,
                self.linear_acceleration,
                self.linear_deceleration,
                dt
            )
            
            self.current_vy = self._apply_acceleration(
                self.current_vy,
                self.target_vy,
                self.linear_acceleration,
                self.linear_deceleration,
                dt
            )

            # Apply acceleration to angular velocity
            self.current_omega = self._apply_acceleration(
                self.current_omega,
                self.target_omega,
                self.angular_acceleration,
                self.angular_deceleration,
                dt
            )

            # Update physics before applying velocities
            self.update_physics(dt)
            
            # Apply traction-affected velocities
            field_velocity = b2Vec2(self.current_vx, self.current_vy) * self.traction_factor
            self.body.linearVelocity = field_velocity
            self.body.angularVelocity = self.current_omega * self.traction_factor

            # Convert field velocities to robot-frame for swerve modules
            angle = self.body.angle
            cos_h = math.cos(angle)
            sin_h = math.sin(angle)
            robot_vx = self.current_vx * cos_h + self.current_vy * sin_h
            robot_vy = -self.current_vx * sin_h + self.current_vy * cos_h

            # Update swerve modules with robot-frame velocities
            self._update_modules(robot_vx, robot_vy, self.current_omega)
        else:
            # Non-field-oriented control
            self.target_vx = vx
            self.target_vy = vy

            self.current_vx = self._apply_acceleration(
                self.current_vx,
                self.target_vx,
                self.linear_acceleration,
                self.linear_deceleration,
                dt
            )
            
            self.current_vy = self._apply_acceleration(
                self.current_vy,
                self.target_vy,
                self.linear_acceleration,
                self.linear_deceleration,
                dt
            )

            self.current_omega = self._apply_acceleration(
                self.current_omega,
                self.target_omega,
                self.angular_acceleration,
                self.angular_deceleration,
                dt
            )

            self.update_physics(dt)
            
            self.body.linearVelocity = b2Vec2(self.current_vx, self.current_vy) * self.traction_factor
            self.body.angularVelocity = self.current_omega * self.traction_factor
            
            self._update_modules(self.current_vx, self.current_vy, self.current_omega)

    def draw(self, screen):
        # Calculate base position
        pos = self.body.position * PPM
        angle = self.body.angle
        half_size = (self.size * PPM) / 2
        corner_radius = 21  # Match the swerve module circle radius
        
        # Create a surface for the rectangle with rounded corners
        surface_size = int(half_size * 2)
        surface = pygame.Surface((surface_size, surface_size), pygame.SRCALPHA)
        
        # Draw the rounded rectangle on the surface
        rect = pygame.Rect(0, 0, surface_size, surface_size)
        pygame.draw.rect(surface, self.color, rect, border_radius=corner_radius)
        
        # Rotate the surface
        rotated_surface = pygame.transform.rotate(surface, -math.degrees(angle))
        rect = rotated_surface.get_rect(center=(int(pos.x), int(pos.y)))
        
        # Draw the rotated rectangle
        screen.blit(rotated_surface, rect)
        
        # Draw direction indicator
        end_x = pos.x + math.cos(angle) * (self.size * PPM) // 2
        end_y = pos.y + math.sin(angle) * (self.size * PPM) // 2
        pygame.draw.line(screen, (255, 255, 255),
                        (int(pos.x), int(pos.y)),
                        (int(end_x), int(end_y)), 3)
        
        # Draw swerve modules
        for module in self.modules:
            module.draw(screen, self.body)
